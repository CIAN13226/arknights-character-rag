# -*- coding: utf-8 -*-
"""RAG 可观测性 Level 1：给每次请求发一个 trace_id，把各阶段记全，一次请求落一行。

为什么需要它：
    llm_log.jsonl 只记录了生成阶段（754 条），有三个致命缺口：
      1. 没有 trace_id —— 一个用户问题可能触发 2-4 次模型调用（重试、逐块重试、
         自检修正），这些记录分散在多行里，事后无法归并。
         连"总共被问了多少个问题"都数不出来。
      2. 没有检索结果 —— 看到一句错答案时，无法判断是"检索没找到"还是"生成没用上"。
         而 RAG 答错里占多数的是检索问题。
      3. 没有路由决策 —— 走错分支（该结构化却走了向量）时，向量检索也能返回
         一堆"看着相关"的块，答案勉强像样但不对，不记路由就永远发现不了。

设计上的三个取舍：
    ① 一次请求写一行，不是一阶段写一行
       否则又要 join，而 join 的键正是缺失的 trace_id。
    ② 只记 chunk 的引用和预览，不记全文
       chunks_lore.json 有 10.8 MB，每次请求把命中原文写进日志会迅速撑爆日志。
       记 chunk_id + 分数 + 前 100 字，需要全文时按 id 回原库查。
    ③ 记"决策理由"，不只记"决策结果"
       出错时要复盘的是"为什么这么决定"。

用法：
    python 9_observe.py --last 5              最近 5 次请求的链路
    python 9_observe.py --trace <trace_id>    看一次完整链路
    python 9_observe.py --suspicious          按信号筛可疑记录
    python 9_observe.py --stats               聚合指标（延迟分位、空检索率等）
"""
import os
import sys
import json
import time
import uuid
import datetime
import contextvars
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
TRACE_FILE = os.path.join(BASE, "logs", "traces.jsonl")

# ---- 可疑判定阈值（都可调，写在这里是为了让判定标准可见）----
LOW_TOP1 = 0.60          # top1 相似度低于此值，疑似检索质量不足
MANY_LLM_CALLS = 3       # 一次请求调模型 >= 3 次，说明首次生成质量差
LONG_PROMPT = 3500       # prompt token 超过此值，可能被上下文长度截断
BIG_CUT = 3              # 裁掉 >= 3 个块，可能把含答案的块裁了

_current = contextvars.ContextVar("current_trace", default=None)


# ---------------- Trace 对象 ----------------

class Trace:
    """一次请求的完整记录。各阶段往里写，最后 finish() 落一行 JSONL。"""

    def __init__(self, question, **meta):
        now = datetime.datetime.now()
        self.trace_id = f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        self.t0 = time.time()
        self.rec = {
            "trace_id": self.trace_id,
            "ts": now.isoformat(timespec="milliseconds"),
            "question": (question or "")[:500],
            "elapsed_ms": None,
            "error": None,
            "flags": [],
        }
        if meta:
            self.rec["meta"] = meta
        self._llm = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                     "elapsed_ms": 0, "errors": 0}

    def stage(self, name, **fields):
        """记录一个阶段。同名阶段重复写入会合并（重试场景保留最后一次 + 次数）。"""
        old = self.rec.get(name)
        if isinstance(old, dict):
            old.update(fields)
        else:
            self.rec[name] = dict(fields)

    def set(self, key, value):
        self.rec[key] = value

    def llm_call(self, elapsed_ms=0, prompt_tokens=None, output_tokens=None, ok=True):
        """由 call_llm 调用，累计本 trace 里的模型调用。"""
        self._llm["calls"] += 1
        self._llm["elapsed_ms"] += int(elapsed_ms or 0)
        if prompt_tokens:
            self._llm["prompt_tokens"] += int(prompt_tokens)
        if output_tokens:
            self._llm["output_tokens"] += int(output_tokens)
        if not ok:
            self._llm["errors"] += 1

    def _flags(self):
        f = []
        r = self.rec
        ret = r.get("retrieval") or {}
        if ret.get("empty"):
            f.append("空检索")
        top1 = ret.get("top1")
        if top1 is not None and top1 < LOW_TOP1:
            f.append(f"top1偏低({top1:.2f})")
        # 注意：只有「模型在有高分资料的情况下拒答」才算值得看一眼。
        # 分数低于阈值时的拒答是设计好的行为，不该报可疑。
        # 这里只描述信号（拒答了、且分数达标），不替人下"资料里有答案"的结论——
        # 因为没有标准答案时机器判断不了，下结论会误导排查方向。
        ref = r.get("refusal") or {}
        if ref.get("reason") == "model_refused":
            f.append("拒答但分数达标")
        if self._llm["calls"] >= MANY_LLM_CALLS:
            f.append(f"模型调用{self._llm['calls']}次")
        if (r.get("cut") or {}).get("dropped", 0) >= BIG_CUT:
            f.append(f"裁掉{r['cut']['dropped']}块")
        if self._llm["prompt_tokens"] >= LONG_PROMPT:
            f.append(f"prompt{self._llm['prompt_tokens']}token")
        if self._llm["errors"]:
            f.append("模型调用失败")
        if r.get("error"):
            f.append("链路异常")
        return f

    def finish(self, answer=None, error=None):
        """补齐收尾字段并落盘。可重复调用，只有第一次生效。"""
        if self.rec.get("elapsed_ms") is not None:
            return self.trace_id
        self.rec["elapsed_ms"] = int((time.time() - self.t0) * 1000)
        if self._llm["calls"] or self._llm["elapsed_ms"]:
            self.rec["llm"] = dict(self._llm)
        if answer is not None:
            self.rec["answer"] = str(answer)[:1000]
            self.rec["answer_preview"] = str(answer)[:120].replace("\n", " ")
        if error:
            self.rec["error"] = str(error)[:500]
        self.rec["flags"] = self._flags()
        _append(self.rec)
        return self.trace_id


def _append(rec):
    try:
        os.makedirs(os.path.dirname(TRACE_FILE), exist_ok=True)
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass      # 观测绝不能因为写盘失败把业务打挂


# ---------------- 全局当前 trace ----------------

def start_trace(question, **meta):
    tr = Trace(question, **meta)
    _current.set(tr)
    return tr


def current():
    return _current.get()


def clear():
    _current.set(None)


def record_llm(elapsed_ms=0, prompt_tokens=None, output_tokens=None, ok=True):
    """给 call_llm 用的钩子：没有当前 trace 时静默跳过。"""
    tr = _current.get()
    if tr is not None:
        tr.llm_call(elapsed_ms, prompt_tokens, output_tokens, ok)


# ---------------- 命中块摘要（只留引用与预览，不留全文）----------------

def hit_summary(hits, vec_ranks=None, limit=8, preview_len=100):
    """把 [(分数, meta, 文本)] 转成可落盘的摘要列表。

    vec_ranks：重排前的向量排名列表（与 hits 按位置对齐），来自 4_search 的 diag。
    记这个是为了区分「重排新提上来的」和「向量本来就排前面的」——
    前者说明重排起了作用，后者说明重排没改变什么。
    """
    out = []
    for i, item in enumerate(hits[:limit]):
        try:
            score, meta, text = item[0], item[1] or {}, item[2] or ""
        except (TypeError, IndexError):
            continue
        source = meta.get("source") or ""
        section = meta.get("section") or ""
        item_name = meta.get("item") or ""
        vr = None
        if vec_ranks and i < len(vec_ranks):
            vr = vec_ranks[i]
        out.append({
            "rank": i + 1,
            "chunk_id": f"{source}|{section}|{item_name}".strip("|"),
            "source": source,
            "section": section[:40],
            "score": round(float(score), 4),
            "vec_rank": vr,
            "preview": text[:preview_len].replace("\n", " "),
        })
    return out


# ---------------- CLI ----------------

def _read_traces(path=TRACE_FILE):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _print_trace(r, verbose=True):
    print(f"\n{'='*72}")
    print(f"trace_id  {r['trace_id']}")
    print(f"时间      {r.get('ts')}")
    print(f"问题      {r.get('question')}")
    meta = r.get("meta") or {}
    if meta:
        print(f"参数      {meta}")
    print(f"总耗时    {r.get('elapsed_ms')} ms")

    route = r.get("route")
    if route:
        print(f"路由      intent={route.get('intent')}  branch={route.get('branch')}")
    sec = r.get("section")
    if sec:
        print(f"章节查询  {sec.get('operator')} · {sec.get('section')}  ({sec.get('chunks')} 块)")

    ret = r.get("retrieval")
    if ret:
        print(f"检索      召回候选 {ret.get('candidates')}  返回 {ret.get('returned')}  "
              f"top1={ret.get('top1')}  重排={ret.get('rerank_used')}  "
              f"空={ret.get('empty')}  耗时 {ret.get('elapsed_ms')} ms")
        if verbose:
            for h in (ret.get("hits") or [])[:6]:
                vr = h.get("vec_rank")
                vr_s = f"向量第{vr}名" if vr else "重排新进"
                print(f"            #{h['rank']} {h['score']:.3f} {h['chunk_id'][:52]:<52} {vr_s}")

    flt = r.get("filter")
    if flt:
        print(f"干员过滤  {flt.get('operators')}  丢弃 {flt.get('dropped')}")
    cut = r.get("cut")
    if cut:
        print(f"分数裁剪  保留 {cut.get('kept')}  丢弃 {cut.get('dropped')}")
    ref = r.get("refusal")
    if ref:
        print(f"拒答      原因={ref.get('reason')}  best={ref.get('best')}")
    llm = r.get("llm")
    if llm:
        print(f"模型调用  {llm.get('calls')} 次  prompt {llm.get('prompt_tokens')} tok  "
              f"输出 {llm.get('output_tokens')} tok  耗时 {llm.get('elapsed_ms')} ms"
              + (f"  失败 {llm.get('errors')} 次" if llm.get("errors") else ""))
    rescue = r.get("rescue")
    if rescue:
        print(f"补救      {rescue.get('how')}")
    sc = r.get("selfcheck")
    if sc:
        print(f"自检      {sc.get('verdict')}")

    print(f"回答      {(r.get('answer_preview') or '')[:110]}")
    if r.get("error"):
        print(f"异常      {r['error']}")
    flags = r.get("flags") or []
    print(f"可疑标记  {('、'.join(flags)) if flags else '无'}")


def cmd_last(n):
    rows = _read_traces()
    if not rows:
        print(f"还没有 trace 记录：{TRACE_FILE}")
        return 1
    print(f"共 {len(rows)} 条 trace，显示最近 {min(n, len(rows))} 条")
    for r in rows[-n:]:
        _print_trace(r, verbose=False)
    return 0


def cmd_trace(tid):
    rows = _read_traces()
    hit = [r for r in rows if r.get("trace_id") == tid]
    if not hit:
        hit = [r for r in rows if tid in (r.get("trace_id") or "")]
    if not hit:
        print(f"没找到 trace_id 含 {tid} 的记录")
        return 1
    _print_trace(hit[-1])
    return 0


def cmd_suspicious(limit=20):
    rows = _read_traces()
    bad = [r for r in rows if r.get("flags")]
    print(f"共 {len(rows)} 条 trace，其中 {len(bad)} 条带可疑标记\n")
    for r in bad[-limit:]:
        ts = (r.get("ts") or "")[11:19]
        print(f"{r['trace_id']}  {ts}  {'、'.join(r['flags'])}")
        print(f"   问题: {(r.get('question') or '')[:60]}")
    if bad:
        print(f"\n看某一条的完整链路： python 9_observe.py --trace {bad[-1]['trace_id']}")
    return 0


def _pct(vals, p):
    if not vals:
        return 0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def cmd_stats():
    rows = _read_traces()
    if not rows:
        print(f"还没有 trace 记录：{TRACE_FILE}")
        return 1

    n = len(rows)
    lat = [r["elapsed_ms"] for r in rows if r.get("elapsed_ms") is not None]
    routes = Counter((r.get("route") or {}).get("branch") or "unknown" for r in rows)
    empty = sum(1 for r in rows if (r.get("retrieval") or {}).get("empty"))
    refused = sum(1 for r in rows if r.get("refusal"))
    errors = sum(1 for r in rows if r.get("error"))
    calls = [(r.get("llm") or {}).get("calls", 0) for r in rows]
    toks = [(r.get("llm") or {}).get("prompt_tokens", 0) for r in rows if r.get("llm")]
    top1s = [(r.get("retrieval") or {}).get("top1") for r in rows]
    top1s = [t for t in top1s if isinstance(t, (int, float))]
    flag_cnt = Counter()
    for r in rows:
        for f in (r.get("flags") or []):
            flag_cnt[f.split("(")[0]] += 1

    print(f"=== 汇总（{n} 次请求）===")
    print(f"时间范围      {rows[0].get('ts')}  →  {rows[-1].get('ts')}")
    print(f"端到端延迟    P50 {_pct(lat,50)} ms   P95 {_pct(lat,95)} ms   最大 {max(lat) if lat else 0} ms")
    print(f"空检索比例    {empty}/{n}  ({empty/n*100:.1f}%)")
    print(f"拒答比例      {refused}/{n}  ({refused/n*100:.1f}%)")
    print(f"异常比例      {errors}/{n}")
    print(f"模型调用次数  平均 {sum(calls)/n:.2f} 次/请求   最多 {max(calls) if calls else 0} 次")
    if toks:
        print(f"prompt token  平均 {sum(toks)/len(toks):.0f}   最大 {max(toks)}")
    if top1s:
        print(f"top1 相似度   平均 {sum(top1s)/len(top1s):.3f}   "
              f"低于 {LOW_TOP1} 的有 {sum(1 for t in top1s if t < LOW_TOP1)} 次")
    print(f"\n路由分布")
    for k, v in routes.most_common():
        print(f"   {k:<12} {v}")
    if flag_cnt:
        print(f"\n可疑信号")
        for k, v in flag_cnt.most_common():
            print(f"   {k:<16} {v} 次")
    return 0


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = args[0]
    if cmd == "--last":
        return cmd_last(int(args[1]) if len(args) > 1 else 5)
    if cmd == "--trace":
        return cmd_trace(args[1]) if len(args) > 1 else 1
    if cmd == "--suspicious":
        return cmd_suspicious(int(args[1]) if len(args) > 1 else 20)
    if cmd == "--stats":
        return cmd_stats()
    print(f"未知参数: {cmd}")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
