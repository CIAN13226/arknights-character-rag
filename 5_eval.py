# -*- coding: utf-8 -*-
"""
第 5 步：跑评测集，算 Recall@K 和 MRR

【这个脚本在干什么】
    把评测集里的问题逐条丢给检索系统，看「标注的正确来源」有没有出现在 Top-K 里。
    命中了算 1 分，没命中算 0 分，加起来除以题目数，就是 Recall@K。

【怎么判「命中」】
    评测集里标的是「干员名 · 章节」，比如 `晓歌 · 技能`。
    检索回来的每个块都带 source（干员名）和 section（章节）元数据。
    只要有一个块的 source 和 section 都对上，就算命中。

    有的题标了多个来源（如 `提丰 · 技能，莱伊 · 技能`），命中任意一个都算对。

【为什么还要算 MRR】
    Recall 只看「有没有」，不看「排第几」。
    正确答案排第 1 和排第 5，Recall 都是 1 分，但实际体验差很多。
    MRR（平均倒数排名）= 对每题取 1/首次命中的排名，再平均。
    排第 1 得 1.0，排第 2 得 0.5，排第 5 得 0.2，没命中得 0。

用法：
    python 5_eval.py              # 跑评测，K=5
    python 5_eval.py -k 10        # 换个 K 看变化
    python 5_eval.py --detail     # 打印每道题的详情
"""
import os
import re
import sys
import argparse

import chromadb
import importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))
EVAL_MD = os.path.join(BASE, "评测集_待填.md")
DB_DIR = os.path.join(BASE, "chroma_db")

# 复用 4_search.py 里的模型加载和检索逻辑
spec = importlib.util.spec_from_file_location("s4", os.path.join(BASE, "4_search.py"))
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)


# ---------------- 解析评测集 ----------------

def parse_eval_set(path):
    """从 markdown 表格里读出题目。

    注意：文件里有两张表 —— 上面是「填表示例」，下面才是真正的评测集。
    只解析「## 开始填」之后的表，否则示例行会被算进题目里（曾经把 20 题读成 23 题）。
    """
    text = open(path, encoding="utf-8").read()
    if "## 开始填" in text:
        text = text.split("## 开始填", 1)[1]

    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) < 5:
            continue
        if not re.match(r"^\d+$", cols[0]):        # 跳过表头和分隔行
            continue
        num = int(cols[0])
        q, typ, ans, src = cols[1], cols[2], cols[3], cols[4]
        note = cols[5] if len(cols) > 5 else ""
        if not q:
            continue
        rows.append({"num": num, "q": q, "type": typ,
                     "answer": ans, "src": src, "note": note})
    return rows


def is_judgeable(row):
    """这道题能不能用「检索有没有命中」来判分。

    不能判的情况：
      1. 来源栏是空的 / 写了「无」—— 比如应拒答题，本来就不该检索到东西
      2. 备注里写明预期失败
      3. 来源栏写的是「XX角色的技能档案」这类泛指，不是具体干员名
    """
    src = (row["src"] or "").strip()
    if not src or src in ("无", "-", "—", "/"):
        return False
    if "预期失败" in row["note"]:
        return False
    # 泛指写法：只有「XX角色」没有具体分隔符的，匹配不到具体块
    if "角色" in src and "·" not in src:
        return False
    return True


def parse_expected_sources(src_field):
    """把『晓歌 · 技能，提丰 · 技能』解析成 [(source, section), ...]"""
    out = []
    for part in re.split(r"[，,、;；]", src_field or ""):
        part = part.strip()
        if not part or part in ("无", "-", "—"):
            continue
        # 收集所有可能的分隔写法
        for sep in ["·", "・", "•"]:
            if sep in part:
                a, _, b = part.partition(sep)
                out.append((a.strip(), b.strip()))
                break
        else:
            # 没写章节，只匹配干员名
            out.append((part.strip(), ""))
    return out


def match(hit_meta, expected):
    """判断一个检索结果是否命中某个期望来源"""
    src = (hit_meta.get("source") or "").strip()
    sec = (hit_meta.get("section") or "").strip()
    for e_src, e_sec in expected:
        if not e_src:
            continue
        if src != e_src:
            continue
        if not e_sec:
            return True
        # 章节允许包含关系，比如期望「技能」，实际是「技能」
        if e_sec in sec or sec in e_sec:
            return True
    return False


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--detail", action="store_true", help="打印每道题的详情")
    ap.add_argument("--lib", choices=["db", "lore", "both"], default="both")
    args = ap.parse_args()

    evals = parse_eval_set(EVAL_MD)
    print(f"评测集：{len(evals)} 道题，K = {args.k}，检索范围 = {args.lib}")
    print("=" * 74)

    model = s4.load_model()
    client = chromadb.PersistentClient(path=DB_DIR)

    results = []
    for e in evals:
        expected = parse_expected_sources(e["src"])
        hits = s4.search(model, client, e["q"], args.lib, args.k)

        # 找首次命中的排名（从 1 开始）
        rank = None
        for i, (score, meta, doc) in enumerate(hits, 1):
            if expected and match(meta, expected):
                rank = i
                break

        judgeable = is_judgeable(e)
        hit = (rank is not None) if judgeable else False
        results.append({**e, "rank": rank, "hit": hit, "judgeable": judgeable,
                        "hits": hits, "expected": expected})

    # ---------- 汇总 ----------
    judge = [r for r in results if r["judgeable"]]
    n = len(judge)
    hit_n = sum(1 for r in judge if r["hit"])
    recall = hit_n / n if n else 0
    mrr = sum(1.0 / r["rank"] for r in judge if r["rank"]) / n if n else 0

    print(f"\n可评判的题（有标注来源）：{n} 道")
    print(f"  Recall@{args.k} = {hit_n}/{n} = {recall*100:.1f}%")
    print(f"  MRR          = {mrr:.3f}")

    # ---------- 按类型拆 ----------
    print(f"\n{'类型':<12}{'命中/总数':<12}{'Recall':<10}")
    print("-" * 36)
    types = {}
    for r in judge:
        t = r["type"] or "未标类型"
        types.setdefault(t, []).append(r)
    for t, rs in sorted(types.items(), key=lambda x: -len(x[1])):
        h = sum(1 for r in rs if r["hit"])
        print(f"{t:<12}{f'{h}/{len(rs)}':<12}{h*100//len(rs):>3}%")

    # ---------- 失败案例 ----------
    fails = [r for r in judge if not r["hit"]]
    if fails:
        print(f"\n失败案例（{len(fails)} 道）：")
        for r in fails:
            print(f"  #{r['num']} [{r['type']}] {r['q'][:38]}")
            print(f"       期望来源：{r['src']}")
            top = r["hits"][0] if r["hits"] else None
            if top:
                print(f"       实际 Top1：{top[1].get('source')} · {top[1].get('section')}"
                      f"  (相似度 {top[0]:.3f})")

    # ---------- 不可评判的题 ----------
    unjudge = [r for r in results if not r["judgeable"]]
    if unjudge:
        print(f"\n不可评判的题（{len(unjudge)} 道，没标来源，脚本判不了对错）：")
        for r in unjudge:
            print(f"  #{r['num']} [{r['type']}] {r['q'][:40]}")

    # ---------- 明细 ----------
    if args.detail:
        print("\n" + "=" * 74)
        print("逐题明细：")
        for r in results:
            mark = "命中" if r["hit"] else ("不可判" if not r["judgeable"] else "未命中")
            rank = f"第{r['rank']}位" if r["rank"] else "-"
            print(f"  #{r['num']:<3} [{mark}] {rank:<8} {r['q'][:44]}")

    # ---------- 存报告 ----------
    out = os.path.join(BASE, "评测报告.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# 检索评测报告\n\n")
        f.write(f"- 评测集规模：{n} 道可评判题\n")
        f.write(f"- K = {args.k}\n")
        f.write(f"- **Recall@{args.k} = {recall*100:.1f}%**（{hit_n}/{n}）\n")
        f.write(f"- **MRR = {mrr:.3f}**\n\n")
        f.write("## 按类型\n\n| 类型 | 命中/总数 | Recall |\n|---|---|---|\n")
        for t, rs in sorted(types.items(), key=lambda x: -len(x[1])):
            h = sum(1 for r in rs if r["hit"])
            f.write(f"| {t} | {h}/{len(rs)} | {h*100//len(rs)}% |\n")
        f.write("\n## 逐题结果\n\n| # | 类型 | 问题 | 期望来源 | 结果 | 排名 |\n|---|---|---|---|---|---|\n")
        for r in results:
            mark = "命中" if r["hit"] else ("不可判" if not r["judgeable"] else "未命中")
            f.write(f"| {r['num']} | {r['type']} | {r['q']} | {r['src']} | {mark} | "
                    f"{r['rank'] or '-'} |\n")
    print(f"\n报告已写出：{out}")


if __name__ == "__main__":
    main()
