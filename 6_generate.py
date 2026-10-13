# -*- coding: utf-8 -*-
"""
第 6 步：生成回答（检索 + 大模型）

这就是完整的 RAG 链路了：
    用户提问 → 向量检索（4_search）→ 拼 prompt → 大模型生成 → 带来源的回答

【这一步真正的难点不是"接上模型"，而是"让它不乱说"】
    大模型的默认行为是：不知道也要编一个听起来像样的答案。
    在问答场景里这是致命的——用户没法分辨哪句是真的。

    所以 prompt 里必须有三条硬约束：
      1. 只能根据【参考资料】回答，不许用自己的知识
      2. 资料里没有的，直接说没有，不要推测
      3. 回答要标出信息来自哪个干员的哪一部分

【为什么还要做"拒答判断"】
    如果检索回来的块相似度都很低，说明语料里根本没有相关内容。
    这时候与其让模型硬答，不如直接告诉用户"资料里没有"。
    这一步是纯工程手段，成本低、收益大。

用法：
    python 6_generate.py                      # 跑内置的测试问题
    python 6_generate.py "凯尔希有什么技能"      # 问一个问题
    python 6_generate.py -i                   # 交互模式，可以连续提问
"""
import os
import sys
import json
import time
import argparse
import importlib.util
import urllib.request

import chromadb

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "chroma_db")
OLLAMA = "http://127.0.0.1:11434"

# 检索模块在导入时就加载好。
# 之前它只在 main() 里被赋值，导致别的代码路径调用时 s4 不存在，
# 抛出 NameError 又被宽泛的 except 吞掉，表现为「功能静默失效」。
_spec4 = importlib.util.spec_from_file_location("s4", os.path.join(BASE, "4_search.py"))
s4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(s4)
MODEL = os.environ.get("RAG_LLM", "qwen2.5:7b")
# 取多少条资料喂给模型。
# 为什么不是 5：一个干员的「基本信息」「招募」「特性」这类短块内容少、
# 关键词密度高，相似度往往偏高，会把真正有用的块挤下去。
# 实测问「凯尔希有什么技能」，技能块排在第 3、7、9 位，K=5 时漏掉大半。
TOP_K = 10

# 相似度低于这个值，就认为语料里没有相关内容，直接拒答。
# 注意：这个阈值只在【没开重排】时可靠。开了重排后，分数是重排模型的输出
# （不相关的内容也能给 0.1~0.5），量纲完全不同，所以：
#   - 开重排时：阈值调低，只拦掉「全部候选都极低」的情况；
#     真正的拒答交给 prompt 里那条「找不到就说没有」的规则。
#   - 关重排时：用原来的向量相似度阈值。
MIN_SCORE = 0.35          # 关重排时用
MIN_SCORE_RERANK = 0.02   # 开重排时用（重排分数普遍偏高）

# 自适应截断：只保留分数不低于「最高分 × 这个比例」的资料。
# 为什么要截断：实测发现把 8 条资料一起给模型，它会拒答；
# 只给 1-5 条却能正确归纳。长上下文里做跨文本归纳会退化——
# 噪声越多，模型越倾向于保守地说「资料里没有」。
# 实测分数分布有明显台阶（相关的在 0.93 以上，无关的掉到 0.6 以下），
# 用 0.8 这个比例正好能切在台阶上。
# 例外：如果分数是连续下降没有台阶（语义模糊类问题就是这样），
# 截断不会砍掉太多，也不会误伤。
SCORE_RATIO = 0.8
MAX_CONTEXT_CHUNKS = 8    # 硬上限，防止极端情况塞太多

# 是否开启「答案自检」：回答完让模型拿资料核对一遍，发现没依据就修正。
# 代价是每次提问多一次模型调用（约 3-5 秒），换来更低的出错率。
# 用环境变量 RAG_NO_CHECK=1 可以关掉。
SELF_CHECK_ON = os.environ.get("RAG_NO_CHECK", "") != "1"


def adaptive_cut(hits, ratio=SCORE_RATIO, hard_cap=MAX_CONTEXT_CHUNKS):
    """按分数台阶自适应保留资料，避免噪声把模型逼到拒答。

    返回 (保留后的列表, 被砍掉的数量)
    """
    if not hits:
        return hits, 0
    top = hits[0][0]
    if top <= 0:
        return hits[:hard_cap], max(0, len(hits) - hard_cap)
    kept = [h for h in hits if h[0] >= top * ratio]
    # 至少留 2 条，避免只剩一条时信息不足
    if len(kept) < 2:
        kept = hits[:2]
    if len(kept) > hard_cap:
        kept = kept[:hard_cap]
    return kept, len(hits) - len(kept)

SYSTEM_PROMPT = """你是一个《明日方舟》游戏知识助手。你的回答必须严格遵守以下规则：

【规则一】只能根据下面提供的【参考资料】回答，绝对不能使用你自己的知识补充内容。
【规则二】如果【参考资料】里找不到答案，就明确回答"参考资料里没有相关信息"，不要猜测、不要编造。
【规则三】回答中提到的每一项信息，都要标注它来自哪个干员的哪一部分，格式如（来源：凯尔希 · 技能）。
【规则四】如果资料里有数字，原样引用，不要换算或近似。
【规则五】注意区分类似名字的干员。如果问题问的是「凯尔希」，而资料里有「凯尔希·思衡托」，
        那是不同的干员，不要把它写进答案。只使用干员名与问题完全一致的那部分资料。
【规则六】如果资料里没有直接给出答案，但提供了能推出答案的线索（比如问两人关系，
        资料里写了「前队友」「并肩作战」），就根据线索归纳出结论，并说明你依据的是哪句话。
        归纳要以资料原文为依据，不能加入资料之外的推测。
【规则七】回答简洁，直接给结论，不要复述问题。"""

USER_TEMPLATE = """【参考资料】
{context}

【问题】
{question}

请根据上面的参考资料回答。记住：资料里没有的就直说没有。"""


def load_search_module():
    spec = importlib.util.spec_from_file_location("s4", os.path.join(BASE, "4_search.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def call_llm(system, user, model=MODEL, timeout=300):
    """调用 Ollama 的 /api/chat 接口"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.1},   # 低温度，减少发挥
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # 把每次调用写进日志，方便出问题时回看现场。
    # 之前遇到过「资料里明明有答案，模型却说没有」，但复现不出来——
    # 没有日志就只能靠猜。日志里带上了模型名、prompt 长度、耗时、原始返回，
    # 这些信息足以判断是「模型没看到」「模型看到了但判断错」还是「调用本身出了问题」。
    t0 = time.time()
    log = {"ts": time.strftime("%H:%M:%S"), "model": model,
           "system_len": len(system), "user_len": len(user)}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        out = data.get("message", {}).get("content", "")
        log.update({"elapsed": round(time.time() - t0, 2), "ok": True,
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "output_tokens": data.get("eval_count"),
                    "done_reason": data.get("done_reason"),
                    "reply": out[:200]})
        return out
    except Exception as e:
        log.update({"elapsed": round(time.time() - t0, 2), "ok": False,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"})
        raise
    finally:
        try:
            with open(os.path.join(BASE, "llm_log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
        except Exception:
            pass


def detect_operators(question, hits):
    """从问题里找出被提到的所有干员。

    为什么返回列表而不是单个名字：
      问题可能涉及多个干员（「伊内丝和W的关系如何」）。
      早期版本只取最长的那个匹配，结果只保留伊内丝的资料、把 W 的全过滤掉，
      模型手上没有 W 的信息，只能回答「没有相关信息」——而 W 的资料明明检索到了，
      而且重排分数高达 0.997。

    异格处理：如果同时匹配到「凯尔希」和「凯尔希·思衡托」，只保留最长的那个，
    因为它们指的是不同干员，同时保留会让回答混淆。
    """
    names = {(meta.get("source") or "").strip() for _, meta, _ in hits}
    # 长度限制放宽到 1：语料里有单字符干员名（「W」），
    # 之前限制 >=2 会让「伊内丝和W的关系」识别不出 W，把 W 的资料全过滤掉。
    found = [n for n in names if n and n in question]
    if not found:
        return []
    # 去掉被更长名字包含的短名字（问「凯尔希」时不要把「凯尔希·思衡托」也算进来）
    result = [n for n in found if not any(n != o and n in o for o in found)]
    return result


def detect_operator(question, hits):
    """兼容旧调用：只返回一个名字（取最长匹配）"""
    ops = detect_operators(question, hits)
    return max(ops, key=len) if ops else None


def filter_hits(hits, operator):
    """把不属于这些干员的块剔掉。

    operator 可以是单个名字或名字列表。问题是多个干员时，相关干员的资料都要保留。
    保留规则：块里的 source 要正好等于其中某个干员名；
    对于「凯尔希·思衡托」这种以「凯尔希」开头的异格，只在没有任何精确匹配时才放行。
    """
    if not operator:
        return hits
    ops = [operator] if isinstance(operator, str) else list(operator)
    exact, extended = [], []
    for h in hits:
        src = (h[1].get("source") or "").strip()
        if src in ops:
            exact.append(h)
        elif any(src.startswith(o) for o in ops):
            extended.append(h)
    # 有精确匹配就只用精确匹配，避免异格污染
    return exact if exact else extended


def looks_like_refusal(text):
    """判断模型的回答是不是「资料里没有」这类拒答"""
    head = (text or "")[:40]
    keys = ["没有相关", "没有提到", "未提及", "没有明确", "无法回答", "资料中没有",
            "没有找到", "未找到", "没有提供", "无从得知", "资料里没有", "无法确定",
            "没有直接"]
    return any(k in head for k in keys)


def retry_with_escalation(reply, question, model, used_hits, all_hits):
    """模型没答好时的逐级补救。

    【为什么需要它】
    实测出来的规律（问「巫恋的娃娃叫什么名字」）：
      只给排名第 1 那条（技能名「诅咒娃娃」）  -> 答成「诅咒娃娃」（错的，而且很肯定）
      只给含答案那条（语音台词「干员报到」）    -> 答对「小莫提」
      第 1 条 + 含答案那条                    -> 拒答
      全部 10 条                             -> 拒答

    **噪声不只是稀释信息，它会破坏模型对正确信息的采信。**
    语义邻近但不对的块排在前面（技能叫「诅咒娃娃」，而问的是娃娃的名字），
    模型就会被带偏，或者干脆不敢下结论。

    【为什么不能只靠「拒答检测」】
    上面第一种情况模型答的是「诅咒娃娃」——语气肯定、格式正确，没有任何拒答特征。
    但它是错的。**对用户来说答错比拒答更糟**，所以补救不能只在拒答时触发。

    【做法】
    换一条资料单独重试，看能不能得到不同的（更可能有依据的）答案。
    从最相关的一条开始，逐条往下试，最多试 MAX_RETRY 条。
    哪条答出了就用哪条；全都答不出就保留原答案。
    """
    if not looks_like_refusal(reply):
        return reply, ""

    pool = all_hits or used_hits
    if not pool:
        return reply, ""

    MAX_RETRY = 5
    for n in range(1, min(MAX_RETRY, len(pool)) + 1):
        sub = pool[:n]
        try:
            out = call_llm(SYSTEM_PROMPT,
                           USER_TEMPLATE.format(context=build_context(sub),
                                                question=question),
                           model=model)
            if not looks_like_refusal(out):
                how = (f"精简到前 {n} 条资料后答出" if n <= 3
                       else f"改用 {n} 条资料后答出")
                return out, how
        except Exception as e:
            print(f"      [警告] 第 {n} 条重试失败：{type(e).__name__}: {e}")

    return reply, ""


def bridge_terms(question):
    """把用户的口语词替换成语料里更可能出现的书面词。

    为什么要这一步：
      用户问「巫恋的娃娃叫什么名字」，而语料里写的是「玩偶」「人偶」。
      用「娃娃」去判断哪条资料含答案会漏判——实测含答案那条
      （「她时常借"小莫提"，也就是她手里的玩偶之口表达自己的想法」）
      在「娃娃」这个维度上和不相关的内容没有区别。

    这就是「查询改写」的一个最小实现：不改检索，只改「挑哪条资料用」这一步。
    完整的查询改写（先让模型把问题重写一遍再去检索）留作后续优化。
    """
    mapping = {
        "娃娃": "玩偶", "毛绒": "玩偶", "布偶": "玩偶",
        "老家": "出身地", "哪里人": "出身地", "故乡": "出身地",
        "多少钱": "部署费用", "费用": "部署费用",
        "血量": "生命上限", "血": "生命上限",
        "几星": "星级", "稀有度": "星级",
    }
    extra = [v for k, v in mapping.items() if k in question]
    return " ".join(extra)


def pick_answer_chunk(question, pool, max_check=6):
    """从候选里挑出「最可能含答案」的一条。

    做法：把问题里的实体（专有名词、引号内容）和问题本身，加上改写过的同义词，
    去候选文本里匹配，按命中数排序。

    这是为了让「逐条重试」有个合理的顺序——
    之前是按相关度顺序试，但相关度第 1 的可能是误导项
    （问娃娃的名字，排第 1 的却是技能名「诅咒娃娃」）。
    """
    # 从问题里抽出可能是实体的词：中文专有名词按 2-4 字滑窗不现实，
    # 改成用问题里的「名词性片段」——去掉疑问词和停用词
    stop = set("的了吗呢是什么叫名字请问有没有哪个谁多少怎么如何和与跟还有")
    core = "".join(ch for ch in question if ch not in stop)
    extra = bridge_terms(question)
    keys = [k for k in (core, extra) if k]
    # 再补上问题里的每个 2 字以上的连续中文片段
    import re as _re
    for seg in _re.findall(r"[\u4e00-\u9fff]{2,}", question):
        if seg not in keys:
            keys.append(seg)

    scored = []
    for i, hit in enumerate(pool[:max_check]):
        doc = hit[2]
        score = 0
        for k in keys:
            if k and k in doc:
                score += len(k)
        # 候选里出现了换写后的词，说明更可能是在讲同一件事
        if extra and extra in doc:
            score += 5
        scored.append((score, i, hit))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [h for _, _, h in scored]


def retry_per_chunk(question, model, pool, max_try=4):
    """逐条单独重试：把每条资料单独喂给模型，看哪条能答出。

    什么时候用：主路径和 retry_with_escalation 都没给出好答案时。
    典型场景是「巫恋的娃娃叫什么名字」——
    排名第 1 的块（技能名「诅咒娃娃」）会误导模型给出一个肯定但错误的答案，
    真正含答案的块排在后面。合并上下文会让两者互相干扰，
    单独喂才有可能让正确的那条发挥作用。

    重试顺序不是按相关度，而是按 pick_answer_chunk 算出的「最可能含答案」，
    因为相关度第 1 的往往就是那个误导项。
    """
    ordered = pick_answer_chunk(question, pool)
    for hit in ordered[:max_try]:
        try:
            out = call_llm(SYSTEM_PROMPT,
                           USER_TEMPLATE.format(context=build_context([hit]),
                                                question=question),
                           model=model)
            if not looks_like_refusal(out):
                src = hit[1].get("source", "?")
                sec = hit[1].get("section", "")
                return out, f"改用「{src} · {sec}」单独回答"
        except Exception as e:
            print(f"      [警告] 单条重试失败：{type(e).__name__}: {e}")
    return None, ""
    """逐条单独重试：把每条资料单独喂给模型，看哪条能答出。

    什么时候用：主路径和 retry_with_escalation 都失败了。
    典型场景是「巫恋的娃娃叫什么名字」——
    排名第 1 的块（技能「诅咒娃娃」）会误导模型给出一个肯定但错误的答案，
    真正含答案的块排在后面。合并上下文会让两者互相干扰，
    单独喂才有可能让正确的那条发挥作用。

    代价是要多跑几次模型，所以只在前面都失败时才走这条路。

    重试顺序不是按相关度，而是按 pick_answer_chunk 算出的「最可能含答案」，
    因为相关度第 1 的往往是误导项。
    """
    ordered = pick_answer_chunk(question, pool)
    for i, hit in enumerate(ordered[:max_try], 1):
        try:
            out = call_llm(SYSTEM_PROMPT,
                           USER_TEMPLATE.format(context=build_context([hit]),
                                                question=question),
                           model=model)
            if not looks_like_refusal(out):
                src = hit[1].get("source", "?")
                sec = hit[1].get("section", "")
                return out, f"改用「{src} · {sec}」单独回答成功"
        except Exception as e:
            print(f"      [警告] 单条重试失败：{type(e).__name__}: {e}")
    return None, ""


CHECK_PROMPT = """你是审核员。下面有一份【参考资料】、【用户问题】和【待审答案】。

请逐句核对【待审答案】里的每一项陈述，看它能不能在【参考资料】里找到依据。

输出格式（严格照这个格式，不要加别的内容）：
判定：通过 / 不通过
问题：如果有陈述在资料里找不到依据，写出来；找不到就写「无」
修正答案：如果不通过，给出一个只依据资料的修正答案；如果通过，重复原答案"""


def self_check(reply, question, model, hits, verbose=True):
    """答案自检：让模型核对自己的回答有没有资料依据。

    【为什么需要这一步】
    实测发现模型会给出**语气肯定但错误**的答案，而且没有任何拒答特征：
      问「巫恋的娃娃叫什么名字」
      排名第 1 的资料是技能名「诅咒娃娃」
      -> 模型答「巫恋的娃娃叫诅咒娃娃」
    真正含答案的那条（档案里写「她时常借"小莫提"，也就是她手里的玩偶」）排在后面，
    被前面那条误导项盖住了。

    我的补救机制原来只在「拒答」时触发，对这种「自信地答错」完全无效。
    而**程序自己没法判断答得对不对**——这需要语义理解，写规则解决不了。
    既然模型有能力判断「这句话在资料里有没有依据」，那就让它自己查一遍。

    【代价】
    每次提问多一次模型调用（约 3-5 秒）。这是用时间换可靠性的取舍，
    对于知识问答场景是值得的：错误答案比慢更伤用户。

    返回 (最终答案, 是否做了修正)
    """
    if not hits:
        return reply, False
    ctx = build_context(hits)
    user = (f"【参考资料】\n{ctx}\n\n【用户问题】\n{question}\n\n"
            f"【待审答案】\n{reply}\n\n请核对。")
    try:
        verdict = call_llm(CHECK_PROMPT, user, model=model)
    except Exception as e:
        if verbose:
            print(f"      [警告] 自检失败，跳过：{type(e).__name__}: {e}")
        return reply, False

    passed = "判定：通过" in verdict or "判定:通过" in verdict
    fixed = ""
    for line in verdict.split("\n"):
        line = line.strip()
        if line.startswith("修正答案"):
            fixed = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            break

    if passed or not fixed or looks_like_refusal(fixed) and not looks_like_refusal(reply):
        if verbose and not passed:
            print(f"      （自检发现问题：{verdict[:80]}）")
        return reply, False

    if verbose:
        print("      （自检发现原答案缺依据，已改用自检修正后的答案）")
    return fixed, True


def boost_answer_chunk(question, hits, pool=None):
    """把「最可能含答案」的那条提到最前面。

    【为什么需要】
    实测（问「巫恋的娃娃叫什么名字」）：
      含答案的那条得分 0.822，而最高分是 0.997，
      按「保留 ≥ 最高分 × 0.8」的规则，阈值是 0.798——
      0.822 只比阈值高一点，但排序上它排第 5，
      而前 4 条（技能名「诅咒娃娃」等）把它盖住了，最后被截断砍掉。

    但把这一条单独给模型，它能准确回答「小莫提」。
    所以问题不在模型，在于这条资料没能进入最终上下文。

    【做法】
    用词面重合度找「最可能含答案」的那条（含改写同义词，比如「娃娃」-> 「玩偶」），
    把它插到最前面。这样它一定在截断保留范围内。

    pool：如果有更宽的候选池（未经截断的全部候选），优先在里面找。
    """
    src = pool if pool else hits
    if not src or not hits:
        return hits
    ordered = pick_answer_chunk(question, src, max_check=len(src))
    if not ordered:
        return hits
    best = ordered[0]
    # 已经在第一位就不用动
    if best[2][:80] == hits[0][2][:80]:
        return hits
    rest = [h for h in hits if h[2][:80] != best[2][:80]]
    return [best] + rest


def detect_operators_by_scan(question, model, client, lib="db", probe_k=40):
    """用一次便宜的向量检索（不重排），扫出问题里提到了哪些干员。

    为什么要单独扫一遍：
      重排模型的分数会波动，同一个问题两次跑，某个干员的块可能一次进前 10、一次不进。
      如果只靠主检索的结果来判断「问题涉及谁」，就会出现
      「问了伊内丝和W，但这次没召回 W 的块，于是 W 的资料被当成无关内容过滤掉」。
      先用向量检索扫 40 条（不重排，几十毫秒），把候选干员名收集全，再决定保留谁。
    """
    try:
        probe = s4.search(model, client, question, lib, probe_k, rerank=False)
    except Exception as e:
        # 这里不能静默失败：早期版本写的是 `return []`，
        # 结果把 NameError 这种编程错误也吞了，功能失效但毫无提示，排查了很久。
        print(f"      [警告] 干员预扫描失败：{type(e).__name__}: {e}")
        return []
    names = {(meta.get("source") or "").strip() for _, meta, _ in probe}
    # 长度限制放宽到 1：语料里有单字符干员名（「W」），
    # 之前限制 >=2 会让「伊内丝和W的关系」识别不出 W，把 W 的资料全过滤掉。
    found = [n for n in names if n and n in question]
    if not found:
        return []
    # 去掉被更长名字包含的短名字（异格处理）
    return [n for n in found if not any(n != o and n in o for o in found)]
    """用一次便宜的向量检索（不重排），扫出问题里提到了哪些干员。

    为什么要单独扫一遍：
      重排模型的分数会波动，同一个问题两次跑，某个干员的块可能一次进前 10、一次不进。
      如果只靠主检索的结果来判断「问题涉及谁」，就会出现
      「问了伊内丝和W，但这次没召回 W 的块，于是 W 的资料被当成无关内容过滤掉」。
      先用向量检索扫 40 条（不重排，几十毫秒），把候选干员名收集全，再决定保留谁。
    """
    try:
        probe = s4.search(model, client, question, lib, probe_k, rerank=False)
    except Exception as e:
        # 这里不能静默失败：早期版本写的是 `return []`，
        # 结果把 NameError 这种编程错误也吞了，功能失效但毫无提示，排查了很久。
        print(f"      [警告] 干员预扫描失败：{type(e).__name__}: {e}")
        return []
    names = {(meta.get("source") or "").strip() for _, meta, _ in probe}
    found = [n for n in names if n and n in question]
    if not found:
        return []
    # 去掉被更长名字包含的短名字（异格处理）
    return [n for n in found if not any(n != o and n in o for o in found)]


def build_context(hits):
    """把检索到的块拼成给模型的参考资料，每块前面标明出处"""
    parts = []
    for i, (score, meta, doc) in enumerate(hits, 1):
        src = f"{meta.get('source')} · {meta.get('section')}"
        if meta.get("item"):
            src += f" · {meta['item']}"
        parts.append(f"[资料{i}] 出处：{src}（相似度 {score:.3f}）\n{doc}")
    return "\n\n".join(parts)


def generate(question, model, emb, client, lib="both", k=None,
             use_rerank=None, verbose=False):
    """完整的问答链路。网页和命令行都必须调用这一个函数。

    【为什么要合成一个函数】
      之前 6_generate.py 和 7_web.py 各写了一遍流程，
      结果每次修 bug 只修了其中一条路径：
      「巫恋的娃娃」在命令行修好了，网页上还是答错——因为网页那条没调用新加的
      boost_answer_chunk（把最可能含答案的那条提到最前面）。
      两条路径必须共用同一份逻辑，否则这种「修了一半」的问题会反复出现。

    返回 dict：reply / hits / route / notes / structured_text
    """
    import importlib.util as _u
    k = k or TOP_K
    notes = []

    # ---------- 第一层：意图路由 ----------
    # 筛选/排序类问题（「散射手都有谁」「谁攻击力最高」）向量检索做不了，
    # 它只能找「最像的 K 个块」，没有「全部」的概念。这类走结构化查询。
    spec = _u.spec_from_file_location("r8", os.path.join(BASE, "8_router.py"))
    r8 = _u.module_from_spec(spec)
    spec.loader.exec_module(r8)
    if r8.detect_intent(question) in ("list", "rank"):
        text, desc = r8.run_structured(question)
        if text:
            notes.append(f"走结构化查询（{desc}）")
            # 表格类结果不能让模型「概括」——实测它会把 5 人压成 1 人。
            reply = call_llm(
                "你是《明日方舟》游戏知识助手。下面是从数据库里查出来的完整结果，"
                "请把它整理成便于阅读的中文回答。\n"
                "【重要】结果里有几条就要列出几条，一条都不能省略或合并。"
                "「简洁」指的是不要加额外的解释，不是让你压缩条目。\n"
                "不要添加结果之外的任何信息。",
                f"【查询结果】\n{text}\n\n【问题】\n{question}\n\n"
                f"请逐条列出查询结果里的每一行，不要遗漏。",
                model=model)
            return {"reply": reply, "hits": [], "route": "structured",
                    "notes": notes, "structured_text": text}

    # ---------- 第一层补充：章节补齐 ----------
    # 「凯尔希有什么技能」这类问题问的是「该干员某章节的全部内容」。
    # 向量检索只返回「最像的 K 个块」——实测它把「招募」「综合检测」排前面，
    # 凯尔希的 3 个技能只进来 1 个，答案就残了。
    # 这和「散射手都有谁」一样，要的是穷举不是相似度，所以直接把该章节的块全捞出来。
    sec_op, sec_name = r8.detect_section_query(question, r8.load_rows()[1])
    if sec_op and sec_name:
        sec_hits = []
        for col_name in [f"arknights_db_{s4.SUFFIX}"]:
            try:
                col = client.get_collection(col_name)
            except Exception:
                continue
            got = col.get(where={"source": sec_op})
            for doc, meta in zip(got["documents"], got["metadatas"]):
                if sec_name in (meta.get("section") or ""):
                    sec_hits.append((1.0, meta, doc))
        if sec_hits:
            # 把章节内容按 item 排一下序，让 1/2/3 技能按顺序出现
            sec_hits.sort(key=lambda h: (h[1].get("section") or "", h[1].get("item") or ""))
            notes.append(f"识别为章节查询：把「{sec_op} · {sec_name}」下的 "
                         f"{len(sec_hits)} 个块全部取出（不靠相似度排序）")
            reply = call_llm(
                SYSTEM_PROMPT,
                USER_TEMPLATE.format(context=build_context(sec_hits), question=question),
                model=model)
            return {"reply": reply, "hits": sec_hits, "route": "section",
                    "notes": notes, "structured_text": None}

    # ---------- 第二层：向量检索 ----------
    # 先扫一遍确定涉及哪些干员（多干员问题要都保留，比如「伊内丝和W的关系」）
    ops = detect_operators_by_scan(question, emb, client, lib)
    hits = s4.search(emb, client, question, lib, k, rerank=use_rerank)
    if not hits:
        return {"reply": "检索没有返回任何结果。", "hits": [], "route": "vector",
                "notes": notes, "structured_text": None}
    if not ops:
        ops = detect_operators(question, hits)

    all_hits = filter_hits(hits, ops)
    if len(hits) - len(all_hits):
        notes.append(f"过滤掉 {len(hits)-len(all_hits)} 条不属于"
                     f"「{'、'.join(ops)}」的资料")

    # ---------- 第三层：挑出最可能含答案的那条，提到最前面 ----------
    # 必须放在截断之前。实测「巫恋的娃娃」含答案的块得分 0.822 排第 5，
    # 会被后面按分数台阶的截断砍掉，导致模型答错或拒答。
    all_hits = boost_answer_chunk(question, all_hits)

    filtered, cut = adaptive_cut(all_hits)
    if cut:
        notes.append(f"按分数台阶砍掉 {cut} 条低相关资料")

    best = filtered[0][0] if filtered else 0.0
    threshold = MIN_SCORE_RERANK if s4.RERANK_ON else MIN_SCORE
    if best < threshold:
        return {"reply": f"参考资料里没有相关信息。\n\n"
                         f"（最相关的资料得分 {best:.3f}，低于阈值 {threshold}）",
                "hits": filtered, "route": "vector", "notes": notes,
                "structured_text": None}

    # ---------- 第四层：生成 ----------
    reply = call_llm(SYSTEM_PROMPT,
                     USER_TEMPLATE.format(context=build_context(filtered),
                                          question=question),
                     model=model)

    # ---------- 第五层：三道补救 ----------
    reply, how = retry_with_escalation(reply, question, model, filtered, all_hits)
    if not how or looks_like_refusal(reply):
        out2, how2 = retry_per_chunk(question, model, all_hits)
        if out2:
            reply, how = out2, how2
    if SELF_CHECK_ON:
        reply, fixed = self_check(reply, question, model, filtered, verbose)
        if fixed:
            how = (how + "；" if how else "") + "经自检修正"
    if how:
        notes.append(f"补救：{how}")

    return {"reply": reply, "hits": filtered, "route": "vector",
            "notes": notes, "structured_text": None}


def answer(model, client, question, lib="both", verbose=True):
    """命令行入口：调用共用的 generate()，保证和网页行为一致。

    早期版本这里和 7_web.py 各写了一遍流程，结果修 bug 只修了一条路径——
    「巫恋的娃娃」在命令行修好了，网页上还是答错。
    现在两边都走 generate()，只有一份逻辑。
    """
    emb = _emb_global()
    res = generate(question, model, emb, client, lib=lib, verbose=verbose)
    if verbose:
        for n in res["notes"]:
            print(f"      （{n}）")
    return res["reply"], res["hits"]


_EMB_CACHE = {}


def _emb_global():
    """缓存一份向量模型，避免每次调用都重新加载"""
    if "m" not in _EMB_CACHE:
        _EMB_CACHE["m"] = s4.load_model()
    return _EMB_CACHE["m"]


def show(question, reply, hits, elapsed):
    print(f"\n{'='*72}")
    print(f"问题：{question}")
    print(f"{'='*72}")
    print(reply)
    print(f"\n--- 检索到的资料（{len(hits)} 条，耗时 {elapsed:.1f}s）---")
    for i, (score, meta, doc) in enumerate(hits, 1):
        head = f"{meta.get('source')} · {meta.get('section')}"
        if meta.get("item"):
            head += f" · {meta['item']}"
        print(f"  [{i}] {score:.3f}  {head}")
    print()


def main():
    global s4
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--lib", choices=["db", "lore", "both"], default="both")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    s4 = load_search_module()
    print(f"[1/3] 加载向量模型 ...")
    emb = s4.load_model()
    client = chromadb.PersistentClient(path=DB_DIR)
    print(f"[2/3] 连接 Ollama（{args.model}）...")
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
            tags = json.loads(r.read().decode("utf-8"))
        names = [m["name"] for m in tags.get("models", [])]
        print(f"      已安装的模型：{', '.join(names) if names else '（无）'}")
        if not any(args.model.split(":")[0] in n for n in names):
            print(f"      [警告] 找不到 {args.model}，可能还没拉取")
    except Exception as e:
        print(f"      [错误] 连不上 Ollama：{e}")
        print("      先启动服务：ollama serve")
        return

    print(f"[3/3] 就绪，开始问答\n")

    if args.interactive:
        print("进入交互模式，输入问题回车即可，输入 q 退出。\n")
        while True:
            try:
                q = input("问：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("q", "quit", "exit"):
                break
            t0 = time.time()
            reply, hits = answer(args.model, client, q, args.lib)
            show(q, reply, hits, time.time() - t0)

    questions = [" ".join(args.question)] if args.question else [
        "凯尔希有什么技能？",
        "星熊二技能的专精三效果是什么？",
        "晓歌的出身是哪里？",
        "找一个性格冷淡、话很少的干员",
        "阿米娅晚上睡觉喜不喜欢踢被子？",
    ]
    for q in questions:
        t0 = time.time()
        reply, hits = answer(args.model, client, q, args.lib)
        show(q, reply, hits, time.time() - t0)

    print("=" * 72)
    print("看结果时重点核对：")
    print("  1. 回答里的信息，在【检索到的资料】里能不能找到？找不到就是幻觉。")
    print("  2. 最后那个『踢被子』的问题，应该回答『没有相关信息』。")
    print("     如果它编了一个答案，说明 prompt 的约束不够，要调。")


if __name__ == "__main__":
    main()
