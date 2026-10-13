# -*- coding: utf-8 -*-
"""
第 8 步：查询路由

【为什么要路由】
    RAG 的向量检索只擅长一件事：给定一句话，找出语料里最像的几段。
    但用户的问题有两类，只有一类适合它：

      A. 语义问答：「凯尔希有什么技能」「找个性格冷淡的干员」
         -> 向量检索，按相关度返回

      B. 筛选列举：「散射手都有谁」「六星里谁防御力最高」「有几个执旗手」
         -> 这是数据库查询。需要遍历全部记录按条件筛选。

    用向量检索做 B 会失败，而且不是「召回率差一点」的失败，是**能力不对**：
    它没有「全部」的概念，只返回最像的 K 个块。
    实测问「散射手都有谁，分别几星」，正确 5 人，向量方案漏了 6 星的维什戴尔。

【怎么判断走哪条路】
    关键词 + 字段识别。听起来简陋，但在垂直领域里非常有效——
    因为问题类型高度集中，而且筛选字段（职业/分支/星级/阵营…）是有限的枚举值。

    这也正是最初设计里的那个想法：
    「让 RAG 在回答之前自己判断这次要用数据库还是资料库」。

用法：
    python 8_router.py "散射手都有谁？分别是几星？"
    python 8_router.py "凯尔希有什么技能？"
    python 8_router.py            # 跑一批内置测试
"""
import os
import re
import sys
import json
import importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))

# 列举意图的信号词
# 注意「都有谁」这类要有，因为用户常问「德拉克种族的干员都有谁」——
# 句末的问法也要覆盖，不能只匹配句首。
LIST_SIGNALS = ["都有谁", "有哪些", "都有哪些", "分别是", "列举", "列出", "全部",
                "所有的", "哪些干员", "几个", "有多少", "一共", "谁是",
                "都有什么", "是哪些", "哪些是",
                # 「谁能吃X加成」是问隐藏势力的典型说法：
                # 隐藏势力决定该干员在战斗的势力判定中额外算哪个势力，
                # 所以「能吃某势力加成」= 隐藏势力包含它，或本身就在该势力。
                "谁能吃", "能吃", "吃得到", "享受", "能吃到"]

# 排序 / 比较意图的信号词
RANK_SIGNALS = ["最高", "最低", "最大", "最小", "最强", "最弱", "排名", "谁最",
                "谁更", "哪个更", "哪个高", "哪个低", "更高", "更低", "比较多"]

# 可以当筛选条件的字段（对应结构化表里的键）
# 「种族」「性别」「出身地」一开始漏了——它们不在 corpus_db 的基本信息段，
# 而在 corpus_lore 的档案里（格式是【种族】德拉克）。
# 「势力」是 PRTS 页面上「所属势力」那一栏，格式「维多利亚 › 塔拉」，
# 匹配时按任一层级命中。
FILTER_FIELDS = ["职业", "分支", "星级", "职位", "标签",
                 "种族", "性别", "出身地",
                 "势力", "隐藏势力"]

# 玩家口语 -> 表里的字段名。
# 为什么必须有这张表：PRTS 页面上显示的字段叫「所属势力」，
# 但玩家会说「阵营」「势力」「国家」「所属」，甚至只说「哪个势力的」。
# 靠关键词精确匹配字段名，换个说法就失灵——实测「有哪些干员的阵营是炎国」
# 就是因为「阵营」不在字段列表里，退化到向量检索，只答出 1 个人（实际 48 人）。
FIELD_ALIASES = {
    "势力": ["阵营", "势力", "所属势力", "所属", "国家", "哪个势力", "什么势力",
             "加成", "势力加成"],
    "隐藏势力": ["隐藏势力", "隐藏阵营", "附加势力"],
    "职业": ["职业", "职业类型"],
    "分支": ["分支", "子职业", "细分职业"],
    "星级": ["星级", "稀有度", "几星", "多少星"],
    "种族": ["种族", "什么族"],
    "性别": ["性别"],
    "出身地": ["出身地", "老家", "故乡", "哪里人", "来自哪里"],
    "标签": ["标签", "特性"],
}

# 字段优先级：同一个词命中多个字段时，只取优先级最高的那个。
# 为什么需要：「杜林」同时是一个种族名和一个出身地名（至简的出身地是杜林），
# 如果两个都当筛选条件取交集，结果会从 5 人缩成 1 人。
FIELD_PRIORITY = ["分支", "职业", "种族", "性别", "星级",
                  "标签", "势力", "隐藏势力", "出身地"]

# 数值字段：问题里出现这些词，且问的是比较/最高，就走数值排序
NUM_FIELDS = {
    "生命上限": "生命上限", "生命": "生命上限",
    "攻击力": "攻击", "攻击": "攻击",
    "防御力": "防御", "防御": "防御",
    "法术抗性": "法抗", "法抗": "法抗",
    "攻击速度": "攻击速度",
    "部署费用": "部署费用", "费用": "部署费用",
    "阻挡数": "阻挡", "阻挡": "阻挡",
}


def load_rows():
    import importlib.util as u
    spec = u.spec_from_file_location("sq", os.path.join(BASE, "structured_query.py"))
    sq = u.module_from_spec(spec)
    spec.loader.exec_module(sq)
    return sq, sq.load_table()


def detect_intent(question):
    """判断问题类型：list（筛选列举）/ rank（排序比较）/ semantic（语义问答）"""
    q = question
    has_list = any(s in q for s in LIST_SIGNALS)
    has_rank = any(s in q for s in RANK_SIGNALS)
    if has_rank:
        return "rank"
    if has_list:
        return "list"
    return "semantic"


def strip_field_phrases(question):
    """删掉问题里的「字段提示短语」和疑问句式，剩下的才可能是干员名。

    【为什么需要】
    「有哪些干员的种族是杜林」里，「杜林」同时是：
      - 一个干员的名字（2 星术师，名字就叫杜林）
      - 一个种族名（杜林族）
      - 一个出身地名（至简的出身地是杜林）
    如果三个都当成筛选条件取交集，结果必然是空。

    注意：光靠「删掉字段名」还不够——删完之后剩「的杜林」，
    「杜林」照样被当成干员名。所以真正管用的是 is_field_value 里的位置判断。
    """
    cleaned = question
    for w in ["有哪些干员", "哪些干员", "都有哪些", "都有谁", "有哪些", "有谁",
              "分别是几星", "分别是", "共有", "一共", "几个", "多少", "是谁",
              "都有什么", "是哪些", "哪些是", "干员们", "干员"]:
        cleaned = cleaned.replace(w, "")
    for f in FILTER_FIELDS:
        cleaned = re.sub(f + r"(是|为|有|的)?", "", cleaned)
        cleaned = re.sub(r"(是|为)" + f, "", cleaned)
    return cleaned


def is_field_value(question, name):
    """判断 name 在问题里是「字段的值」还是「被提到的干员」。

    【为什么不能用字符串包含判断】
    问「种族是杜林的干员」时，句子里有「杜林」，但它是种族值不是干员名。
    单纯看「杜林在不在句子里」分不出来——而语料里恰好有个干员就叫杜林。

    【改用位置判断】
    如果这个名字前面紧跟着字段提示（「种族是」「出身地是」「职业为」…），
    那它就是字段的值；否则才算提到干员。

    只有它「不是任何字段值」、或者「虽然也是字段值但另外单独被提到」时，
    才当作干员名来限制范围。
    """
    idx = question.find(name)
    while idx != -1:
        before = question[:idx]
        for f in FILTER_FIELDS:
            for suffix in (f + "是", f + "为", f + "有", f + "的", f):
                if before.endswith(suffix):
                    return True      # 前面是字段提示 -> 这是值，不是干员名
        idx = question.find(name, idx + 1)
    return False


def _masked(question, spans):
    """把 question 里 spans 覆盖的字符换成占位符，返回一个新串。

    spans 是 [(起点, 终点), ...] —— 已知是「字段值」的片段。
    把它们挖掉之后再去找干员名，就不会把字段值里的字当成干员名。

    为什么必须这么做：语料里有单字干员名（「W」「林」），
    问「种族是杜林的干员」时，「林」会匹配到「杜林」里面的那个「林」，
    结果把范围限制到一个叫「林」的干员，和种族条件取交集变成空。
    """
    chars = list(question)
    for a, b in spans:
        for i in range(a, min(b, len(chars))):
            chars[i] = "\u0000"
    return "".join(chars)


def is_field_value(question, name, spans):
    """判断 name 是否只作为「字段的值」出现在问题里（而不是被提到作为干员）。

    做法：找出 name 在问题里所有出现位置的起止区间，
    看是不是**每一个**出现位置都被字段值区间覆盖了。
    如果全部被覆盖，说明它只是值；只要有一处露在外面，就算提到干员。
    """
    idx, total, covered = question.find(name), 0, 0
    while idx != -1:
        total += 1
        end = idx + len(name)
        if any(a <= idx and end <= b for a, b in spans):
            covered += 1
        idx = question.find(name, idx + 1)
    return total > 0 and covered == total


def build_value_index(rows):
    """把每个字段的所有取值收集成集合，供「按问题里的词反查」用。

    为什么要反查，而不是「拿数据值去问题里找」：
      数据里的值是「炎」，用户会说「炎国」；数据里是「东国」，用户可能说「东」。
      如果要求「数据值必须原样出现在问题里」，这两种说法都会漏。
      反过来：把问题里的词拿出来去数据值里找包含关系，才能兼容这些说法。
    """
    idx = {}
    for f in FILTER_FIELDS:
        vals = set()
        for r in rows:
            v = r.get(f)
            if not v:
                continue
            for part in re.split(r"›|、|[/、,，\s]+", str(v)):
                part = part.strip()
                if part:
                    vals.add(part)
        idx[f] = vals
    return idx


def find_value_in_question(question, val):
    """在问题里找这个取值，返回出现的起始位置；找不到返回 None。

    三种匹配方式，按可靠性排序：
      1. 原样出现：问「维多利亚」，数据值就是「维多利亚」
      2. 数据值在问题里，后面跟着「种类后缀」：问「炎国」「杜林族」，数据值是「炎」「杜林」
      3. 问题里的词是数据值的前缀：问「东」，数据值是「东国」
         —— 只在问题里的词 ≥ 2 字时用，避免单字误命中
    """
    i = question.find(val)
    if i != -1:
        return i
    for m in re.finditer(re.escape(val), question):
        nxt = question[m.end():m.end() + 1]
        if nxt in ("国", "族", "人", "的", "势力", "阵营"):
            return m.start()
    if len(val) >= 2:
        for m in re.finditer(r"[\u4e00-\u9fff]{2,}", question):
            w = m.group(0)
            if len(w) < len(val) and val.startswith(w):
                return m.start()
    return None


def extract_filters(question, rows, value_index=None):
    """从问题里找出筛选条件。

    两类条件：
      1. 字段值：先看问题里提到了哪些「字段名或别名」（阵营 -> 势力），
         再只在那些字段的取值集合里找值。这样「阵营是炎国」能正确
         映射到「势力=炎」，而不是靠「阵营」这个词去撞字段名。
      2. 干员名：问题里提到具体干员时，用来限制范围。

    字段被提到时只在该字段里找值，没提到任何字段时才全字段搜。
    这解决了「一个词同时是多个字段的值」的歧义：
    问「德拉克种族」时因为提到了「种族」，就只在种族里找。
    """
    if value_index is None:
        value_index = build_value_index(rows)

    filters = {}
    spans = []

    # ---- 第一步：问题里提到了哪些字段（含别名）----
    # 被提到的字段只是「优先试」，不是「只在它们里面找」。
    # 为什么不能只在里面找：问「散射手都有谁？分别是几星？」时，
    # 「几星」让「星级」被提到，但真正的筛选条件是分支=散射手。
    # 如果只搜星级字段就找不到值，条件就丢了——实测这条会退化成向量检索。
    mentioned = [f for f in FIELD_ALIASES if
                 any(a in question for a in FIELD_ALIASES[f])]
    # 被提到的排前面，其余按 FIELD_PRIORITY 兜底
    order = sorted(FILTER_FIELDS, key=lambda f: (
        0 if f in mentioned else 1,
        FIELD_PRIORITY.index(f) if f in FIELD_PRIORITY else 99))

    # ---- 第二步：按上面的顺序找值 ----
    # 关键规则：同一段问题文字只能被一个字段占用。
    # 为什么需要：问题「罗德岛有哪些干员」里，「罗德岛」这三个字
    # 同时是「势力」「隐藏势力」「出身地」三个字段的合法取值。
    # 如果三个字段都拿它当条件去取交集，结果必然是空——
    # 实测就是这样：问「罗德岛有哪些干员」返回「没有符合条件的干员」。
    # 而用户说「罗德岛」时想表达的是势力，所以让高优先级字段先占位。
    used = list(spans)
    for f in order:
        if f in filters:
            continue
        # 同一个字段里优先匹配更长的值（「散射手」优先于「狙击」）
        for val in sorted(value_index.get(f, ()), key=len, reverse=True):
            pos = find_value_in_question(question, val)
            if pos is None:
                continue
            end = pos + len(val)
            # 这段文字已经被更高优先级的字段用掉了，跳过
            if any(a < end and pos < b for a, b in used):
                continue
            filters[f] = val
            spans.append((pos, end))
            used.append((pos, end))
            break

    # 问「谁能吃X加成」时，隐藏势力也算在内。
    # 隐藏势力的作用就是让该干员在战斗的势力判定中额外算这个势力，
    # 所以「予愿安洁莉娜（所属罗德岛 + 隐藏叙拉古）」吃得到叙拉古加成。
    # 这类问法要查「所属势力 + 隐藏势力」的并集，不能只查所属势力。
    if any(w in question for w in ["加成", "享受", "吃得到", "能吃到", "吃到"]) \
            and "势力" in filters:
        filters["_union_faction"] = filters.pop("势力")

    # 星级：从问题里找「N星」
    m = re.search(r"([一二三四五六1-6])\s*星", question)
    if m:
        cn = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
        s = m.group(1)
        filters["星级"] = cn.get(s, int(s) if s.isdigit() else None)
        spans.append((m.start(), m.end()))

    # 干员名：把字段值挖掉之后再匹配，单字名（W、林）也不会误伤
    masked = _masked(question, spans)
    names = [n for r in rows
             for n in [r.get("名称") or r.get("_file")]
             if n and len(n) >= 2 and n in masked]
    if names:
        # 去掉被更长名字包含的短名字
        names = [n for n in names if not any(n != o and n in o for o in names)]
        filters["_names"] = names
    return filters


# 章节关键词：问题里出现这些词，且前面有具体干员名时，
# 就把该干员该章节下的所有块都返回（而不是靠相似度排序取前 K 个）。
#
# 为什么需要：「凯尔希有什么技能」这类问题的正确答案是她有 3 个技能，
# 但向量检索只返回「最像的 K 个块」——实测它把「招募」「综合检测」排前面，
# 真正的技能块只进来 1 个，答案就残了。
# 这和「散射手都有谁」是同一类问题：**要的是穷举，不是相似度**。
SECTION_KEYWORDS = {
    "技能": ["技能", "技能效果", "技能名"],
    "天赋": ["天赋"],
    "潜能": ["潜能"],
    "基建技能": ["基建技能", "后勤技能"],
    "属性": ["属性", "面板", "数值", "攻击力", "防御力", "生命上限", "法抗"],
}
# 明确表示「要全部」的问法
ALL_SIGNALS = ["有什么技能", "有哪些技能", "都有什么技能", "技能是什么", "技能有哪些",
               "有什么天赋", "有哪些天赋", "天赋是什么", "有什么潜能",
               "有哪些属性", "属性是什么"]


def detect_section_query(question, rows):
    """判断是不是「问某个干员的整个章节」。

    返回 (干员名, 章节名)，不是这类问题就返回 (None, None)。
    """
    # 必须先锁定一个具体干员，否则「技能都有什么用」这种泛问不该走这里
    names = []
    for r in rows:
        n = r.get("名称") or r.get("_file")
        if n and len(n) >= 2 and n in question:
            names.append(n)
    if not names:
        return None, None
    names = [n for n in names if not any(n != o and n in o for o in names)]
    # 取最长的那个（问「凯尔希」时别被更短的同名干扰）
    op = max(names, key=len)

    # 看问的是哪个章节：优先匹配更长的关键词（「基建技能」优先于「技能」）
    hit_section, hit_len = None, 0
    for sec, kws in SECTION_KEYWORDS.items():
        for kw in kws:
            if kw in question and len(kw) > hit_len:
                hit_section, hit_len = sec, len(kw)
    if not hit_section:
        return None, None
    # 「基建技能」和「技能」容易混：如果问题里明确写了「基建」或「后勤」，按基建算
    if hit_section == "技能" and any(w in question for w in ["基建", "后勤"]):
        hit_section = "基建技能"
    return op, hit_section


def run_structured(question):
    """走结构化路径：筛选 + 可选排序，返回 (文本结果, 说明)"""
    sq, rows = load_rows()
    intent = detect_intent(question)
    filters = extract_filters(question, rows)
    names = filters.pop("_names", [])

    res = rows
    if names:
        res = [r for r in res if (r.get("名称") or r.get("_file")) in names]

    for f, v in filters.items():
        if f == "星级":
            res = [r for r in res if r.get("星级") == v]
        elif f == "势力":
            # 势力按「任一层级命中」匹配，且「炎」要能匹配到「炎-龙门」
            res = [r for r in res if sq.match_faction(r, v, include_hidden=False)]
        elif f == "_union_faction":
            # 「谁能吃X加成」：所属势力和隐藏势力都算
            res = [r for r in res if sq.match_faction(r, v, include_hidden=True)]
        elif f == "隐藏势力":
            res = [r for r in res if v in r.get("隐藏势力列表", [])]
        else:
            res = [r for r in res if v in str(r.get(f, ""))]

    parts = []
    if names:
        parts.append("干员=" + "/".join(names))
    parts += [f"{k}={v}" for k, v in filters.items()]
    desc = "、".join(parts) or "（无筛选条件）"

    # ---- 排序类：按数值字段排序 ----
    if intent == "rank":
        field = next((NUM_FIELDS[k] for k in NUM_FIELDS if k in question), None)
        if not field:
            return None, f"{desc}（是排序类问题，但没识别出要比哪个数值）"

        sq.enrich_with_attrs(res)
        res = [r for r in res if r.get("_attrs", {}).get(field) is not None]
        if not res:
            return None, f"{desc}（没有干员有「{field}」这个数值）"

        # 判断是求最高还是最低
        low = any(w in question for w in ["最低", "最小", "更少", "更低"])
        res.sort(key=lambda r: r["_attrs"][field], reverse=not low)
        # 比较类问题（A和B谁高）只看前几名
        topn = 5 if ("和" in question or "跟" in question or "比" in question) else 10
        res = res[:topn]
        desc += f"，按「{field}」{'升序' if low else '降序'}排列"

        lines = [f"{'干员':<18}{'星级':<6}{field}"]
        lines.append("-" * 44)
        for r in res:
            lines.append(f"{r.get('名称','?'):<18}{str(r.get('星级','?')):<6}"
                         f"{r['_attrs'][field]:g}")
        return "\n".join(lines), desc

    # ---- 列举类 ----
    if not filters:
        return None, "没识别出可筛选的字段"

    if not res:
        return f"（没有符合条件的干员：{desc}）", desc
    return sq.format_rows(res), desc


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    tests = sys.argv[1:] or [
        "散射手都有谁？分别是几星？",
        "六星狙击有哪些？",
        "执旗手一共几个？",
        "莱伊和提丰谁攻击力更高？",
        "凯尔希有什么技能？",
        "找一个性格冷淡话少的干员",
    ]
    for q in tests:
        intent = detect_intent(q)
        print("=" * 70)
        print(f"问题：{q}")
        print(f"识别意图：{intent}")
        if intent in ("list", "rank"):
            text, desc = run_structured(q)
            print(f"筛选条件：{desc}")
            if text:
                print(text)
            else:
                print("  -> 结构化路径不适用，应转向量检索")
        else:
            print("  -> 走向量检索（4_search.py）")
        print()
