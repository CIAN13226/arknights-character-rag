# -*- coding: utf-8 -*-
"""
从 PRTS（明日方舟 Wiki，MediaWiki 建站）批量抓取干员页面，转成 RAG 用的纯文本语料。

取数思路（这段比代码重要，面试也问得到）：
  1. 走 MediaWiki 官方 API（action=query / action=parse），不解析页面 HTML 外壳，
     避免把导航、页脚、样式表混进语料
  2. 页面内容模型是 wikitext。正文按 ==章节== 组织，核心数据在章节内：
        ==干员信息==  -> {{CharinfoV2|干员名=..|职业=..}}      结构化字段
        ==技能==      -> {{技能|技能名=..|技能类型1=..}}       嵌套模板，要配对花括号
        ==天赋==      -> {{天赋列表3|天赋1=..|天赋1效果=..}}
        ==干员档案==  -> 剧情文本
     所以做法是「先按章节切，再在章节里解模板」，而不是全文正则扫模板名
  3. 材料、语音、立绘这类章节对检索没价值，直接丢掉，语料更干净
  4. 请求之间加 sleep，失败重试。别把人家的站打挂

用法：
    python fetch_prts.py --limit 3 --dry-run     # 只解析不写文件，看输出
    python fetch_prts.py --limit 30              # 抓 30 个干员写入 corpus/
"""
import os
import re
import time
import html
import argparse
import requests

API = "https://prts.wiki/api.php"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

# 完整浏览器请求头。实测缺 Referer 会被 WAF 挡成 403
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://prts.wiki/",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

SLEEP = 1.2

# 要保留的章节（顺序即输出顺序）；没列进来的一律丢掉
KEEP_SECTIONS = ["干员信息", "获得方式", "属性", "天赋", "潜能提升",
                 "技能", "后勤技能", "干员档案", "悖论模拟"]


def api_get(params, retry=3):
    for i in range(retry):
        try:
            r = SESSION.get(API, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            print(f"    [HTTP {r.status_code}] 重试 {i+1}/{retry}")
        except Exception as e:
            print(f"    [异常] {e}")
        time.sleep(SLEEP * (i + 1))
    return None


def list_operators(limit):
    """取「干员」分类下的页面标题，处理 MediaWiki 分页"""
    titles, cont = [], None
    while len(titles) < limit:
        p = {"action": "query", "list": "categorymembers",
             "cmtitle": "Category:干员", "cmlimit": 500, "format": "json"}
        if cont:
            p["cmcontinue"] = cont
        data = api_get(p)
        if not data:
            break
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        time.sleep(SLEEP)
    return titles[:limit]


def get_wikitext(title):
    data = api_get({"action": "parse", "page": title, "prop": "wikitext",
                    "format": "json", "redirects": 1})
    if not data or "parse" not in data:
        return None
    return data["parse"]["wikitext"]["*"]


# ---------------- wikitext 解析 ----------------

def split_sections(wt):
    """按 == 标题 == 切分，返回 {章节名: 正文}"""
    out = {}
    marks = list(re.finditer(r"\n(={2,4})\s*([^=\n]+?)\s*\1\s*\n", wt))
    for i, m in enumerate(marks):
        name = m.group(2).strip()
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(wt)
        out.setdefault(name, wt[start:end])
    return out


def find_templates(text, name):
    """抠出所有 {{name|...}} 的内容（不含模板名、也不含前导的 |）。
    正确处理嵌套花括号。返回的字符串统一以「第一个参数」开头，
    所以调用方要自己补一个 "|" 再交给 parse_params。"""
    res = []
    for m in re.finditer(r"\{\{\s*" + re.escape(name) + r"\s*(\|)?", text):
        i, depth, start = m.end(), 0, m.end()
        while i < len(text):
            two = text[i:i + 2]
            if two == "{{":
                depth += 1; i += 2; continue
            if two == "}}":
                if depth == 0:
                    res.append(text[start:i]); break
                depth -= 1; i += 2; continue
            i += 1
    return res


def parse_params(body):
    """{{模板|k=v|k=v}} 的 body -> dict

    顺序很重要：先把值里的嵌套模板剥成纯文字，再按顶层 | 切分。
    如果反过来（先切再洗），切分时嵌套的 {{ }} 会打乱深度计数，
    导致后面的字段被整段吞掉——技能效果丢失的根因就在这里。
    """
    body = unwrap_templates(body)
    parts, depth, cur = [], 0, ""
    for i, ch in enumerate(body):
        two = body[i:i + 2]
        if two == "{{":
            depth += 1
        elif two == "}}":
            depth -= 1
        if ch == "|" and depth <= 0:
            parts.append(cur); cur = ""; continue
        cur += ch
    parts.append(cur)
    out = {}
    for part in parts:
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = clean(v.strip())
    return out


def _split_top_level(inner):
    """按顶层 | 切分（跳过嵌套模板里的 |）"""
    parts, depth, cur = [], 0, ""
    i = 0
    while i < len(inner):
        two = inner[i:i + 2]
        if two == "{{":
            depth += 1
        elif two == "}}":
            depth -= 1
        if inner[i] == "|" and depth == 0:
            parts.append(cur); cur = ""; i += 1; continue
        cur += inner[i]; i += 1
    parts.append(cur)
    return parts


def _unwrap_one(s):
    """把 s 里『最内层』的模板替换成它的文字参数。返回 (新串, 是否发生了替换)"""
    # 找内部不含花括号的模板：{{ 任意名字 | 参数 }}
    # 名字部分要允许 {{+ ...}} {{* ...}} 这类「符号名 + 空格」的写法，
    # 所以用 [^{}]*? 而不是 [^{}]*（后者要求名字后面紧跟 | 或 }}）
    best = None
    for m in re.finditer(r"\{\{([^{}]*?)\}\}", s):
        if best is None or m.start() < best.start():
            best = m
    if best is None:
        return s, False
    inner = best.group(1)
    if "|" not in inner:
        return s[:best.start()] + s[best.end():], True   # 没有参数，整体删掉
    name, _, rest = inner.partition("|")
    parts = _split_top_level(rest)
    # 保留「不含 = 号」的参数，那才是要显示的文字
    texts = [p.strip() for p in parts if "=" not in p and p.strip()]
    return s[:best.start()] + " ".join(texts) + s[best.end():], True


def unwrap_templates(s):
    """把 {{模板|a|b|文字}} 变成它的文字参数。

    为什么这么做：wiki 里到处是 {{color|#0098DC|3}}、{{*|60%|+60%}} 这类
    「只负责排版」的模板，真正有用的文字在参数里。
    直接删模板会把数值和术语一起删掉，语料就残缺了。

    关键：必须「从最内层往外」剥。
    早期版本用一条正则从最外层开始替换，嵌套模板（如
    {{*|120%|{{color|#0098DC|+120%}}}}）会被切错边界，
    产生多余的 }}，把后面整段字段一起吞掉——技能效果全部丢失就是这么来的。
    """
    # 每次迭代只剥掉一个「最内层」模板，所以循环次数要够。
    # 一个技能模板里可能有几百个嵌套小模板（{{color}} {{*}} {{+}} …），
    # 早期版本写死 40 次，结果剥到一半就停了，剩下的原样留在文本里
    # （技能效果里出现 {{+ 这种残渣就是这么来的）。
    max_iter = s.count("{{") + 10
    for _ in range(max_iter):
        s, changed = _unwrap_one(s)
        if not changed:
            break
    return s


def clean(s):
    """去掉 wiki 标记，留下可读文本"""
    if not s:
        return ""
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = s.replace("<br/>", "\n").replace("<br />", "\n").replace("<br>", "\n")
    s = re.sub(r"<[^>]+>", "", s)
    s = unwrap_templates(s)                # 模板的参数值要留下来
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"'''?", "", s)
    # 清掉从模板参数里带出来的噪声
    s = re.sub(r"#[0-9A-Fa-f]{6}\b", "", s)          # 颜色值
    s = re.sub(r"(?<![A-Za-z0-9_.])ba\.[A-Za-z_]+\s*", "", s)   # ba.sluggish 这类内部标记
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"[ \t]+([，。；：、）])", r"\1", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    s = re.sub(r"（\s*[—\-–]*\s*）", "", s)   # 清掉「（）」「（——）」这种空括号
    return s.strip()


def line(label, val):
    v = clean(val)
    return f"{label}：{v}" if v else None


def parse_skills(sec):
    """==技能== 章节 -> 每个技能一组字段"""
    out = []
    for body in find_templates(sec, "技能"):
        # find_templates 返回的是 body（| 之后的内容），重新拼一个 dict
        p = parse_params("|" + body)
        if not p.get("技能名"):
            continue
        buf = [f"技能名：{clean(p.get('技能名', ''))}"]
        for k, label in [("技能类型1", "回复类型"), ("技能类型2", "触发方式")]:
            v = clean(p.get(k, ""))
            if v:
                buf.append(f"{label}：{v}")
        # 技能1描述..技能7描述 是同一技能的不同等级，取 7 级和专精3 各一条最有代表性
        for k, label in [("技能7描述", "7级效果"), ("技能专精3描述", "专精三效果")]:
            v = clean(p.get(k, ""))
            if v:
                buf.append(f"{label}：{v}")
        for k, label in [("技能7消耗", "7级技力消耗"), ("技能7持续", "7级持续时间")]:
            v = clean(p.get(k, ""))
            if v:
                buf.append(f"{label}：{v}")
        out.append("；".join(buf))
    return out


def parse_talents(sec):
    """==天赋== 章节 -> 天赋列表。

    页面里的编号方式是「同一模板内，N 个连续编号 = 同一个天赋的不同等级」，
    用 {{#if:}} 切换。所以不能直接叫「天赋N」，否则第二个天赋的编号会和第一个撞车。
    做法：按天赋名分组，同名视为同一天赋的不同等级。

    同一份内容在「天赋列表3 / 天赋列表2」里会重复出现，取第一个能解析出内容的就停。
    """
    for name in ["天赋列表3", "天赋列表2", "天赋列表"]:
        groups, order = {}, []
        for body in find_templates(sec, name):
            p = parse_params("|" + body)
            i = 1
            while f"天赋{i}" in p or f"天赋{i}效果" in p:
                nm = clean(p.get(f"天赋{i}", "")) or f"天赋{i}"
                cond = clean(p.get(f"天赋{i}条件", ""))
                eff = clean(p.get(f"天赋{i}效果", ""))
                if eff or nm:
                    lvl = f"（{cond}）" if cond else ""
                    if nm not in groups:
                        groups[nm] = []
                        order.append(nm)
                    item = f"{lvl}：{eff}" if eff else lvl
                    if item not in groups[nm]:
                        groups[nm].append(item)
                i += 1
        if order:
            out = []
            for idx, nm in enumerate(order, 1):
                out.append(f"天赋{idx}「{nm}」")
                for item in groups[nm]:
                    out.append(f"    {item}")
            return out
    return []


def parse_potential(sec):
    out = []
    for body in find_templates(sec, "潜能提升"):
        p = parse_params("|" + body)
        for k in sorted(p):
            if k.startswith("潜能"):
                v = clean(p[k])
                if v:
                    out.append(f"{k}：{v}")
    return out


def parse_base_skills(sec):
    out = []
    for body in find_templates(sec, "后勤技能"):
        p = parse_params("|" + body)
        for k in sorted(p):
            if k.startswith("后勤技能") and not k.endswith("阶段") and not k.endswith("等级"):
                v = clean(p[k])
                if v:
                    out.append(f"{k}：{v}")
    return out


def parse_faction(wt):
    """从 {{属性}} 模板里取「所属势力」和「隐藏势力」。

    【为什么不能继续用 CharinfoV2 里的所属国家/所属组织】
      那两个是 wiki 模板的内部参数名，PRTS 页面上根本不显示这两个词。
      页面上显示的是「所属势力」，数据源在 {{属性}} 模板里，
      用逗号分隔层级：
          死芒        |所属势力=维多利亚,塔拉      -> 页面显示 维多利亚 › 塔拉
          银灰        |所属势力=谢拉格,喀兰贸易     -> 页面显示 谢拉格 › 喀兰贸易
          芬          |所属势力=罗德岛,行动预备组A1  -> 页面显示 罗德岛 › 行动预备组A1
          予愿安洁莉娜   |所属势力=罗德岛 |隐藏势力=叙拉古

    【隐藏势力是什么】
      它是「附加势力」——该干员在战斗的势力判定中**额外也算**这个势力。
      例：予愿安洁莉娜显示是罗德岛，隐藏势力是叙拉古，
      所以可露希尔的罗德岛加成和荒芜拉普兰德的叙拉古加成她都能吃到。
      多个值用 ;; 分隔，例如煌是「炎;;维多利亚」，炎和维多利亚的加成都能吃。

    返回 (所属势力列表, 隐藏势力列表)
    """
    bodies = find_templates(wt, "属性")
    if not bodies:
        return [], []
    p = parse_params("|" + bodies[0])

    def split_multi(v, seps):
        if not v:
            return []
        pat = "|".join(re.escape(s) for s in seps)
        return [x.strip() for x in re.split(pat, v) if x.strip()]

    # 所属势力用逗号分隔（表示层级），隐藏势力用 ;; 分隔（表示并列）
    owned = split_multi(p.get("所属势力", ""), [",", "，", "、"])
    hidden = split_multi(p.get("隐藏势力", ""), [";;", ";"])
    return owned, hidden


def parse_charinfo(wt):
    """CharinfoV2 模板在 ==干员信息== 标题之前，所以要全文搜，不能只在章节里找。

    注意：模板里的「稀有度」是 0 起始的（0=一星，5=六星），
    直接写进语料会整体低一星（凯尔希会变成 5 星），所以这里要 +1 换算成实际星级。

    所属国家/所属组织 不再从这里取——改用 {{属性}} 里的「所属势力 / 隐藏势力」，
    那才是 PRTS 页面上显示给玩家的字段。详见 parse_faction。
    """
    for body in find_templates(wt, "CharinfoV2"):
        p = parse_params("|" + body)
        fields = [("干员名", "名称"), ("稀有度", "星级"), ("职业", "职业"), ("分支", "分支"),
                  ("位置", "站位"), ("标签", "标签"), ("特性", "特性")]
        got = []
        for k, lbl in fields:
            v = p.get(k, "")
            if k == "稀有度":
                v = _rarity_to_star(v)
            got.append(line(lbl, v))
        got = [x for x in got if x]
        if got:
            return got
    return []


def _rarity_to_star(v):
    """模板稀有度 -> 实际星级（0 起始，所以要 +1）"""
    v = (v or "").strip()
    if not v.isdigit():
        return v
    return str(int(v) + 1)


def parse_elite_intro(wt):
    """精英化立绘介绍里往往有一句干员简介，能当语料"""
    out = []
    for body in find_templates(wt, "CharinfoV2"):
        p = parse_params("|" + body)
        for k in ["精英0介绍", "精英1介绍", "精英2介绍"]:
            v = clean(p.get(k, ""))
            if len(v) > 20:
                out.append(v)
        if out:
            break
    return out


def page_to_text(title, wt):
    secs = split_sections(wt)
    lines = [f"《明日方舟》干员：{title}", ""]

    info = parse_charinfo(wt)
    if info:
        lines += ["【基本信息】"] + info + [""]

    intro = parse_elite_intro(wt)
    if intro:
        lines += ["【干员介绍】", " ".join(intro[:2])[:400], ""]

    if "获得方式" in secs:
        v = clean(re.sub(r"\{\{[^{}]*\}\}", "", secs["获得方式"]))
        if v:
            lines += ["【获得方式】", v[:300], ""]

    if "属性" in secs:
        v = clean(re.sub(r"\{\{[^{}]*\}\}", "", secs["属性"]))
        if len(v) > 20:
            lines += ["【属性】", v[:600], ""]

    if "天赋" in secs:
        t = parse_talents(secs["天赋"])
        if t:
            lines += ["【天赋】"] + t + [""]

    if "潜能提升" in secs:
        t = parse_potential(secs["潜能提升"])
        if t:
            lines += ["【潜能提升】"] + t + [""]

    if "技能" in secs:
        t = parse_skills(secs["技能"])
        if t:
            lines += ["【技能】"] + t + [""]

    if "后勤技能" in secs:
        t = parse_base_skills(secs["后勤技能"])
        if t:
            lines += ["【基建技能】"] + t + [""]

    for sec_name, label in [("干员档案", "干员档案"), ("悖论模拟", "悖论模拟")]:
        if sec_name in secs:
            v = clean(secs[sec_name])
            v = re.sub(r"\s+", " ", v)
            if len(v) > 30:
                lines += [f"【{label}】", v[:1500], ""]

    return "\n".join(lines).strip()


def run_dump(args):
    """全量抓取并输出到 corpus_db / corpus_lore。
    复用 dump_operators 里已经验证过的解析器，避免两套逻辑不一致。"""
    import dump_operators as D

    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_db")
    lore_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_lore")
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(lore_dir, exist_ok=True)

    print(f"[1/3] 取干员列表（最多 {args.limit} 个）")
    titles = list_operators(args.limit)
    if args.skip:
        titles = titles[args.skip:]
        print(f"      跳过前 {args.skip} 个，本次处理 {len(titles)} 个")
    print(f"      拿到 {len(titles)} 个")

    stat = []
    skipped = 0
    for i, title in enumerate(titles, 1):
        safe = re.sub(r'[\\/:*?"<>|]', "_", title)
        db_path = os.path.join(db_dir, f"{safe}.md")
        lore_path = os.path.join(lore_dir, f"{safe}.md")
        # 断点续跑：两个文件都在、且内容确实是「当前版本」才跳过。
        # 判断条件踩过两次坑，所以这里写得严一点：
        #   ① db 必须含 ## 技能 段
        #   ② db 里不能有没剥干净的 {{ 残渣
        #   ③ lore 里也不能有 {{ 残渣
        #   ④ 有技能模板的干员，db 里必须出现「7级：」效果行
        #      —— 早期解析器有 bug，会把部分技能效果吃掉，
        #         那种残缺文件用前三条检查不出来，会被误当成已完成
        if os.path.exists(db_path) and os.path.exists(lore_path):
            try:
                db_txt = open(db_path, encoding="utf-8").read()
                lo_txt = open(lore_path, encoding="utf-8").read()
            except Exception:
                db_txt, lo_txt = "", ""
            skill_seg = ""
            if "## 技能" in db_txt:
                skill_seg = db_txt.split("## 技能")[1].split("\n##")[0]
            n_names = skill_seg.count("技能名：")
            n_effect = skill_seg.count("7级：")
            # 有技能的干员，效果行数必须和技能名数一样多，否则就是解析残缺。
            # 这条是最直接的判据：早期解析器把技能效果吃掉时，
            # 技能名还在（第一行），效果行没了，两个数就对不上。
            #
            # 再加两条：语料里不该出现「所属国家」这个旧字段名
            # （PRTS 页面上没这个词，现在用 {{属性}} 里的「所属势力/隐藏势力」），
            # 而且「所属势力」必须写在 ## 基本信息 段里——
            # 曾经有一版写到属性段去了，靠这两条能识别出来。
            basic_seg = ""
            if "## 基本信息" in db_txt:
                basic_seg = db_txt.split("## 基本信息")[1].split("\n##")[0]
            stale = ("所属国家：" in db_txt) or ("所属势力：" not in basic_seg)
            ok_db = ("## 技能" in db_txt) and ("{{" not in db_txt) and \
                    (n_names == 0 or n_effect == n_names) and not stale
            ok_lo = ("{{" not in lo_txt)
            if ok_db and ok_lo:
                skipped += 1
                if skipped <= 3:
                    print(f"[2/3] ({i}/{len(titles)}) {title}  已抓过，跳过")
                continue

        with_voice = not args.no_voice
        try:
            db, lo, err = D.dump_one(title, with_voice)
        except Exception as e:
            err, db, lo = f"异常 {e}", None, None
        if err or not db:
            print(f"[2/3] ({i}/{len(titles)}) {title}  {err}")
            stat.append((title, 0, 0, err))
            continue

        with open(db_path, "w", encoding="utf-8") as f:
            f.write(db)
        with open(lore_path, "w", encoding="utf-8") as f:
            f.write(lo)
        has_attr = "无属性数据" not in db
        has_arch = "无档案数据" not in lo
        stat.append((title, len(db), len(lo), ""))
        print(f"[2/3] ({i}/{len(titles)}) {title:<14} 结构化 {len(db):>5} 字  "
              f"剧情 {len(lo):>6} 字  {'属性有' if has_attr else '属性无'} "
              f"{'档案有' if has_arch else '档案无'}")
        time.sleep(fp_sleep())

    if skipped:
        print(f"\n      跳过已抓过的 {skipped} 个")
    ok = [s for s in stat if not s[3]]
    bad = [s for s in stat if s[3]]
    print(f"\n[3/3] 完成：成功 {len(ok)} 篇，失败 {len(bad)} 篇")
    if bad:
        print("      失败：" + "、".join(f"{t}({e})" for t, _, _, e in bad[:20]))
    if ok:
        print(f"      结构化平均 {sum(s[1] for s in ok)//len(ok)} 字，"
              f"剧情平均 {sum(s[2] for s in ok)//len(ok)} 字")
    print(f"\n输出目录：\n  {db_dir}\n  {lore_dir}")


def fp_sleep():
    return SLEEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dump", action="store_true",
                    help="用完整解析器输出到 corpus_db/ 与 corpus_lore/（推荐）")
    ap.add_argument("--skip", type=int, default=0,
                    help="跳过前 N 个（断点续跑用，比如 --skip 200）")
    ap.add_argument("--no-voice", action="store_true", help="不抓语音子页面（省一半时间）")
    args = ap.parse_args()

    if args.dump:
        return run_dump(args)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[1/3] 取干员列表（最多 {args.limit} 个）")
    titles = list_operators(args.limit)
    print(f"      拿到 {len(titles)} 个")

    ok, fail = 0, []
    for i, title in enumerate(titles, 1):
        wt = get_wikitext(title)
        if not wt:
            print(f"[2/3] ({i}/{len(titles)}) {title}  抓取失败"); fail.append(title); continue
        text = page_to_text(title, wt)
        if len(text) < 300:
            print(f"[2/3] ({i}/{len(titles)}) {title}  解析后仅 {len(text)} 字，跳过")
            fail.append(title); continue
        if args.dry_run:
            print(f"[2/3] ({i}/{len(titles)}) {title}  解析出 {len(text)} 字")
            print("=" * 60)
            print(text[:2500])
            print("=" * 60)
            if i >= 2:
                break
            time.sleep(SLEEP)
            continue
        safe = re.sub(r'[\\/:*?"<>|]', "_", title)
        with open(os.path.join(OUT_DIR, f"干员_{safe}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        ok += 1
        print(f"[2/3] ({i}/{len(titles)}) {title}  -> {len(text)} 字")
        time.sleep(SLEEP)

    if not args.dry_run:
        print(f"\n[3/3] 完成：成功 {ok} 篇，失败 {len(fail)} 篇")
        if fail:
            print("      失败：" + ", ".join(fail))


if __name__ == "__main__":
    main()
