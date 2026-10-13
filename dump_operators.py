# -*- coding: utf-8 -*-
"""
抓指定干员，输出到 dump/ 目录，每个干员一个文件，按「结构化」和「剧情」分区。

这是验证解析器用的脚本。确认没问题后，全量产出会走 corpus_db / corpus_lore 两个目录。

用法：
    python dump_operators.py                     # 内置测试名单
    python dump_operators.py 凯尔希 星熊          # 指定干员
    python dump_operators.py --no-voice          # 不抓语音（省时间）
"""
import os
import re
import sys
import time
import fetch_prts as fp

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "dump")

DEFAULT_OPS = ["克莱门莎", "晓歌", "W", "丰川祥子", "维什戴尔", "桃金娘",
               "罗小黑", "凯尔希", "夜刀", "星熊", "斩业星熊"]

STAT_LABELS = [("生命上限", "生命"), ("攻击", "攻击"), ("防御", "防御"), ("法术抗性", "法抗")]


# ---------------- 结构化部分 ----------------

def parse_attr(wt):
    bodies = fp.find_templates(wt, "属性")
    if not bodies:
        return None
    p = fp.parse_params("|" + bodies[0])
    if not p:
        return None
    lines = []
    base = []
    for k in ["再部署", "部署费用", "阻挡数", "攻击速度"]:
        v = fp.clean(p.get(k, ""))
        if v:
            base.append(f"{k} {v}")
    if base:
        lines.append("  ".join(base))

    for phase, label in [("精英0_1级", "精英0 1级"), ("精英0_满级", "精英0 满级"),
                         ("精英1_满级", "精英1 满级"), ("精英2_满级", "精英2 满级")]:
        vals = []
        for key, name in STAT_LABELS:
            v = fp.clean(p.get(f"{phase}_{key}", ""))
            if v:
                vals.append(f"{name} {v}")
        if vals:
            lv = fp.clean(p.get(phase, "")) if phase.endswith("满级") else ""
            suffix = f"（{lv}级）" if lv else ""
            lines.append(f"{label}{suffix}：" + " / ".join(vals))

    tr = [f"{name} +{fp.clean(p.get(f'信赖加成_{key}', ''))}"
          for key, name in STAT_LABELS
          if fp.clean(p.get(f"信赖加成_{key}", "")) not in ("", "0")]
    if tr:
        lines.append("信赖加成：" + " / ".join(tr))

    mods = []
    for i in ["", "1", "2", "3"]:
        nm = fp.clean(p.get(f"模组{i}名", ""))
        if nm:
            data = fp.clean(p.get(f"模组{i}数据", ""))
            mods.append(nm + (f"（{data}）" if data else ""))
    if mods:
        lines.append("模组：" + "；".join(mods))

    # 势力不在这里输出。早期版本在这里加过「所属势力/隐藏势力」，
    # 后来改到 ## 基本信息 段（dump_one 里用 fp.parse_faction 处理），
    # 两处都写会重复。属性段只放数值。
    return lines


def parse_skill_effects(sec):
    """技能只取 7 级与专精三"""
    out = []
    for body in fp.find_templates(sec, "技能"):
        p = fp.parse_params("|" + body)
        name = fp.clean(p.get("技能名", ""))
        if not name:
            continue
        t1, t2 = fp.clean(p.get("技能类型1", "")), fp.clean(p.get("技能类型2", ""))
        head = f"技能名：{name}"
        if t1 or t2:
            head += f"（{t1} / {t2}）"
        out.append(head)
        for prefix, label in [("技能7", "7级"), ("技能专精3", "专精三")]:
            desc = fp.clean(p.get(f"{prefix}描述", ""))
            if not desc:
                continue
            extra = []
            for k, lb in [("初始", "初始"), ("消耗", "消耗"), ("持续", "持续")]:
                v = fp.clean(p.get(f"{prefix}{k}", ""))
                if v:
                    extra.append(f"{lb}{v}")
            line = f"    {label}：{desc}"
            if extra:
                line += "（" + "，".join(extra) + "）"
            out.append(line)
    return out


# ---------------- 剧情部分 ----------------

def parse_archive(sec):
    """干员档案：体能检测 + 档案1..N 文本"""
    out = []
    for body in fp.find_templates(sec, "人员档案set"):
        p = fp.parse_params("|" + body)
        keys = ["性别", "战斗经验", "出身地", "生日", "种族", "身高",
                "矿石病感染情况", "物理强度", "战场机动", "生理耐受",
                "战术规划", "战斗技巧", "源石技艺适应性"]
        got = [f"{k}：{fp.clean(p.get(k, ''))}" for k in keys if fp.clean(p.get(k, ""))]
        if got:
            out.append(("[体能检测]", "；".join(got)))
        break

    # 注意：《人员档案》模板在同一页可能出现两次——
    # 一个只放体能/性别等基础项，另一个才放「档案1..10文本」。
    # 所以要遍历所有，取第一个真能解析出档案文本的，不能碰到就 break。
    for body in fp.find_templates(sec, "人员档案"):
        p = fp.parse_params("|" + body)
        items = []
        i = 1
        while f"档案{i}" in p:
            title = fp.clean(p.get(f"档案{i}", ""))
            cond = fp.clean(p.get(f"档案{i}条件", ""))
            text = fp.clean(p.get(f"档案{i}文本", ""))
            if text and len(text) > 10:
                head = title + (f"（{cond}）" if cond else "")
                items.append((head, re.sub(r"\s+", " ", text)))
            i += 1
        if items:
            out += items
            break
    return out


def parse_voice(title):
    """语音：抓 干员名/语音记录 子页面，取中文台词"""
    wt = fp.get_wikitext(title + "/语音记录")
    if not wt:
        return []
    out = []
    for body in fp.find_templates(wt, "VoiceTable"):
        p = fp.parse_params("|" + body)
        i = 1
        while f"标题{i}" in p or f"台词{i}" in p:
            label = fp.clean(p.get(f"标题{i}", ""))
            raw = p.get(f"台词{i}", "")
            # 台词是 {{VoiceData/word|中文|xxx}}{{VoiceData/word|日文|yyy}}...
            cn = ""
            for m in re.finditer(r"\{\{\s*VoiceData/word\s*\|([^|]*)\|", raw):
                lang = m.group(1).strip()
                if lang in ("中文", "中文-普通话", "简体中文"):
                    rest = raw[m.end():]
                    depth, buf = 0, ""
                    for j, ch in enumerate(rest):
                        two = rest[j:j + 2]
                        if two == "{{":
                            depth += 1
                        elif two == "}}":
                            if depth == 0:
                                break
                            depth -= 1
                        buf += ch
                    cn = buf.strip()
                    break
            if not cn:
                cn = fp.clean(raw)
            if cn and len(cn) > 2:
                out.append((label or f"语音{i}", re.sub(r"\s+", " ", cn)))
            i += 1
        if out:
            break
    return out


def dump_one(title, with_voice=True):
    wt = fp.get_wikitext(title)
    if not wt:
        return None, None, f"抓取失败"
    secs = fp.split_sections(wt)

    db, lo = [], []

    db.append(f"# {title}")
    db.append("")
    db.append("## 基本信息")
    for line in (fp.parse_charinfo(wt) or []):
        db.append("- " + line)
    # 势力信息来自 {{属性}} 模板，和上面 CharinfoV2 的字段不在一处。
    # 「所属势力」用 › 连接层级（页面就这么显示）；「隐藏势力」是附加势力，
    # 战斗的势力判定中额外也算它（例如予愿安洁莉娜同时算罗德岛和叙拉古）。
    owned, hidden = fp.parse_faction(wt)
    if owned:
        db.append("- 所属势力：" + " › ".join(owned))
    if hidden:
        db.append("- 隐藏势力：" + "、".join(hidden))

    attr = parse_attr(wt)
    db += ["", "## 属性"]
    db += (["- " + x for x in attr] if attr else ["- （无属性数据）"])

    tl = fp.parse_talents(secs.get("天赋", ""))
    db += ["", "## 天赋"]
    db += (["- " + x for x in tl] if tl else ["- （无）"])

    pot = fp.parse_potential(secs.get("潜能提升", ""))
    db += ["", "## 潜能"]
    db += (["- " + x for x in pot] if pot else ["- （无）"])

    sk = parse_skill_effects(secs.get("技能", ""))
    db += ["", "## 技能（7级 / 专精三）"]
    db += (["- " + x for x in sk] if sk else ["- （无技能）"])

    bs = fp.parse_base_skills(secs.get("后勤技能", ""))
    db += ["", "## 基建技能"]
    db += (["- " + x for x in bs] if bs else ["- （无）"])

    lo.append(f"# {title} · 剧情与语音")
    lo.append("")
    lo.append("## 档案")
    arch = parse_archive(secs.get("干员档案", ""))
    if arch:
        for head, text in arch:
            lo += [f"### {head}", text, ""]
    else:
        lo.append("（无档案数据）")
        lo.append("")

    if with_voice:
        time.sleep(fp.SLEEP)
        voice = parse_voice(title)
        lo.append("## 语音台词")
        if voice:
            for label, text in voice:
                lo.append(f"- 【{label}】{text}")
        else:
            lo.append("（无语音数据）")

    return "\n".join(db), "\n".join(lo), None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    with_voice = "--no-voice" not in sys.argv
    ops = args if args else DEFAULT_OPS
    os.makedirs(OUT_DIR, exist_ok=True)

    stat = []
    for i, t in enumerate(ops, 1):
        print(f"({i}/{len(ops)}) {t}")
        db, lo, err = dump_one(t, with_voice)
        if err:
            stat.append((t, err, "", "", ""))
            continue
        safe = re.sub(r'[\\/:*?"<>|]', "_", t)
        open(os.path.join(OUT_DIR, f"{safe}_结构化.md"), "w", encoding="utf-8").write(db)
        open(os.path.join(OUT_DIR, f"{safe}_剧情.md"), "w", encoding="utf-8").write(lo)
        stat.append((t, f"{len(db)}字", f"{len(lo)}字",
                     "有" if "## 属性" in db and "无属性" not in db.split("## 属性")[1][:30] else "无",
                     "有" if "（无档案数据）" not in lo else "无"))
        if i < len(ops):
            time.sleep(fp.SLEEP)

    print(f"\n输出目录：{OUT_DIR}")
    print(f"\n{'干员':<12}{'结构化':<10}{'剧情':<10}{'属性':<6}{'档案'}")
    for t, a, b, c, d in stat:
        print(f"{t:<12}{a:<10}{b:<10}{c:<6}{d}")


if __name__ == "__main__":
    main()
