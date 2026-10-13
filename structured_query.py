# -*- coding: utf-8 -*-
"""
结构化查询：把干员的属性字段抽成表，用于「筛选/列举」类问题。

【为什么需要它】
    向量检索只能找「最像的 K 个块」，没有「全部」的概念。
    问「散射手都有谁、分别几星」，正确答案是 5 个人，
    但向量检索返回的是「散射手这个词出现得最像的 10 个块」——
    不保证覆盖全部 5 个人，实测漏了 6 星的维什戴尔。

    这类问题是"筛选 + 列举"，本质是数据库查询，应该走结构化路径。

【这张表怎么来的】
    语料里每个干员的「## 基本信息」小节已经是结构化的键值对
    （职业、分支、星级、站位、标签…），直接解析成表即可，不需要额外抓取。

用法：
    python structured_query.py --build          # 重建表
    python structured_query.py --branch 散射手    # 按分支筛
    python structured_query.py --star 6 --job 狙击
"""
import os
import re
import json
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "corpus_db")
LORE_DIR = os.path.join(BASE, "corpus_lore")
TABLE = os.path.join(BASE, "operators.json")

# 从「## 基本信息」里抽这些字段（在 corpus_db 里）
FIELDS = ["名称", "星级", "职业", "分支", "站位", "标签", "特性"]

# 档案里用【】标记的字段，要去 corpus_lore 里抽。
# 为什么分两处抽：职业/分支/星级这些在 corpus_db 的基本信息段，
# 而种族/性别/出身地在 corpus_lore 的「基础档案」段，格式是【种族】德拉克。
LORE_FIELDS = {
    "种族": "种族",
    "性别": "性别",
    "出身地": "出身地",
    "战斗经验": "战斗经验",
    "生日": "生日",
    "身高": "身高",
}

# 势力相关字段。值已经在语料里排好格式了：
#   所属势力：维多利亚 › 塔拉      （› 表示层级，页面就这么显示）
#   隐藏势力：炎、维多利亚          （、 表示并列，战斗中额外也算这些势力）
FACTION_FIELD = "所属势力"
HIDDEN_FIELD = "隐藏势力"


def build_table():
    """扫描 corpus_db 和 corpus_lore，抽出所有干员的属性，存成 JSON 表"""
    rows = []
    for fn in sorted(os.listdir(DB_DIR)):
        if not fn.endswith(".md"):
            continue
        name = fn[:-3]
        text = open(os.path.join(DB_DIR, fn), encoding="utf-8").read()
        if "## 基本信息" not in text:
            continue
        seg = text.split("## 基本信息")[1].split("\n##")[0]
        row = {"_file": name}
        for line in seg.split("\n"):
            m = re.match(r"^-\s*(.+?)：(.+)$", line.strip())
            if not m:
                continue
            k, v = m.group(1).strip(), m.group(2).strip()
            if k in FIELDS:
                row[k] = v
            elif k == FACTION_FIELD:
                # 「维多利亚 › 塔拉」拆成层级数组，方便按任一层级匹配
                parts = [x.strip() for x in v.split("›") if x.strip()]
                row["势力"] = v
                row["势力层级"] = parts
            elif k == HIDDEN_FIELD:
                # 「炎、维多利亚」拆成并列数组
                parts = [x.strip() for x in v.split("、") if x.strip()]
                row["隐藏势力"] = v
                row["隐藏势力列表"] = parts

        # 战斗势力判定用的完整集合：所属势力 + 隐藏势力（去重）
        # 例：予愿安洁莉娜 -> {罗德岛, 叙拉古}，两个加成都能吃
        combat = list(row.get("势力层级", []))
        for h in row.get("隐藏势力列表", []):
            if h not in combat:
                combat.append(h)
        row["全部势力"] = combat

        # 从档案里补种族/性别/出身地等
        lore_path = os.path.join(LORE_DIR, f"{name}.md")
        if os.path.exists(lore_path):
            lore = open(lore_path, encoding="utf-8").read()
            for tag, field in LORE_FIELDS.items():
                mm = re.search(r"【" + tag + r"】\s*([^\s【】]+)", lore)
                if mm:
                    row[field] = mm.group(1).strip()

        # 星级转成整数，方便比较排序
        if str(row.get("星级", "")).isdigit():
            row["星级"] = int(row["星级"])
        if row.get("名称"):
            rows.append(row)

    with open(TABLE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return rows


def load_table():
    if not os.path.exists(TABLE):
        return build_table()
    return json.load(open(TABLE, encoding="utf-8"))


# 属性里可比较的数值字段（用于「谁攻击力最高」这类问题）
ATTR_FIELDS = {
    "生命": "生命上限", "生命上限": "生命上限",
    "攻击": "攻击", "攻击力": "攻击", "攻击力上限": "攻击",
    "防御": "防御", "防御力": "防御",
    "法抗": "法抗", "法术抗性": "法抗",
    "攻击速度": "攻击速度",
    "部署费用": "部署费用", "费用": "部署费用",
    "阻挡": "阻挡", "阻挡数": "阻挡",
}

# 精英化阶段标记 -> 用于挑「满级」那一档做比较
def parse_attrs(file_name):
    """从 corpus_db/<干员>.md 的「## 属性」段解析出各阶段数值。

    属性段长这样：
        - 再部署 70s   部署费用 28 阻挡数 1   攻击速度 2.3s
        - 精英0 1级（精0 1）：生命 865 / 攻击 167 / 防御 94 / 法抗 0
        - 精英0 满级（50）：生命 1219 / 攻击 274 / ...
        - 精英1 满级（80）：生命 1469 / 攻击 392 / ...
        - 精英2 满级（90）：生命 1633 / 攻击 468 / ...
    我们取「精英2 满级」那一行（没有就取最高阶段），用于横向比较。
    """
    path = os.path.join(DB_DIR, f"{file_name}.md")
    if not os.path.exists(path):
        return {}
    text = open(path, encoding="utf-8").read()
    if "## 属性" not in text:
        return {}
    seg = text.split("## 属性")[1].split("\n##")[0]

    best_rank, best = -1, {}
    for line in seg.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.search(r"精英(\d)", line)
        rank = int(m.group(1)) if m else -1
        vals = {}
        # 匹配「生命 1633」「攻击 468」这种
        for key, val in re.findall(r"(生命上限|攻击|防御|法抗|生命)\s*([\d.]+)", line):
            vals[ATTR_FIELDS.get(key, key)] = float(val)
        if vals and rank >= best_rank:
            best_rank, best = rank, vals
    return best


def enrich_with_attrs(rows, only=None):
    """给表补充数值属性（只处理需要的行，避免每次都读 463 个文件）"""
    for r in rows:
        if only and r.get("名称") not in only:
            continue
        r["_attrs"] = parse_attrs(r.get("_file", r.get("名称", "")))
    return rows


def query(rows, branch=None, job=None, star=None, position=None, tag=None,
          race=None, gender=None, origin=None, faction=None, hidden=None):
    """按条件筛选，返回匹配的行。

    势力匹配用「任一层级」规则：
      问「维多利亚」-> 死芒（维多利亚 › 塔拉）也算，因为维多利亚出现在它的层级里
      问「塔拉」    -> 同样命中死芒，因为塔拉也出现在层级里
      问「炎」      -> 炎-龙门 的干员也算（按用户要求，"算一类"）
    隐藏势力单独一列，问「谁能吃叙拉古加成」时用它。
    """
    out = []
    for r in rows:
        if branch and branch not in str(r.get("分支", "")):
            continue
        if job and job not in str(r.get("职业", "")):
            continue
        if star is not None and r.get("星级") != star:
            continue
        if position and position not in str(r.get("站位", "")):
            continue
        if tag and tag not in str(r.get("标签", "")):
            continue
        if race and race != str(r.get("种族", "")):
            continue
        if gender and gender != str(r.get("性别", "")):
            continue
        if origin and origin != str(r.get("出身地", "")):
            continue
        if faction and not match_faction(r, faction, include_hidden=False):
            continue
        if hidden and hidden not in r.get("隐藏势力列表", []):
            continue
        out.append(r)
    # 按星级从高到低排
    out.sort(key=lambda x: -(x.get("星级") or 0))
    return out


def match_faction(row, value, include_hidden=True):
    """判断干员是否属于某个势力。

    匹配规则：该势力出现在「所属势力」的任一层级里，或（可选）在「隐藏势力」里。
    「炎」这种还要前缀匹配到「炎-龙门」——用户说过这两个算一类。
    """
    pool = list(row.get("势力层级", []))
    if include_hidden:
        pool = list(row.get("全部势力", []))
    for p in pool:
        if p == value:
            return True
        # 前缀匹配：问「炎」时，「炎-龙门」也算
        if p.startswith(value + "-") or value.startswith(p + "-"):
            return True
    return False


def format_rows(rows):
    """格式化成给模型看的表"""
    if not rows:
        return "（没有符合条件的干员）"
    lines = [f"{'干员':<16}{'星级':<6}{'职业':<8}{'分支':<12}{'所属势力':<22}隐藏势力"]
    lines.append("-" * 74)
    for r in rows:
        lines.append(f"{r.get('名称','?'):<16}{str(r.get('星级','?')):<6}"
                     f"{r.get('职业','?'):<8}{r.get('分支','?'):<12}"
                     f"{r.get('势力','(空)'):<22}{r.get('隐藏势力','-')}")
    lines.append(f"\n共 {len(rows)} 人")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--branch")
    ap.add_argument("--job")
    ap.add_argument("--star", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--race")
    ap.add_argument("--faction", help="按所属势力筛（任一层级命中）")
    ap.add_argument("--hidden", help="按隐藏势力筛")
    args = ap.parse_args()

    if args.build:
        rows = build_table()
        print(f"已重建表，共 {len(rows)} 个干员 -> {TABLE}")
        cov = {}
        for r in rows:
            for k in r:
                cov[k] = cov.get(k, 0) + 1
        for k, c in sorted(cov.items(), key=lambda x: -x[1]):
            print(f"  {k:<12} {c}/{len(rows)}")
    else:
        rows = load_table()
        print(f"表里 {len(rows)} 个干员")
        res = query(rows, args.branch, args.job, args.star, tag=args.tag,
                    race=args.race, faction=args.faction, hidden=args.hidden)
        print(format_rows(res))
