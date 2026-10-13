# -*- coding: utf-8 -*-
"""
第 2 步：切块（Chunking）

设计原则：准确性优先，尽量压低模型胡说的空间。具体落到四条：

  1. 在语义边界切，不用字数硬切
     按字数切会把一句话劈成两半，模型看到半句话就可能顺着编下去。
     所以：技能按「一个技能」切，档案按「一个小节」切，语音按「一条」切。

  2. 短块不合并
     两个不相干的小节合在一起，检索到的就是「混合体」，模型容易张冠李戴。
     宁可块短一点。

  3. 每块加来源头
     形如「凯尔希 · 技能 · 指令：结构加固」。
     一是让检索更容易命中（用户提问里往往带干员名），
     二是生成回答时能标注出处，用户可核对 —— 这是对抗幻觉最有效的手段。

  4. 两个库分开产出
     db   ：结构化数据，适合精确筛选（数值、职业、标签）
     lore ：剧情与语音，适合语义检索（性格、经历、说话方式）

用法：
    python 2_chunk.py
输出去 chunks_db.json / chunks_lore.json，并打印统计报告。
"""
import os
import re
import json
import glob

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "corpus_db")
LORE_DIR = os.path.join(BASE, "corpus_lore")


def read_dir(d):
    docs = []
    for p in sorted(glob.glob(os.path.join(d, "*.md"))):
        text = open(p, encoding="utf-8").read().strip()
        if text:
            docs.append((os.path.basename(p)[:-3], text))
    return docs


def make_chunk(lib, source, section, item, body):
    """组装一个块：统一加来源头"""
    head = f"{source} · {section}" + (f" · {item}" if item else "")
    return {
        "lib": lib,
        "source": source,
        "section": section,
        "item": item,
        "text": head + "\n" + body.strip(),
        "length": len(head) + 1 + len(body.strip()),
    }


# ---------------- db 库：按条目切 ----------------

def split_db(source, text):
    """结构化文件切法：## 章节 -> 条目。

    「技能」和「天赋」章节里，一个条目会占连续多行：
        技能名：指令：结构加固（自动回复 / 手动触发）
            7级：...
            专精三：...
    所以按「顶格行」切，而不是按每行切，这样一个技能就是一个完整的块。
    """
    chunks = []
    parts = re.split(r"\n(?=## )", text)
    for part in parts:
        lines = part.strip().split("\n")
        if not lines:
            continue
        head = lines[0].strip()
        if not head.startswith("## "):
            continue
        section = head[3:].strip()
        body_lines = lines[1:]

        if section in ("技能（7级 / 专精三）", "天赋"):
            items, cur = [], []
            for ln in body_lines:
                if ln.startswith("- ") and not ln.startswith("-     ") and cur:
                    items.append(cur); cur = [ln]
                elif ln.strip():
                    cur.append(ln)
            if cur:
                items.append(cur)
            for it in items:
                body = "\n".join(it).strip()
                if not body or body in ("- （无技能）", "- （无）"):
                    continue
                title = it[0].lstrip("- ").split("（")[0].strip()
                chunks.append(make_chunk("db", source, section.split("（")[0], title, body))
        else:
            body = "\n".join(x for x in body_lines if x.strip()).strip()
            if not body or body in ("- （无属性数据）", "- （无）", "- （无技能）"):
                continue
            chunks.append(make_chunk("db", source, section, "", body))
    return chunks


# ---------------- lore 库：按 ### 小节 + 语音按条 ----------------

def split_lore(source, text):
    """剧情文件切法：二级标题（##）和三级标题（###）都要切。

    踩过的坑：早期版本只按 `### ` 切，结果 `## 语音台词`（二级标题）不被识别，
    整段语音被并进了上一个三级小节（「晋升记录」）里——
    一个 628 字的档案块被撑成 6283 字，而且「语音按条切」的分支永远不执行。
    全库 463 个干员的语音都没能单独成块。
    """
    chunks = []
    # 二级和三级标题都要切：## 档案 / ## 语音台词 / ### 档案资料一 …
    parts = re.split(r"\n(?=#{2,3} )", text)
    for part in parts:
        part = part.strip()
        if not part.startswith("##"):
            continue
        lines = part.split("\n")
        heading = lines[0].lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue

        if heading == "语音台词":
            # 语音按条切：每条以 "- 【...】" 开头
            for it in re.split(r"\n(?=- 【)", body):
                it = it.strip()
                if not it:
                    continue
                m = re.match(r"- 【(.+?)】", it)
                label = m.group(1) if m else ""
                chunks.append(make_chunk("lore", source, "语音台词", label, it))
        else:
            # 二级标题（## 档案）本身没有正文，它的内容在下面的 ### 里，
            # 所以这里只处理 ### 小节；## 档案 会被 body 为空跳过。
            if not part.startswith("### "):
                continue
            chunks.append(make_chunk("lore", source, "档案", heading, body))
    return chunks


def report(name, chunks):
    if not chunks:
        print(f"{name}: 0 块")
        return
    lens = sorted(c["length"] for c in chunks)
    n = len(lens)
    print(f"{name}: {n} 块")
    print(f"    长度  最短 {lens[0]}  中位 {lens[n//2]}  最长 {lens[-1]}  平均 {sum(lens)//n}")
    from collections import Counter
    top = Counter(c["source"] for c in chunks).most_common(1)[0]
    print(f"    块数最多的干员：{top[0]}（{top[1]} 块）")
    print(f"    样例：{chunks[0]['text'][:80]!r}")


def main():
    db_chunks = []
    for source, text in read_dir(DB_DIR):
        db_chunks += split_db(source, text)
    with open(os.path.join(BASE, "chunks_db.json"), "w", encoding="utf-8") as f:
        json.dump(db_chunks, f, ensure_ascii=False, indent=1)

    lore_chunks = []
    for source, text in read_dir(LORE_DIR):
        lore_chunks += split_lore(source, text)
    with open(os.path.join(BASE, "chunks_lore.json"), "w", encoding="utf-8") as f:
        json.dump(lore_chunks, f, ensure_ascii=False, indent=1)

    print("=" * 56)
    report("corpus_db ", db_chunks)
    report("corpus_lore", lore_chunks)
    print("=" * 56)

    # ---- 准确性自检 ----
    print("\n自检：")
    allc = db_chunks + lore_chunks
    long_ones = [c for c in allc if c["length"] > 1200]
    print(f"  超过 1200 字的块：{len(long_ones)} 个"
          + (f"  例：{long_ones[0]['source']}·{long_ones[0]['item']}" if long_ones else ""))
    print(f"  短于 60 字的块：{len([c for c in allc if c['length'] < 60])} 个")
    print(f"  空块：{len([c for c in allc if not c['text'].strip()])} 个")
    print(f"  含模板残渣的块：{len([c for c in allc if '{{' in c['text']])} 个")
    print(f"  缺来源头的块：{len([c for c in allc if '·' not in c['text'].split(chr(10))[0]])} 个")


if __name__ == "__main__":
    main()
