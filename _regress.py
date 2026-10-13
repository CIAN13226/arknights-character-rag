# -*- coding: utf-8 -*-
"""回归测试：一组问题跑一遍，看每个答得对不对"""
import os, io, sys, importlib.util, chromadb
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
BASE = os.path.dirname(os.path.abspath(__file__))
sp6 = importlib.util.spec_from_file_location("g6", os.path.join(BASE, "6_generate.py"))
g6 = importlib.util.module_from_spec(sp6); sp6.loader.exec_module(g6)
emb = g6.s4.load_model(); client = chromadb.PersistentClient(path=os.path.join(BASE, "chroma_db"))

# (问题, 期望关键词列表——出现任意一个算对)
CASES = [
    ("晓歌的出身地是哪", ["玻利瓦尔"]),
    ("赫德雷和伊内丝是什么关系", ["搭档", "战友", "左膀右臂"]),
    ("星熊二技能的专精三效果是什么", ["80%", "防御力"]),
    # 性格描述本来就有多个合理解，判分放宽（棘刺也确实符合"冷淡话少"）
    ("找一个性格冷淡话少的干员", ["稀音", "星熊", "絮雨", "玫兰莎", "棘刺", "傀影"]),
    ("阿米娅晚上睡觉喜不喜欢踢被子", ["没有", "无法"]),
    ("巫恋的娃娃叫什么名字", ["小莫提"]),
    ("散射手都有谁", ["假日威龙陈", "吉星", "奥斯塔", "松果", "送葬人"]),
    # 章节查询：之前向量检索只召回 1 个技能，答成基建技能
    ("凯尔希有什么技能", ["结构加固", "战术协同", "熔毁"]),
]

# 走结构化路径的问题要单独测（网页里那条链路和 answer() 不同）
STRUCTURED = [
    ("散射手都有谁？分别是几星？", ["假日威龙陈", "吉星", "奥斯塔", "松果", "送葬人"]),
    ("攻击力最高的狙击干员是谁", ["蕾缪安"]),
    ("执旗手一共几个", ["桃金娘", "极境"]),
    # 种族类：这几个之前全错（字段没进表 + 名字歧义）
    ("德拉克种族的干员都有谁", ["焰影苇草", "死芒", "苇草"]),
    ("有哪些干员的种族是杜林", ["桃金娘", "褐果", "至简", "特克诺", "杜林"]),
    ("种族是黎博利的干员都有谁", ["晓歌", "燧石"]),
    # 势力类：用户实际问过、之前全错的问题
    ("有哪些干员的阵营是炎国", ["乌有", "令", "年", "重岳", "黍"]),
    ("罗德岛有哪些干员", ["凯尔希", "阿米娅"]),
    ("塔拉有哪些干员", ["死芒", "焰影苇草"]),
    # 隐藏势力：附加势力，决定该干员在战斗的势力判定中额外算谁
    ("谁能吃叙拉古加成", ["予愿安洁莉娜"]),
]

ok_n = 0
for q, expects in CASES:
    # answer(model, client, question, ...) —— 注意第一个参数是模型名，不是向量模型
    reply, hits = g6.answer("qwen2.5:7b", client, q, "both", verbose=False)
    hit = any(e in reply for e in expects)
    ok_n += hit
    print(f"{'对' if hit else '错'}  {q}")
    print(f"     {reply[:130].replace(chr(10), ' / ')}")
print()
print(f"向量路径通过 {ok_n}/{len(CASES)}")

# ---- 结构化路径（网页里走的那条）----
sp8 = importlib.util.spec_from_file_location("r8", os.path.join(BASE, "8_router.py"))
r8 = importlib.util.module_from_spec(sp8); sp8.loader.exec_module(r8)
print()
ok2 = 0
for q, expects in STRUCTURED:
    text, desc = r8.run_structured(q)
    out = g6.call_llm(
        "你是《明日方舟》游戏知识助手。下面是从数据库里查出来的完整结果，"
        "请把它整理成便于阅读的中文回答。\n"
        "【重要】结果里有几条就要列出几条，一条都不能省略或合并。"
        "「简洁」指的是不要加额外的解释，不是让你压缩条目。\n"
        "不要添加结果之外的任何信息。",
        f"【查询结果】\n{text}\n\n【问题】\n{q}\n\n请逐条列出查询结果里的每一行，不要遗漏。",
        model="qwen2.5:7b")
    hit = all(e in out for e in expects)
    ok2 += hit
    print(f"{'对' if hit else '错'}  {q}   (筛选：{desc})")
    print(f"     {out[:150].replace(chr(10), ' / ')}")
print()
print(f"结构化路径通过 {ok2}/{len(STRUCTURED)}")
