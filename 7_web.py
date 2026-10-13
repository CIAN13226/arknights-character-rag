# -*- coding: utf-8 -*-
"""
第 7 步：Web 界面（Streamlit）

【为什么不用 Ollama 自带的界面】
    Ollama 官方 GUI 是个「直接和大模型聊天」的窗口，它只调用模型本身，
    不会走我们的检索。用它问「凯尔希有什么技能」，回答完全来自模型记忆，
    很容易编——实测直接问 3B 模型，它把凯尔希说成「辅助」干员（实际是医疗）。
    我们这两周做的语料、检索、重排、评测全部用不上。

【这个界面多做了什么】
    右边实时显示「检索到了哪些块、每块打了几分」。
    这不只是好看——RAG 出问题时，你要能一眼看出是「没检索到」还是
    「检索到了但模型说错了」，这两种原因的修法完全不同。

启动：
    streamlit run 7_web.py
然后浏览器打开 http://localhost:8501
"""
import os
import time
import importlib.util

import streamlit as st
import chromadb

BASE = os.path.dirname(os.path.abspath(__file__))

# 默认模型：7B 效果好，3B 快。可以在侧边栏切换。
DEFAULT_LLM = os.environ.get("RAG_LLM", "qwen2.5:7b")

st.set_page_config(page_title="明日方舟知识助手", page_icon="🎮", layout="wide")


# ---------------- 加载各模块（缓存，避免每次交互都重新加载模型） ----------------

@st.cache_resource(show_spinner="正在加载检索模块...")
def load_modules():
    """加载 4_search（检索）和 6_generate（生成）这两个模块"""
    spec4 = importlib.util.spec_from_file_location("s4", os.path.join(BASE, "4_search.py"))
    s4 = importlib.util.module_from_spec(spec4)
    spec4.loader.exec_module(s4)

    spec6 = importlib.util.spec_from_file_location("g6", os.path.join(BASE, "6_generate.py"))
    g6 = importlib.util.module_from_spec(spec6)
    spec6.loader.exec_module(g6)
    return s4, g6


@st.cache_resource(show_spinner="正在加载向量模型...")
def load_embedder():
    s4, _ = load_modules()
    return s4.load_model()


@st.cache_resource
def load_client():
    s4, _ = load_modules()
    return chromadb.PersistentClient(path=os.path.join(BASE, "chroma_db"))


@st.cache_resource(show_spinner="正在预热模型（第一次会慢一点）...")
def warm_up(llm_name):
    """启动时把三个模型都加载好，并让大模型跑一次极短推理。

    为什么要预热：不预热的话，第一次提问会同时触发
    「向量模型加载 + 重排模型加载 + 大模型加载」三件事，
    用户要干等半分钟，而且 Ollama 在加载模型期间可能返回不正常的响应。
    预热后第一次提问就和后续一样快。
    """
    import json
    import urllib.request
    s4, g6 = load_modules()
    s4.load_model()          # 向量模型
    try:
        s4.get_reranker()    # 重排模型（内部自动判断设备）
    except Exception:
        pass
    # 让大模型做一次极短推理，把它从磁盘加载进显存
    try:
        payload = {"model": llm_name,
                   "messages": [{"role": "user", "content": "回复：就绪"}],
                   "stream": False, "options": {"num_predict": 4}}
        req = urllib.request.Request(f"{g6.OLLAMA}/api/chat",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
    except Exception:
        pass
    return True


def check_ollama():
    """检查 Ollama 服务是否在跑，返回可用模型名列表"""
    import json
    import urllib.request
    s4, g6 = load_modules()
    try:
        with urllib.request.urlopen(f"{g6.OLLAMA}/api/tags", timeout=5) as r:
            tags = json.loads(r.read().decode("utf-8"))
        return [m["name"] for m in tags.get("models", [])]
    except Exception as e:
        return None


# ---------------- 侧边栏：参数 ----------------

st.sidebar.title("⚙️ 设置")

models = check_ollama()
if models is None:
    st.sidebar.error("❌ 连不上 Ollama\n\n请先运行：`ollama serve`")
    st.stop()
if not models:
    st.sidebar.error("❌ 没有已安装的模型\n\n请先拉取：`ollama pull qwen2.5:7b`")
    st.stop()

llm = st.sidebar.selectbox("生成模型", models,
                           index=models.index(DEFAULT_LLM) if DEFAULT_LLM in models else 0)

k = st.sidebar.slider("检索返回条数 K", 3, 15, 10,
                      help="喂给模型的资料条数。太小会漏掉正确内容，太大会引入噪声。")

use_rerank = st.sidebar.checkbox("启用重排", value=True,
                                 help="两阶段检索：先向量粗筛 30 条，再用重排模型精排。"
                                      "实测能提升排序质量，但会慢一点。")

lib = st.sidebar.radio("检索范围", ["both", "db", "lore"],
                       format_func=lambda x: {"both": "两个库", "db": "结构化库（数值/技能）",
                                              "lore": "剧情库（档案/语音）"}[x],
                       horizontal=False)

show_debug = st.sidebar.checkbox("显示检索详情", value=True,
                                 help="把检索到的每个块和它的分数显示出来。"
                                      "回答不对时，靠这个判断是「没检索到」还是「模型说错了」")

st.sidebar.markdown("---")

# ---------- 对话持久化 ----------
# 把对话存到文件里，这样重启网页、或者重新打开浏览器都不会丢。
# 之前每次重启都清空，测几个问题就没了，很影响连续测试。
HISTORY_FILE = os.path.join(BASE, "chat_history.json")


def load_history():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # 检索到的资料不存进文件（体积大且每次都可能变），只留问答文本
        for m in data:
            m.pop("sources", None)
        return data
    except Exception:
        return []


def save_history(msgs):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([{"role": m["role"], "content": m["content"]} for m in msgs],
                      f, ensure_ascii=False, indent=1)
    except Exception:
        pass


if st.sidebar.button("🗑 清空对话"):
    st.session_state.messages = []
    save_history([])
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "**数据规模**\n\n"
    "- 463 个干员\n"
    "- 13,450 个语义块\n"
    "- 事实库（3,492）+ 剧情库（9,958）分开"
)

# 预热（放在侧边栏参数确定之后，因为要用到选择的模型名）
warm_up(llm)


# ---------------- 主界面 ----------------

st.title("🎮 明日方舟知识助手")
st.caption("基于本地语料的 RAG 问答 · 回答只依据检索到的资料，资料里没有的会明说")

# 首次打开时从文件恢复历史对话
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_debug:
            with st.expander(f"📚 检索到 {len(msg['sources'])} 条资料"):
                for i, (score, meta, doc) in enumerate(msg["sources"], 1):
                    head = f"{meta.get('source')} · {meta.get('section')}"
                    if meta.get("item"):
                        head += f" · {meta['item']}"
                    st.markdown(f"**[{i}] {score:.3f}** — {head}")
                    st.code(doc[:600] + ("..." if len(doc) > 600 else ""), language=None)

# 输入框
if prompt := st.chat_input("问点什么，比如：星熊二技能的专精三效果是什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        s4, g6 = load_modules()
        emb = load_embedder()
        client = load_client()

        t0 = time.time()

        # 调用共用的问答链路（和命令行跑的是同一个函数）。
        # 早期这里自己写了一遍流程，导致修 bug 时两条路径不一致——
        # 「巫恋的娃娃」在命令行修好了，网页上还是答错的。现在只有一份逻辑。
        with st.spinner(f"正在用 {llm} 回答..."):
            res = g6.generate(prompt, llm, emb, client,
                              lib=lib, k=k, use_rerank=use_rerank)

        reply = res["reply"]
        filtered = res["hits"]
        structured_text = res["structured_text"]
        route_note = " · ".join(res["notes"]) if res["notes"] else ""

        elapsed = time.time() - t0
        st.markdown(reply)

        info = f"耗时 {elapsed:.1f}s"
        if route_note:
            info = f"🧭 {route_note} · " + info
        if structured_text:
            with st.expander("📊 结构化查询结果（原始表格）"):
                st.code(structured_text, language=None)
        if filtered and show_debug:
            with st.expander(f"📚 检索到 {len(filtered)} 条资料（点击展开原文）", expanded=False):
                for i, (score, meta, doc) in enumerate(filtered, 1):
                    head = f"{meta.get('source')} · {meta.get('section')}"
                    if meta.get("item"):
                        head += f" · {meta['item']}"
                    tag = " 🎯" if i == 1 else ""
                    st.markdown(f"**[{i}] 得分 {score:.3f}** — {head}{tag}")
                    st.code(doc[:600] + ("..." if len(doc) > 600 else ""), language=None)

        st.caption(info)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": filtered})
    save_history(st.session_state.messages)
