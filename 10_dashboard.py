# -*- coding: utf-8 -*-
"""可观测性 Level 2：把 trace 聚合成看板。

Level 1（9_observe.py）解决的是「这一次为什么差」——单条链路能回放。
Level 2 解决的是「整体在往哪走」——趋势、分布、异常比例。

两者用的是同一份 logs/traces.jsonl，不额外埋点：
    Level 1 = 明细（一次请求一行）
    Level 2 = 聚合（把明细按维度汇总）

为什么指标要放在这一层而不是 Level 1：
    Level 1 的字段设计错了可以改，但**指标口径**要等有数据了才能定。
    比如「top1 低于多少算差」，我一开始凭经验写 0.6，
    实测这个语料的 top1 平均 0.93，那个阈值永远不会触发。

启动：
    streamlit run 10_dashboard.py
    然后打开终端里给出的 http://localhost:8501
"""
import os
import sys
import json
import importlib.util

import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
TRACE_FILE = os.path.join(BASE, "logs", "traces.jsonl")

# 复用 Level 1 的读取逻辑和阈值定义，避免两套口径
# （9_observe.py 以数字开头，不能直接 import，用路径加载）
_spec = importlib.util.spec_from_file_location("obs9", os.path.join(BASE, "9_observe.py"))
obs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(obs)

BRANCH_COLORS = {"vector": "🔍", "section": "📑", "structured": "🗂️", "unknown": "❔"}


# ---------------- 数据准备（纯函数，可单测）----------------

def load_traces():
    """读 trace 明细。文件不存在时返回空列表。"""
    return obs._read_traces(TRACE_FILE)


def flatten(rows):
    """把嵌套的 trace 记录摊平成一张表。

    嵌套字段（retrieval / llm / route / cut / refusal）在这里被提成独立列，
    这样 pandas 才能筛和聚合。原始 dict 保留在 raw 列里，供详情页展开。
    """
    out = []
    for r in rows:
        route = r.get("route") or {}
        ret = r.get("retrieval") or {}
        llm = r.get("llm") or {}
        cut = r.get("cut") or {}
        ref = r.get("refusal") or {}
        flags = r.get("flags") or []
        out.append({
            "trace_id": r.get("trace_id"),
            "时间": r.get("ts"),
            "问题": (r.get("question") or "")[:60],
            "路由": route.get("branch") or "unknown",
            "意图": route.get("intent"),
            "耗时ms": r.get("elapsed_ms"),
            "top1": ret.get("top1"),
            "空检索": bool(ret.get("empty")),
            "返回条数": ret.get("returned"),
            "裁剪丢弃": cut.get("dropped"),
            "模型调用": llm.get("calls", 0),
            "prompt_token": llm.get("prompt_tokens", 0),
            "拒答": bool(ref),
            "可疑数": len(flags),
            "可疑": "、".join(flags),
            "异常": r.get("error"),
            "raw": r,
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df["时间"] = pd.to_datetime(df["时间"], errors="coerce")
        df = df.sort_values("时间").reset_index(drop=True)
    return df


def _pct(vals, p):
    if not len(vals):
        return 0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def kpis(df):
    """算顶部那排汇总指标。全部基于筛选后的 df。"""
    n = len(df)
    if n == 0:
        return {}
    lat = df["耗时ms"].dropna().tolist()
    top1 = df["top1"].dropna().tolist()
    return {
        "请求数": n,
        "P50 延迟": _pct(lat, 50),
        "P95 延迟": _pct(lat, 95),
        "空检索率": df["空检索"].mean() * 100,
        "拒答率": df["拒答"].mean() * 100,
        "可疑率": (df["可疑数"] > 0).mean() * 100,
        "平均模型调用": df["模型调用"].mean(),
        "平均 prompt": df["prompt_token"].mean(),
        "平均 top1": (sum(top1) / len(top1)) if top1 else None,
        "低分次数": sum(1 for t in top1 if t < obs.LOW_TOP1),
    }


def latency_buckets(df, minutes=2):
    """按时间分桶算 P50/P95，用来画趋势线。"""
    if df.empty:
        return pd.DataFrame()
    d = df.set_index("时间").sort_index()
    g = d["耗时ms"].resample(f"{minutes}min")
    out = pd.DataFrame({
        "P50": g.apply(lambda s: _pct(s.dropna().tolist(), 50)),
        "P95": g.apply(lambda s: _pct(s.dropna().tolist(), 95)),
        "请求数": g.count(),
    }).reset_index()
    return out[out["请求数"] > 0]


def top1_histogram(df):
    """top1 分数的分布，用固定分桶画柱状图。"""
    vals = df["top1"].dropna()
    if vals.empty:
        return pd.DataFrame()
    bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    labels = ["<0.3", "0.3-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8",
              "0.8-0.9", "0.9-0.95", "0.95-1.0"]
    cut = pd.cut(vals, bins=bins, labels=labels, right=False)
    return cut.value_counts().reindex(labels).fillna(0).astype(int).rename_axis("top1 区间").reset_index(name="次数")


# ---------------- 页面 ----------------

def main():
    st.set_page_config(page_title="RAG 可观测性看板", layout="wide")
    st.title("RAG 可观测性看板")
    st.caption("数据来源：logs/traces.jsonl（Level 1 记录的每条请求明细，这里只做聚合）")

    rows = load_traces()
    if not rows:
        st.warning("还没有 trace 记录。先跑几次问答再看：\n\n"
                   "```\npython 6_generate.py \"凯尔希的部署费用是多少\"\n```\n\n"
                   "或者用 7_web.py 的网页界面问几轮。")
        return

    df_all = flatten(rows)

    # ---- 侧边栏筛选 ----
    with st.sidebar:
        st.header("筛选")
        branches = sorted(df_all["路由"].unique().tolist())
        pick = st.multiselect("路由分支", branches, default=branches)
        only_bad = st.checkbox("只看可疑记录", value=False)
        limit = st.slider("最多显示最近 N 条", 10, 2000, 500, step=10)
        if st.button("刷新", width="stretch"):
            st.rerun()
        st.divider()
        st.caption(f"`LOW_TOP1` 阈值：{obs.LOW_TOP1}")
        st.caption(f"`MANY_LLM_CALLS`：{obs.MANY_LLM_CALLS} 次")
        st.caption(f"trace 文件：{os.path.basename(TRACE_FILE)}")

    df = df_all[df_all["路由"].isin(pick)].tail(limit)
    if only_bad:
        df = df[df["可疑数"] > 0]

    if df.empty:
        st.info("当前筛选条件下没有记录。")
        return

    k = kpis(df)

    # ---- KPI ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("请求数", k["请求数"])
    c2.metric("P50 延迟", f"{k['P50 延迟']:.0f} ms")
    c3.metric("P95 延迟", f"{k['P95 延迟']:.0f} ms")
    c4.metric("可疑率", f"{k['可疑率']:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("空检索率", f"{k['空检索率']:.1f}%")
    c6.metric("拒答率", f"{k['拒答率']:.1f}%")
    c7.metric("平均模型调用", f"{k['平均模型调用']:.2f} 次")
    c8.metric("平均 prompt", f"{k['平均 prompt']:.0f} tok")

    if k["平均 top1"] is not None:
        st.caption(f"平均 top1 相似度 {k['平均 top1']:.3f}；"
                   f"低于阈值（{obs.LOW_TOP1}）的有 {k['低分次数']} 次")

    st.divider()

    # ---- 图表 ----
    left, right = st.columns(2)

    with left:
        st.subheader("路由分布")
        vc = df["路由"].value_counts().rename_axis("路由").reset_index(name="次数")
        st.bar_chart(vc, x="路由", y="次数", height=240)
        st.caption("structured = 走结构化查询；section = 章节整段取出；vector = 向量检索")

    with right:
        st.subheader("top1 相似度分布")
        hist = top1_histogram(df)
        if hist.empty:
            st.info("没有可用的 top1 数据")
        else:
            st.bar_chart(hist, x="top1 区间", y="次数", height=240)
            st.caption(f"低于 {obs.LOW_TOP1} 的区间值得重点关注")

    st.subheader("延迟趋势（按 2 分钟分桶）")
    lb = latency_buckets(df)
    if lb.empty:
        st.info("记录太少，还画不出趋势")
    else:
        st.line_chart(lb, x="时间", y=["P50", "P95"], height=260)
        st.caption("请求数少的桶波动大，别被 P95 的尖峰带偏——看请求数那一列")

    # ---- 可疑信号汇总 ----
    st.divider()
    st.subheader("可疑信号汇总")
    flag_cnt = {}
    for s in df["可疑"]:
        for f in [x for x in s.split("、") if x]:
            key = f.split("(")[0]
            flag_cnt[key] = flag_cnt.get(key, 0) + 1
    if not flag_cnt:
        st.success("当前筛选范围内没有被标记为可疑的请求。")
    else:
        fdf = pd.DataFrame(sorted(flag_cnt.items(), key=lambda x: -x[1]),
                           columns=["信号", "次数"])
        st.bar_chart(fdf, x="信号", y="次数", height=240)

    # ---- 明细与单条回放 ----
    st.divider()
    st.subheader("请求明细")
    show_cols = ["时间", "问题", "路由", "耗时ms", "top1", "模型调用",
                 "prompt_token", "空检索", "拒答", "可疑"]
    st.dataframe(df[show_cols], width="stretch", height=260)

    st.subheader("单条回放（Level 1 的详情）")
    pick_id = st.selectbox("选一条 trace 看完整链路",
                           df["trace_id"].tolist()[::-1],
                           format_func=lambda x: x)
    rec = next((r for r in df["raw"] if r.get("trace_id") == pick_id), None)
    if rec:
        st.markdown(obs.trace_markdown(rec))


if __name__ == "__main__":
    main()
