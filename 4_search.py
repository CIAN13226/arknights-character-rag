# -*- coding: utf-8 -*-
"""
第 4 步：检索（Retrieval）

【检索在干什么】
    把用户的问题也变成向量 → 在库里找最像的 K 个块 → 按相似度排序返回。

【两个关键参数】
    K（返回几个块）
        太小：正确答案可能排在第 6 位，你只取 3 个就漏了
        太大：噪声进来，后面喂给模型时容易被带偏
        先用 5，之后靠评测集调。

    查询库（db / lore / 都查）
        db   ：问数值、职业、技能效果
        lore ：问性格、经历、说话方式
        这一步先用参数手动指定，下一步再做「自动路由」。

【为什么只返回原文、不生成回答】
    先验证检索这一环准不准。检索都不准，后面接大模型只会把错误信息编得更像真的。
    所以这个脚本只负责「找出最相关的几段原文」，并打印相似度分数，方便你核对。

用法：
    python 4_search.py "凯尔希有什么技能"
    python 4_search.py "性格冷淡话少的干员" --lib lore
    python 4_search.py "生命上限最高的狙击" --lib db -k 8
    python 4_search.py                 # 不带参数就跑内置的测试问题
"""
import os
import sys
import argparse
import chromadb
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "chroma_db")

# 必须和 3_embed.py 里用同一个模型，否则向量空间对不上，检索结果会是乱的
MODEL_NAME = os.environ.get("RAG_MODEL", "BAAI/bge-large-zh-v1.5")
SUFFIX = MODEL_NAME.split("/")[-1].replace("bge-", "").replace("-zh-v1.5", "")

# 重排模型：对「问题-文档」成对打分，用来精排。
# 它和向量模型是两回事——向量模型把问题和文档各自压成一个向量，只能算粗略相似；
# 重排模型把问题和文档拼在一起过一次完整网络，判断准得多，但慢，所以只用来精排。
RERANK_NAME = os.environ.get("RAG_RERANK", "BAAI/bge-reranker-base")
RERANK_ON = os.environ.get("RAG_NO_RERANK", "") != "1"
# 先粗筛多少条给重排模型。取大一点，避免正确答案在向量排序时被埋太深。
RECALL_K = int(os.environ.get("RAG_RECALL_K", "30"))
_reranker = None


def get_reranker():
    """按需加载重排模型（只加载一次）"""
    global _reranker
    if _reranker is not None:
        return _reranker
    from sentence_transformers import CrossEncoder
    path = find_local_snapshot(RERANK_NAME) or RERANK_NAME
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    _reranker = (CrossEncoder(path, max_length=512, device=device,
                              local_files_only=bool(find_local_snapshot(RERANK_NAME))),
                 device)
    return _reranker


def find_local_snapshot(name):
    """找模型的本地缓存目录。和 3_embed.py 保持同一套逻辑，改一处要改两处。

    两个来源都找：ModelScope（国内直连可用）和 HuggingFace 缓存。
    直接给本地路径 + local_files_only=True，可以跳过联网检查，
    否则 transformers 在连不上 Hub 时会抛 OSError。
    """
    import glob
    home = os.path.expanduser("~")
    key = name.replace("/", "--")
    pats = [
        os.path.join(home, ".cache", "modelscope", "models", key, "snapshots", "*"),
        os.path.join(home, ".cache", "modelscope", "hub", key, "snapshots", "*"),
        os.path.join(home, ".cache", "modelscope", "hub", "models", key, "snapshots", "*"),
        os.path.join(home, ".cache", "huggingface", "hub",
                     "models--" + key, "snapshots", "*"),
    ]
    for pat in pats:
        for d in sorted(glob.glob(pat)):
            has_cfg = os.path.exists(os.path.join(d, "config.json"))
            has_weight = any(os.path.exists(os.path.join(d, f)) for f in
                             ("model.safetensors", "pytorch_model.bin"))
            if has_cfg and has_weight:
                return d
    return None


def load_model():
    snap = find_local_snapshot(MODEL_NAME)
    if snap:
        m = SentenceTransformer(snap, local_files_only=True)
    else:
        m = SentenceTransformer(MODEL_NAME)
    try:
        import torch
        if torch.cuda.is_available():
            m = m.to("cuda")
    except Exception:
        pass
    return m


def search(model, client, query, lib="both", k=5, rerank=None, diag=None):
    """两阶段检索，返回 [(分数, 元数据, 文本)]。

    为什么用两阶段：
      第一阶段（向量召回）快，但排序粗——短块因为关键词密度高，
        相似度往往虚高，会把真正的长答案挤到十几名开外。
        实测问「凯尔希有什么技能」，两个技能块排在第 14 和 16 位。
      第二阶段（重排）把「问题+文档」拼起来过一次完整模型，
        判断准得多，能把真正相关的挑到前面。代价是慢，所以只对少量候选做。

    rerank 参数：None 表示跟随 RERANK_ON 的全局设置，
                True/False 可强制开关。
    diag 参数：传一个 dict 进来，函数会把本次检索的诊断信息填进去
              （召回候选数、候选总数、是否重排、每条最终结果的"重排前向量排名"）。
              **重排前的排名只在内存里存在，跑完就没了**，不在这里记就永久丢失，
              所以可观测性必须在检索这一层取数。
    """
    use_rerank = RERANK_ON if rerank is None else rerank
    # 开了重排就先多召回一些候选，否则精排没有选择的余地
    fetch_k = max(RECALL_K, k) if use_rerank else k

    qv = model.encode([query], normalize_embeddings=True)[0].tolist()
    if lib == "both":
        names = [f"arknights_db_{SUFFIX}", f"arknights_lore_{SUFFIX}"]
    else:
        names = [f"arknights_{lib}_{SUFFIX}"]

    hits = []
    for name in names:
        try:
            col = client.get_collection(name)
        except Exception:
            print(f"  [警告] 找不到 collection {name}")
            continue
        # 两个库分开查，各自取 fetch_k，合并后再统一排序
        res = col.query(query_embeddings=[qv], n_results=fetch_k,
                        include=["documents", "metadatas", "distances"])
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                   res["distances"][0]):
            # Chroma 用余弦距离，转成相似度更直观
            hits.append((1.0 - dist, meta, doc))

    hits.sort(key=lambda x: -x[0])

    # 重排前的向量排名快照（按 doc 前 80 字做标识，和下面合并去重用的键一致）
    vec_order = [doc[:80] for _, _, doc in hits]

    if use_rerank and hits:
        try:
            rk, device = get_reranker()
            # 用重排分数给所有候选排序，但前面 k 个位置里，靠前的部分留给重排结果，
            # 靠后的部分留给向量召回结果（保底）。
            #
            # 为什么要保底：重排模型只判「字面相关性」，而向量模型懂「语义相似」。
            # 实测直接采用重排结果，Recall@5 会从 66.7% 掉到 60.0%——
            # 就是因为它把向量模型靠语义召回对的答案挤出去了。
            # 两者各有盲区，所以都要保留。
            vec_top = hits[:max(k, 10)]
            pairs = [[query, doc] for _, _, doc in hits]
            scores = rk.predict(pairs, batch_size=32)
            reranked = sorted(
                ((float(s), meta, doc) for s, (_, meta, doc) in zip(scores, hits)),
                key=lambda x: -x[0])

            half = max(1, k // 2)
            merged, seen = [], set()
            # 前半：重排认为最相关的
            for item in reranked:
                if len(merged) >= half:
                    break
                key = item[2][:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            # 后半：向量召回里最相关的（补足保底）
            for item in vec_top:
                if len(merged) >= k:
                    break
                key = item[2][:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            # 还不够就用剩下的重排结果填满
            for item in reranked:
                if len(merged) >= k:
                    break
                key = item[2][:80]
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            hits = merged
        except Exception as e:
            print(f"  [警告] 重排失败，退回向量排序：{type(e).__name__}: {e}")

    final = hits[:k]

    # 把诊断信息交出去（只在调用方传了 diag 时才做，不影响其他调用方）
    if diag is not None:
        try:
            vec_ranks = []
            for _, _, doc in final:
                key = doc[:80]
                vec_ranks.append(vec_order.index(key) + 1 if key in vec_order else None)
            diag.update({
                "recall_k": fetch_k,
                "candidates": len(vec_order),
                "rerank_used": bool(use_rerank),
                "vec_ranks": vec_ranks,
            })
        except Exception:
            pass

    return final


def show(query, hits, k):
    print(f"\n{'='*70}")
    print(f"问题：{query}")
    print(f"{'='*70}")
    if not hits:
        print("  没有结果")
        return
    for i, (score, meta, doc) in enumerate(hits, 1):
        head = doc.split("\n")[0]
        body = "\n".join(doc.split("\n")[1:])[:200].replace("\n", " ")
        lib = meta.get("lib", "?")
        print(f"\n  [{i}] 相似度 {score:.3f}   来源库 {lib}")
        print(f"      {head}")
        print(f"      {body}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="要检索的问题")
    ap.add_argument("--lib", choices=["db", "lore", "both"], default="both")
    ap.add_argument("-k", type=int, default=5, help="返回几个块")
    args = ap.parse_args()

    model = load_model()
    client = chromadb.PersistentClient(path=DB_DIR)

    queries = [" ".join(args.query)] if args.query else [
        "凯尔希有什么技能",
        "攻击力最高的狙击干员是谁",
        "找一个性格冷淡、话很少的干员",
        "谁适合当新手的第一个六星",
    ]

    for q in queries:
        hits = search(model, client, q, args.lib, args.k)
        show(q, hits, args.k)

    print(f"\n{'='*70}")
    print("看结果时重点核对：")
    print("  1. Top1 的块，是不是真的和问题相关？")
    print("  2. 有没有明显不相关的块混进 Top5？（那就是召回了噪声）")
    print("  3. 正确答案如果排在第 4、5 位，说明 K 取小了，或者该调切块方式")
    print("这两个现象就是你后面要优化的地方，也是评测集要量化的问题。")


if __name__ == "__main__":
    main()
