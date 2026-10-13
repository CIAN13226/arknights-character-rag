# -*- coding: utf-8 -*-
"""
第 3 步：向量化 + 存库（两个库分开存）

【向量化在干什么】
    把一段文字映射成一串固定长度的数字（这里是 512 个数）。
    关键性质：语义相近的文字，向量距离就小。
        "凯尔希有什么技能"  <->  "技能名：指令：结构加固"        距离近
        "凯尔希有什么技能"  <->  "博士别乱动，武器不长眼"        距离远
    所以模型选得对不对，直接决定检索质量。中文必须用中文模型。

【为什么两个库分开存】
    db 库是「事实」——数值、职业、标签，适合精确筛选。
    lore 库是「叙述」——性格、经历、说话方式，只能靠语义匹配。
    两者的查询意图完全不同，混在一个 collection 里会互相干扰：
    你问「谁的生命上限最高」，lore 库那些剧情文本可能因为提到"生命"而抢占结果位。
    所以物理上分开，之后靠「查询路由」决定查哪个。

【为什么要向量库】
    7000 个块的时候，自己写个 for 循环算余弦相似度也能跑。
    但 100 万个块呢？每次提问都要跟 100 万个向量算一遍。
    向量库用近似最近邻（ANN）建索引，牺牲一点精度换数量级的速度。

用法：
    python 3_embed.py
"""
import os
import json
import time
import glob
import chromadb
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE, "chroma_db")

# 向量模型可以换。换之后必须重跑本脚本 + 5_eval.py 重新评测，
# 而且 collection 名要跟着变，否则会和旧模型的向量混在一起。
#   bge-small-zh-v1.5  约 95MB   512 维   （已经跑过：Recall@5 = 60%）
#   bge-base-zh-v1.5   约 400MB  768 维
#   bge-large-zh-v1.5  约 1.3GB  1024 维
MODEL_NAME = os.environ.get("RAG_MODEL", "BAAI/bge-large-zh-v1.5")
SUFFIX = MODEL_NAME.split("/")[-1].replace("bge-", "").replace("-zh-v1.5", "")

COLLECTIONS = [
    (f"arknights_db_{SUFFIX}", os.path.join(BASE, "chunks_db.json")),
    (f"arknights_lore_{SUFFIX}", os.path.join(BASE, "chunks_lore.json")),
]


def find_local_snapshot(name):
    """找模型的本地缓存目录。

    两个来源都找一遍：
      1. ModelScope（魔搭）—— 国内直连可用，推荐
      2. HuggingFace 缓存 —— 之前下载过的话还能用

    为什么要这么做：国内直连 huggingface.co 和 hf-mirror.com 都会超时，
    而 transformers 5.x 在联网失败时会走到一条有报错的加载路径。
    直接用本地路径 + local_files_only=True 可以完全跳过网络检查。
    """
    home = os.path.expanduser("~")
    key = name.replace("/", "--")
    # ModelScope 的实际目录结构：~/.cache/modelscope/models/BAAI--bge-large-zh-v1.5/snapshots/master/
    # 注意别用太宽的递归 glob —— 里面还有 1_Pooling/config.json 这种子目录配置，
    # 容易被误当成模型根目录。
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
    """优先用本地快照加载。

    为什么不让它先试联网：国内连 huggingface.co 会超时重试 5 次，
    一次要卡好几分钟，而且失败了才回退。本地明明有完整缓存（约 95MB），
    直接用它更快也更稳。
    """
    snap = find_local_snapshot(MODEL_NAME)
    if snap:
        print(f"      使用本地快照：{snap}")
        try:
            return SentenceTransformer(snap, local_files_only=True), "本地快照"
        except Exception as e:
            print(f"      本地快照加载失败（{type(e).__name__}），改试联网")
    try:
        print(f"      本地没有缓存，联网加载 {MODEL_NAME}")
        return SentenceTransformer(MODEL_NAME), "联网/缓存"
    except Exception as e:
        raise RuntimeError(
            "模型加载失败，本地无缓存且联网不可用。请先下载一次：\n"
            "  $env:HF_ENDPOINT='https://hf-mirror.com'\n"
            f"  python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{MODEL_NAME}')\""
        ) from e


def main():
    # ---------- 1. 加载模型 ----------
    print(f"[1/4] 加载向量模型 {MODEL_NAME}")
    t0 = time.time()
    model, how = load_model()
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
            print(f"      设备：GPU {torch.cuda.get_device_name(0)}")
        else:
            print("      设备：CPU")
    except Exception as e:
        print(f"      设备检测失败，用默认：{e}")
    print(f"      加载方式：{how}，耗时 {time.time()-t0:.1f}s")

    client = chromadb.PersistentClient(path=DB_DIR)
    total = 0

    for name, path in COLLECTIONS:
        if not os.path.exists(path):
            print(f"\n[跳过] 找不到 {os.path.basename(path)}，先跑 2_chunk.py")
            continue
        chunks = json.load(open(path, encoding="utf-8"))
        print(f"\n[2/4] 处理 {name} 库：{len(chunks)} 个块")

        texts = [c["text"] for c in chunks]
        t0 = time.time()
        embs = model.encode(texts, batch_size=64,
                            normalize_embeddings=True,
                            show_progress_bar=True)
        print(f"      向量化完成，耗时 {time.time()-t0:.1f}s，维度 {embs.shape[1]}")

        try:
            client.delete_collection(name)
            print(f"      已删除旧的 collection「{name}」")
        except Exception:
            pass
        col = client.create_collection(name, metadata={"hnsw:space": "cosine"})

        B = 1000          # Chroma 单次 add 有上限，分批写
        for i in range(0, len(chunks), B):
            part = chunks[i:i + B]
            col.add(
                ids=[f"{name}-{i+j}" for j in range(len(part))],
                documents=[c["text"] for c in part],
                embeddings=[e.tolist() for e in embs[i:i + B]],
                metadatas=[{"lib": c["lib"], "source": c["source"],
                            "section": c["section"], "item": c["item"] or ""}
                           for c in part],
            )
        print(f"      写入完成，collection 里有 {col.count()} 条")
        total += col.count()

    # ---------- 3. 语义区分度自检 ----------
    print(f"\n[3/4] 语义区分度自检（看模型在这个语料上能不能分辨）")
    pairs = [
        ("凯尔希有什么技能", "技能名：指令：结构加固"),
        ("凯尔希有什么技能", "博士别乱动，武器不长眼，小心伤了你"),
        ("谁的生命上限最高", "精英2 满级（90级）：生命 1633"),
    ]
    texts = [p[0] for p in pairs] + [p[1] for p in pairs]
    v = model.encode(texts, normalize_embeddings=True)
    for i, (q, a) in enumerate(pairs):
        sim = float(v[i] @ v[len(pairs) + i])
        print(f"      「{q}」 vs 「{a[:22]}」 = {sim:.3f}")

    # ---------- 4. 汇总 ----------
    print(f"\n[4/4] 完成，两个 collection 共 {total} 条")
    print("\n下一步：4_search.py 做检索")


if __name__ == "__main__":
    main()
