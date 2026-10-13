# -*- coding: utf-8 -*-
"""
从 ModelScope（魔搭）下载 bge-large-zh-v1.5。

为什么不用 HuggingFace：国内直连 huggingface.co 和 hf-mirror.com 都会超时，
但 ModelScope 是国内站点，直连可用，而且 BAAI 的模型在上面有官方镜像。

下载后会打印本地路径，直接把这个路径填给 SentenceTransformer 即可。

用法：
    python download_model.py                    # 下 bge-large-zh-v1.5
    python download_model.py bge-base-zh-v1.5   # 换一个
"""
import os
import sys
import time

CANDIDATES = {
    # 向量模型：把文本变成向量，用来从库里召回候选
    "bge-large-zh-v1.5": ["BAAI/bge-large-zh-v1.5", "AI-ModelScope/bge-large-zh-v1.5"],
    "bge-base-zh-v1.5": ["BAAI/bge-base-zh-v1.5", "AI-ModelScope/bge-base-zh-v1.5"],
    "bge-small-zh-v1.5": ["BAAI/bge-small-zh-v1.5", "AI-ModelScope/bge-small-zh-v1.5"],
    # 重排模型：给「问题-文档」这一对打分，用来对召回结果精排。
    # 注意它不能单独做检索，必须配合上面的向量模型用。
    "bge-reranker-base": ["BAAI/bge-reranker-base"],
    "bge-reranker-large": ["BAAI/bge-reranker-large"],
    "bge-reranker-v2-m3": ["BAAI/bge-reranker-v2-m3"],
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "bge-large-zh-v1.5"
    ids = CANDIDATES.get(which)
    if not ids:
        print(f"不认识 {which}，可选：{list(CANDIDATES)}")
        return

    from modelscope.hub.snapshot_download import snapshot_download

    for mid in ids:
        print(f"尝试下载 {mid} ...")
        t0 = time.time()
        try:
            path = snapshot_download(mid, cache_dir=os.path.join(
                os.path.expanduser("~"), ".cache", "modelscope"))
            print(f"\n下载成功，用时 {time.time()-t0:.0f}s")
            print(f"本地路径：{path}")
            # 列出关键文件确认完整
            for f in ["config.json", "pytorch_model.bin", "model.safetensors",
                      "tokenizer.json", "modules.json"]:
                p = os.path.join(path, f)
                if os.path.exists(p):
                    print(f"  {f}  {os.path.getsize(p)/1024/1024:.1f} MB")
            print("\n把这个路径填给 3_embed.py / 4_search.py 即可")
            return
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {str(e)[:160]}")
    print("\n所有候选 ID 都失败了")


if __name__ == "__main__":
    main()
