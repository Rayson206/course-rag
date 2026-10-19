"""
M1 第2步：向量化 + 入库（构建索引）
运行：uv run python -m course_rag.index
作用：把每个文本块调 BGE Embedding API 变成向量，存入本地 ChromaDB。
"""
import math
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from course_rag.chunking import DATA_PATH, chunk_text

load_dotenv()

EMBED_MODEL = "BAAI/bge-large-zh-v1.5"  # 中文向量模型，1024 维，免费
DB_PATH = Path(__file__).resolve().parents[2] / "chroma_db"

client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量调用 Embedding API，把一批文字变成向量。"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # API 返回顺序可能乱，按 index 排序后只取向量
    return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度：两个向量方向越一致，值越接近 1；方向无关则接近 0。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm = lambda v: math.sqrt(sum(x * x for x in v))
    return dot / (norm(a) * norm(b))


def build_index() -> None:
    # 1. 切块（复用 M1.1 的成果）
    text = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    print(f"① 切块完成：{len(chunks)} 块")

    # 2. 向量化：7 个块一次性送进 Embedding API
    vectors = embed_texts([c["text"] for c in chunks])
    print(f"② 向量化完成：{len(vectors)} 个向量，每个 {len(vectors[0])} 维")
    print(f"   第 0 块前 5 个数字：{[round(x, 4) for x in vectors[0][:5]]}")

    # 3. 语义相似度小实验：三个问题各自最像哪个块？
    print("③ 语义相似度小实验（越接近 1 越相似）：")
    probes = ["进程有哪几种状态", "阻塞状态是什么意思", "我今天中午吃什么"]
    probe_vectors = embed_texts(probes)
    for i, question in enumerate(probes):
        sims = [cosine_similarity(probe_vectors[i], vectors[j]) for j in range(len(chunks))]
        best = max(range(len(sims)), key=lambda j: sims[j])
        print(f"   「{question}」→ 最像第 {best} 块《{chunks[best]['heading']}》（{sims[best]:.3f}）")

    # 4. 写入 ChromaDB（持久化到磁盘，关机也不丢）
    db = chromadb.PersistentClient(path=str(DB_PATH))
    collection = db.get_or_create_collection(name="lecture")
    collection.upsert(  # upsert：重复运行时按 id 覆盖，不会报错
        ids=[f"chunk-{c['index']}" for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"heading": c["heading"]} for c in chunks],
    )
    print(f"④ 已写入 ChromaDB：{collection.count()} 条记录，持久化目录 {DB_PATH.name}/")


if __name__ == "__main__":
    build_index()
