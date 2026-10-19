"""
M1 第3-4步：RAG 问答闭环（检索 + 生成）
运行方式：
    无参数，跑三个内置演示问题：uv run python -m course_rag.qa
    问自己的问题：uv run python -m course_rag.qa "死锁的四个必要条件是什么"
"""
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from course_rag.index import embed_texts

load_dotenv()

TOP_K = 5  # 每次检索取最相关的 5 块
DB_PATH = Path(__file__).resolve().parents[2] / "chroma_db"

llm = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

# 打开本地向量库（索引由 index.py 预先构建）
db = chromadb.PersistentClient(path=str(DB_PATH))
collection = db.get_or_create_collection(
    name="lecture", metadata={"hnsw:space": "cosine"}
)


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """问题向量化 → 从 ChromaDB 取回最相似的 k 块。"""
    query_vector = embed_texts([question])[0]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(result["ids"][0])):
        hits.append(
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "heading": result["metadatas"][0][i]["heading"],
                "distance": result["distances"][0][i],
            }
        )
    return hits


def build_prompt(question: str, hits: list[dict]) -> str:
    """把参考资料和问题拼成 Prompt——这是约束模型'只照讲义回答'的关键。"""
    refs = []
    for i, h in enumerate(hits, 1):
        refs.append(f"[资料{i}]（章节：{h['heading']}）\n{h['text']}")
    refs_text = "\n\n".join(refs)

    return f"""你是一个严谨的课程答疑助教。请严格根据下面的【讲义参考资料】回答问题。

规则：
1. 只能使用资料中出现的信息，不得使用资料外的知识，不得编造；
2. 如果资料不足以回答问题，直接回答"讲义中没有相关内容"，不要硬答；
3. 回答分条、简洁。

【讲义参考资料】
{refs_text}

【问题】
{question}
"""


def answer(question: str) -> None:
    # 第一步：检索
    hits = retrieve(question)
    # 第二步：拼 Prompt 让模型基于资料生成
    prompt = build_prompt(question, hits)
    response = llm.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )

    print("=" * 64)
    print("问题：" + question)
    print("-" * 64)
    print("AI：" + response.choices[0].message.content)
    print("-" * 64)
    print("参考来源（按相关度排序，距离越小越相关）：")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h['heading']}（距离 {h['distance']:.3f}）")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行参数即问题，例如：uv run python -m course_rag.qa "什么是PCB"
        answer(" ".join(sys.argv[1:]))
    else:
        # 无参数：两个讲义内问题 + 一个讲义外问题，观察模型表现
        for q in [
            "进程有哪几种基本状态？它们之间怎么转换？",
            "线程和进程有什么区别？",
            "什么是虚拟内存？",
        ]:
            answer(q)
