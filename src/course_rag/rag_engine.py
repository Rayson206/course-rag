"""
M2 核心：RAG 引擎（索引构建 + 检索 + 流式问答）
网页（app.py）和命令行（qa.py）共用这一套逻辑，避免两份代码。
"""
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from course_rag.chunking import chunk_pages
from course_rag.document_parser import parse_pdf

load_dotenv()

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
TOP_K = 5
# 拒答阈值：M1 实测讲义内问题 Top1 距离约 0.33-0.41，讲义外 > 0.56，取中间值
REJECT_THRESHOLD = 0.45

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "chroma_db"


class RAGEngine:
    def __init__(self, db_path: Path = DB_PATH, collection_name: str = "course_docs"):
        self._client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url="https://api.siliconflow.cn/v1",
        )
        self._db = chromadb.PersistentClient(path=str(db_path))
        self.collection = self._db.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """批量文字转向量。"""
        resp = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]

    def index_pdf(self, pdf_path: str | Path) -> int:
        """解析 PDF → 切块 → 向量化 → 入库，返回入库块数。"""
        pages = parse_pdf(pdf_path)
        chunks = chunk_pages(pages)
        vectors = self._embed([c["text"] for c in chunks])
        self.collection.upsert(
            ids=[f"{c['source']}-{c['index']}" for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=vectors,
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
        )
        return len(chunks)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """问题向量化，检索最相关的 k 块。"""
        query_vector = self._embed([question])[0]
        r = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(r["ids"][0])):
            hits.append(
                {
                    "text": r["documents"][0][i],
                    "source": r["metadatas"][0][i]["source"],
                    "page": r["metadatas"][0][i]["page"],
                    "distance": r["distances"][0][i],
                }
            )
        return hits

    def _build_prompt(self, question: str, hits: list[dict]) -> str:
        """把参考资料（含来源页码）和问题拼成 Prompt。"""
        refs = []
        for i, h in enumerate(hits, 1):
            refs.append(f"[资料{i}]（来源：{h['source']} 第 {h['page']} 页）\n{h['text']}")
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

    def answer_stream(self, question: str, history: list[dict] | None = None):
        """流式生成答案，逐字 yield（网页打字机效果用）。"""
        hits = self.retrieve(question)
        if hits[0]["distance"] > REJECT_THRESHOLD:
            yield "📚 讲义中没有找到与这个问题相关的内容，换个关键词试试，或确认该内容是否在你上传的资料里。"
            return

        messages = list(history or [])
        messages.append({"role": "user", "content": self._build_prompt(question, hits)})
        stream = self._client.chat.completions.create(
            model=LLM_MODEL, messages=messages, stream=True
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    def answer(self, question: str) -> tuple[str, list[dict]]:
        """非流式：一次性返回完整答案和来源（命令行用）。"""
        hits = self.retrieve(question)
        if hits[0]["distance"] > REJECT_THRESHOLD:
            return "📚 讲义中没有相关内容。", hits
        resp = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": self._build_prompt(question, hits)}],
        )
        return resp.choices[0].message.content, hits
