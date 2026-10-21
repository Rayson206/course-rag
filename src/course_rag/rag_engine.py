"""
RAG 引擎：索引构建 + 混合检索（向量 + BM25，RRF 融合）+ 流式问答
网页（app.py）和命令行（qa.py）共用这一套逻辑，避免两份代码。
"""
import os
from pathlib import Path

import chromadb
import httpx
import jieba
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from course_rag.chunking import chunk_pages
from course_rag.document_parser import parse_pdf

load_dotenv()

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# 默认用官方海外加速端点 api-st.siliconflow.cn（同一把中国区 key，国内外均可用），
# 避免 Render 等海外节点直连 api.siliconflow.cn 偶发 400(20015 参数无效)。
# 国内本地若想更快，可在 .env 设 SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1。
# 注意：api.siliconflow.com 是国际站独立账号体系，中国区 key 会 401，切勿混用。
BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api-st.siliconflow.cn/v1")
TOP_K = 5
CANDIDATE_K = 10  # 每路检索先粗召回这么多块，再融合排序
RRF_K = 60        # Reciprocal Rank Fusion 常数，越大越弱化头部排名差异
# 拒答阈值（余弦距离，越小越相关）：强相关约 0.26-0.54、完全无关 > 0.57
REJECT_THRESHOLD = 0.55
# Rerank 拒答阈值（relevance_score，越大越相关）：实测讲义内 > 0.75、讲义外 < 0.001
RERANK_REJECT = 0.1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "chroma_db"

NO_ANSWER = "📚 讲义中没有找到与这个问题相关的内容，换个关键词试试，或确认该内容是否在你上传的资料里。"


class RAGEngine:
    def __init__(self, db_path: Path = DB_PATH, collection_name: str = "course_docs"):
        self._client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=BASE_URL,
        )
        self._db = chromadb.PersistentClient(path=str(db_path))
        self.collection = self._db.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._rebuild_bm25()

    # ---------------- 索引 ----------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """批量文字转向量。"""
        try:
            resp = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
        except Exception as e:
            import sys, json
            import httpx as _httpx
            key = os.getenv("SILICONFLOW_API_KEY") or ""
            print(
                f"[EMBED ERROR] base_url={BASE_URL} model={EMBED_MODEL} n={len(texts)} "
                f"lens={[len(t) if isinstance(t, str) else 'NONSTR' for t in texts]} "
                f"repr0={repr(texts[0])[:120] if texts else 'EMPTY'} "
                f"key_prefix={key[:7]} key_len={len(key)} key_stripped={key == key.strip()} "
                f"err={type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )
            # 用原生 httpx 按同样参数重发一次，剥离 SDK 因素，看真实请求/响应
            try:
                body = {"model": EMBED_MODEL, "input": [str(t) for t in texts]}
                print(f"[EMBED DEBUG] raw request body={json.dumps(body, ensure_ascii=False)[:600]}",
                      file=sys.stderr, flush=True)
                r = _httpx.post(
                    f"{BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json=body, timeout=30,
                )
                print(f"[EMBED DEBUG] httpx retry status={r.status_code} body={r.text[:600]}",
                      file=sys.stderr, flush=True)
            except Exception as e2:
                print(f"[EMBED DEBUG] httpx retry threw: {type(e2).__name__}: {e2}",
                      file=sys.stderr, flush=True)
            raise
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
        self._rebuild_bm25()
        return len(chunks)

    # ---------------- 检索 ----------------

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """向量检索（M1 基线，保留用于方案对比）。"""
        query_vector = self._embed([question])[0]
        r = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": r["ids"][0][i],
                "text": r["documents"][0][i],
                "source": r["metadatas"][0][i]["source"],
                "page": r["metadatas"][0][i]["page"],
                "distance": r["distances"][0][i],
            }
            for i in range(len(r["ids"][0]))
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文分词：BM25 是词匹配，中文必须先分词（jieba）。"""
        return [t for t in jieba.lcut(text) if t.strip()]

    def _load_all_chunks(self) -> list[dict]:
        """从向量库取出全部块，作为 BM25 的语料。"""
        data = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": cid, "text": doc, "source": meta["source"], "page": meta["page"]}
            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    def _rebuild_bm25(self) -> None:
        """重建 BM25 索引（上传新 PDF 后调用）。"""
        self._chunks = self._load_all_chunks()
        tokenized = [self._tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _bm25_retrieve(self, question: str, n: int = CANDIDATE_K) -> list[dict]:
        """BM25 关键词检索（TF-IDF 的近亲，词频 + 文档长度归一化）。"""
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(self._tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for idx in order[:n]:
            if scores[idx] <= 0:
                continue
            hits.append({**self._chunks[idx], "bm25_score": float(scores[idx]),
                         "distance": None})
        return hits

    def hybrid_retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """向量 + BM25 双路召回，用 RRF 倒数排名融合。"""
        vector_hits = self.retrieve(question, CANDIDATE_K)
        bm25_hits = self._bm25_retrieve(question, CANDIDATE_K)

        fused: dict[str, dict] = {}
        for hits in (vector_hits, bm25_hits):
            for rank, h in enumerate(hits):
                rec = fused.setdefault(h["id"], {**h, "rrf": 0.0})
                rec["rrf"] += 1 / (RRF_K + rank + 1)
                if h.get("distance") is not None:
                    rec["distance"] = h["distance"]
                if h.get("bm25_score") is not None:
                    rec["bm25_score"] = h["bm25_score"]

        ranked = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)
        return ranked[:top_k]

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        """Cross-Encoder 精排：把问题与每个候选块直接配对打分，比向量相似度更准。"""
        resp = httpx.post(
            f"{BASE_URL}/rerank",
            headers={"Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}"},
            json={
                "model": RERANK_MODEL,
                "query": question,
                "documents": [c["text"] for c in candidates],
            },
            timeout=60,
        )
        if resp.status_code != 200:
            import sys
            print(f"[RERANK ERROR] base_url={BASE_URL} model={RERANK_MODEL} "
                  f"status={resp.status_code} body={resp.text[:500]}",
                  file=sys.stderr, flush=True)
        resp.raise_for_status()
        results = resp.json()["results"]
        ranked = []
        for item in sorted(results, key=lambda x: x["relevance_score"], reverse=True):
            ranked.append({**candidates[item["index"]],
                           "rerank_score": item["relevance_score"]})
        return ranked

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """完整检索管道：混合召回 → Rerank 精排。"""
        candidates = self.hybrid_retrieve(question, CANDIDATE_K)
        return self.rerank(question, candidates)[:top_k]

    @staticmethod
    def best_distance(hits: list[dict]) -> float:
        """候选块中最小的向量距离（用于拒答判定），无向量分则视为不相关。"""
        distances = [h["distance"] for h in hits if h.get("distance") is not None]
        return min(distances) if distances else 1.0

    def should_reject(self, hits: list[dict]) -> bool:
        """统一拒答判定：有 Rerank 分用 Rerank 阈值，否则退回向量距离阈值。"""
        if not hits:
            return True
        top = hits[0]
        if top.get("rerank_score") is not None:
            return top["rerank_score"] < RERANK_REJECT
        return self.best_distance(hits) > REJECT_THRESHOLD

    # ---------------- 生成 ----------------

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

    def generate_stream(self, question: str, hits: list[dict],
                        history: list[dict] | None = None):
        """基于已检索的 hits 流式生成答案（网页打字机效果用）。"""
        messages = list(history or [])
        messages.append({"role": "user", "content": self._build_prompt(question, hits)})
        stream = self._client.chat.completions.create(
            model=LLM_MODEL, messages=messages, stream=True
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    def answer_stream(self, question: str, history: list[dict] | None = None):
        """完整管道：混合召回 + Rerank 精排 + 阈值 + 流式生成。"""
        hits = self.search(question)
        if self.should_reject(hits):
            yield NO_ANSWER
            return
        yield from self.generate_stream(question, hits, history)

    def answer(self, question: str) -> tuple[str, list[dict]]:
        """非流式：一次性返回完整答案和来源（命令行用）。"""
        hits = self.search(question)
        if self.should_reject(hits):
            return "📚 讲义中没有相关内容。", hits
        resp = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": self._build_prompt(question, hits)}],
        )
        return resp.choices[0].message.content, hits
