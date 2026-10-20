"""
FastAPI 服务化：把 RAG 问答封装成标准 HTTP 接口。

启动：uv run uvicorn course_rag.api:app --host 0.0.0.0 --port 8000
文档：启动后访问 http://127.0.0.1:8000/docs（自动生成的 Swagger 文档）
"""
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from course_rag.rag_engine import DB_PATH, PROJECT_ROOT, RAGEngine

DATA_DIR = PROJECT_ROOT / "data"


class AskRequest(BaseModel):
    question: str
    history: list[dict] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时若库为空且 data 目录里有 PDF，自动索引（容器首次启动用）。"""
    global engine
    engine = RAGEngine()
    if engine.collection.count() == 0 and DATA_DIR.exists():
        for pdf in sorted(DATA_DIR.glob("*.pdf")):
            engine.index_pdf(pdf)
    yield


app = FastAPI(title="course-rag 课程讲义问答 API", lifespan=lifespan)
engine: RAGEngine


@app.get("/health")
def health():
    """健康检查：服务是否存活、索引了多少块。"""
    return {"status": "ok", "indexed_chunks": engine.collection.count()}


@app.post("/api/ask")
def ask(req: AskRequest):
    """非流式问答：返回完整答案 + 来源页码。"""
    answer, hits = engine.answer(req.question)
    return {
        "answer": answer,
        "sources": [
            {"source": h["source"], "page": h["page"],
             "rerank_score": round(h["rerank_score"], 4) if h.get("rerank_score") else None}
            for h in hits
        ],
    }


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """流式问答：逐字返回答案（打字机效果）。"""
    def gen():
        for chunk in engine.answer_stream(req.question, req.history):
            yield chunk
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.post("/api/index")
async def index_pdf(file: UploadFile = File(...)):
    """上传 PDF 建索引（可重复调用，会自动重建 BM25）。"""
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "仅支持 PDF 文件"}
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    n = engine.index_pdf(tmp_path)
    return {"filename": file.filename, "indexed_chunks": n}
