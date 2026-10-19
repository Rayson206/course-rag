"""
M2 命令行问答：基于 PDF 讲义的 RAG 引擎（薄壳，逻辑都在 rag_engine.py）
运行方式：
    无参数，跑三个内置演示问题：uv run python -m course_rag.qa
    问自己的问题：uv run python -m course_rag.qa "死锁的四个必要条件是什么"
"""
import sys

from course_rag.document_parser import PDF_PATH
from course_rag.rag_engine import RAGEngine

engine = RAGEngine()

# 首次运行时自动把 PDF 建入索引
if engine.collection.count() == 0:
    n = engine.index_pdf(PDF_PATH)
    print(f"首次运行，已索引 {n} 个文本块\n")


def show(question: str) -> None:
    answer, hits = engine.answer(question)
    print("=" * 64)
    print("问题：" + question)
    print("-" * 64)
    print(answer)
    print("-" * 64)
    print("参考来源（距离越小越相关）：")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. {h['source']} 第 {h['page']} 页（距离 {h['distance']:.3f}）")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        for q in [
            "进程有哪几种基本状态？它们之间怎么转换？",
            "线程和进程有什么区别？",
            "什么是虚拟内存？",
        ]:
            show(q)
