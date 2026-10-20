"""
M3 评测脚本：用标注题集量化检索质量（检索器是可插拔的，方便对比优化前后）
运行：uv run python -m course_rag.evaluate
指标：
  - Top1 准确率：最相关块是否正好在期望页（排序质量）
  - Recall@5：期望页是否出现在前 5 块（召回覆盖，Rerank 能救的前提）
  - 讲义外拒答率：超纲问题是否被阈值正确拦截
"""
import json
from collections.abc import Callable
from pathlib import Path

from course_rag.rag_engine import REJECT_THRESHOLD, RAGEngine

EVAL_PATH = Path(__file__).resolve().parents[2] / "evals" / "questions.jsonl"

Retriever = Callable[[str, int], list[dict]]


def load_questions() -> list[dict]:
    return [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(name: str, retrieve_fn: Retriever,
             reject_threshold: float = REJECT_THRESHOLD) -> dict:
    questions = load_questions()
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    top1_hit = 0
    recall5_hit = 0
    reject_hit = 0
    misses = []

    for q in answerable:
        hits = retrieve_fn(q["question"], 5)
        pages = [h["page"] for h in hits]
        if pages and pages[0] == q["expected_page"]:
            top1_hit += 1
        else:
            misses.append((q["question"], q["expected_page"], pages[:3],
                           round(hits[0]["distance"], 2)))
        if q["expected_page"] in pages:
            recall5_hit += 1

    for q in unanswerable:
        hits = retrieve_fn(q["question"], 5)
        if hits[0]["distance"] > reject_threshold:
            reject_hit += 1

    n_ans, n_un = len(answerable), len(unanswerable)
    metrics = {
        "top1": top1_hit / n_ans,
        "recall5": recall5_hit / n_ans,
        "reject": reject_hit / n_un,
        "overall": (top1_hit + reject_hit) / (n_ans + n_un),
    }
    print(f"===== 检索方案：{name} =====")
    print(f"Top1 准确率：{top1_hit}/{n_ans} = {metrics['top1']:.1%}")
    print(f"Recall@5   ：{recall5_hit}/{n_ans} = {metrics['recall5']:.1%}")
    print(f"讲义外拒答率：{reject_hit}/{n_un} = {metrics['reject']:.1%}")
    print(f"综合正确率  ：{metrics['overall']:.1%}")
    if misses:
        print("Top1 未命中明细：")
        for question, expected, got, dist in misses:
            print(f"  「{question}」期望第 {expected} 页，Top3={got}（距离 {dist}）")
    print()
    return metrics


if __name__ == "__main__":
    engine = RAGEngine()
    evaluate("纯向量检索（基线）", engine.retrieve)
