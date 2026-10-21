"""
M2 第3步：Gradio 网页问答应用
运行：uv run python -m course_rag.app
启动后访问 http://127.0.0.1:7860
"""
import os

import gradio as gr

from course_rag.document_parser import PDF_PATH
from course_rag.rag_engine import (
    BASE_URL,
    EMBED_MODEL,
    LLM_MODEL,
    RERANK_MODEL,
    RAGEngine,
)

print(
    f"[BOOT] base_url={BASE_URL} embed={EMBED_MODEL} rerank={RERANK_MODEL} "
    f"llm={LLM_MODEL} key_set={bool(os.getenv('SILICONFLOW_API_KEY'))}",
    flush=True,
)
engine = RAGEngine()

# 启动时确保示例讲义已索引，开箱即用
if engine.collection.count() == 0:
    print("[BOOT] collection empty, building index from built-in PDF ...", flush=True)
    engine.index_pdf(PDF_PATH)
    print(f"[BOOT] index built, chunks={engine.collection.count()}", flush=True)
else:
    print(f"[BOOT] reuse existing index, chunks={engine.collection.count()}", flush=True)

NO_ANSWER = "📚 讲义中没有找到相关内容，换个关键词试试，或确认该内容是否在你上传的资料里。"


def upload_pdf(pdf_path: str) -> str:
    """上传 PDF 后建立索引。"""
    if not pdf_path:
        return "⚠️ 还没有选择文件"
    n = engine.index_pdf(pdf_path)
    return f"✅ 已上传并建立索引，共 {n} 个文本块，现在可以开始提问"


def format_sources(hits: list[dict]) -> str:
    """把命中的来源整理成答案下方的引用标注（优先显示 Rerank 相关度）。"""
    lines = []
    for i, h in enumerate(hits, 1):
        if h.get("rerank_score") is not None:
            score = f"相关度 {h['rerank_score']:.2f}"
        elif h.get("distance") is not None:
            score = f"向量距离 {h['distance']:.2f}"
        else:
            score = ""
        lines.append(f"{i}. {h['source']} 第 {h['page']} 页（{score}）")
    return "\n\n---\n📖 **参考来源**（相关度越高越匹配）：\n" + "\n".join(lines)


def _message_text(content) -> str:
    """从 Gradio messages 的 content 提取纯文本。

    Gradio 6.x 可能给纯字符串，也可能给 OpenAI 风格的分段列表
    [{"type": "text", "text": "..."}]，两种都要兼容，否则会把 list/dict
    直接送进 embedding，云端报 400(20015 参数无效)。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for seg in content:
            parts.append(seg.get("text", "") if isinstance(seg, dict) else str(seg))
        return "".join(parts)
    return str(content)


def respond(history: list[dict]):
    """history 为 messages 格式，最后一条是用户本轮提问。"""
    question = _message_text(history[-1]["content"])
    # 统一成纯文本 content 再喂给 LLM，最多携带最近 3 轮上下文
    prior = [
        {"role": m["role"], "content": _message_text(m["content"])}
        for m in history[:-1][-6:]
    ]
    hits = engine.search(question)

    if engine.should_reject(hits):
        yield history + [{"role": "assistant", "content": NO_ANSWER}]
        return

    # 逐字流式输出，每来一小段就刷新一次界面（打字机效果）
    partial = ""
    for delta in engine.generate_stream(question, hits, prior):
        partial += delta
        yield history + [{"role": "assistant", "content": partial + format_sources(hits)}]


with gr.Blocks(title="课程讲义问答助手") as demo:
    gr.Markdown(
        "# 📚 课程讲义问答助手\n"
        "基于 RAG 的课程答疑：上传 PDF 讲义，回答只依据讲义内容，并标注来源页码"
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_box = gr.File(label="上传课程 PDF", file_types=[".pdf"])
            upload_btn = gr.Button("建立索引", variant="primary")
            status = gr.Textbox(label="索引状态", interactive=False)
            gr.Markdown(
                "### 使用说明\n"
                "1. 可上传自己的 PDF 课件（不传则使用内置操作系统示例）\n"
                "2. 在右侧针对讲义内容提问\n"
                "3. 回答末尾会显示引用来源页码"
            )
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=460, label="答疑对话")
            question = gr.Textbox(
                placeholder="针对讲义提问，例如：死锁的四个必要条件是什么？",
                label="",
            )
            send_btn = gr.Button("发送", variant="primary")

    upload_btn.click(upload_pdf, inputs=file_box, outputs=status)

    def append_user(question_text: str, history: list[dict]):
        if not question_text.strip():
            return question_text, history
        return "", history + [{"role": "user", "content": question_text}]

    # 点击按钮和回车都能发送
    send_btn.click(append_user, [question, chatbot], [question, chatbot]).then(
        respond, chatbot, chatbot
    )
    question.submit(append_user, [question, chatbot], [question, chatbot]).then(
        respond, chatbot, chatbot
    )


if __name__ == "__main__":
    # 监听 0.0.0.0 与 PORT 环境变量：本地默认 7860，HF Space 等云平台会注入 PORT
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        inbrowser=False,
    )
