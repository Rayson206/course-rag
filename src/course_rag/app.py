"""
M2 第3步：Gradio 网页问答应用
运行：uv run python -m course_rag.app
启动后浏览器自动打开 http://127.0.0.1:7860
"""
import gradio as gr

from course_rag.document_parser import PDF_PATH
from course_rag.rag_engine import REJECT_THRESHOLD, RAGEngine

engine = RAGEngine()

# 启动时确保示例讲义已索引，开箱即用
if engine.collection.count() == 0:
    engine.index_pdf(PDF_PATH)

NO_ANSWER = "📚 讲义中没有找到相关内容，换个关键词试试，或确认该内容是否在你上传的资料里。"


def upload_pdf(pdf_path: str) -> str:
    """上传 PDF 后建立索引。"""
    if not pdf_path:
        return "⚠️ 还没有选择文件"
    n = engine.index_pdf(pdf_path)
    return f"✅ 已上传并建立索引，共 {n} 个文本块，现在可以开始提问"


def format_sources(hits: list[dict]) -> str:
    """把命中的来源整理成答案下方的引用标注。"""
    lines = [
        f"{i}. {h['source']} 第 {h['page']} 页（相关度距离 {h['distance']:.2f}）"
        for i, h in enumerate(hits, 1)
    ]
    return "\n\n---\n📖 **参考来源**（距离越小越相关）：\n" + "\n".join(lines)


def respond(history: list[dict]):
    """history 为 messages 格式，最后一条是用户本轮提问。"""
    question = history[-1]["content"]
    prior = history[:-1][-6:]  # 最多携带最近 3 轮上下文
    hits = engine.retrieve(question)

    if hits[0]["distance"] > REJECT_THRESHOLD:
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
    demo.launch(inbrowser=True)
