"""
M1 第1步：文档切块（Chunking）
运行：uv run python -m course_rag.chunking
作用：把讲义切成带标题的小块，每一块就是后续检索的最小单元。
"""
from pathlib import Path

# 切块的核心调参旋钮（M3 会拿评测数据来调它，现在先记住这两个数字）
CHUNK_SIZE = 300  # 每块最多约 300 字

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "lecture.txt"


def split_by_heading(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题（# / ## / ###）把文档切成 (标题, 正文) 列表。"""
    sections: list[tuple[str, str]] = []
    heading = "（开头）"
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((heading, body))

    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            buf = []
            heading = line.lstrip("#").strip()
        else:
            buf.append(line)
    flush()
    return sections


def split_by_sentence(text: str) -> list[str]:
    """按中文句号/问号/叹号拆成句子（保留标点）。"""
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？":
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


def chunk_text(text: str, max_len: int = CHUNK_SIZE) -> list[dict]:
    """完整切块流程：先按章节，再按句子打包，每块携带所属章节标题。"""
    chunks: list[dict] = []
    for heading, body in split_by_heading(text):
        cur = ""
        for sent in split_by_sentence(body):
            # 单句本身超长时，硬切
            while len(sent) > max_len:
                if cur:
                    chunks.append({"heading": heading, "text": cur})
                    cur = ""
                chunks.append({"heading": heading, "text": sent[:max_len]})
                sent = sent[max_len:]
            if len(cur) + len(sent) <= max_len:
                cur += sent
            else:
                chunks.append({"heading": heading, "text": cur})
                cur = sent
        if cur:
            chunks.append({"heading": heading, "text": cur})

    for i, c in enumerate(chunks):
        c["index"] = i
    return chunks


if __name__ == "__main__":
    text = DATA_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    print(f"文档共切成 {len(chunks)} 块（每块上限 {CHUNK_SIZE} 字）\n")
    for c in chunks:
        print(f"[{c['index']}] 《{c['heading']}》 {len(c['text'])} 字")
        print(f"    {c['text'][:48]}……")
        print()
