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


def _pack_sentences(sentences: list[str], max_len: int) -> list[str]:
    """把句子贪心打包成不超过 max_len 字的块，单句超长时硬切。"""
    blocks: list[str] = []
    cur = ""
    for sent in sentences:
        while len(sent) > max_len:
            if cur:
                blocks.append(cur)
                cur = ""
            blocks.append(sent[:max_len])
            sent = sent[max_len:]
        if len(cur) + len(sent) <= max_len:
            cur += sent
        else:
            blocks.append(cur)
            cur = sent
    if cur:
        blocks.append(cur)
    return blocks


def chunk_text(text: str, max_len: int = CHUNK_SIZE) -> list[dict]:
    """TXT 讲义切块：先按章节，再按句子打包，每块携带章节标题。"""
    chunks: list[dict] = []
    for heading, body in split_by_heading(text):
        for block in _pack_sentences(split_by_sentence(body), max_len):
            chunks.append({"heading": heading, "text": block})
    for i, c in enumerate(chunks):
        c["index"] = i
    return chunks


def chunk_pages(pages: list[dict], max_len: int = CHUNK_SIZE) -> list[dict]:
    """PDF 切块：parse_pdf 的每页文本切块，每块携带来源文件名与页码。"""
    chunks: list[dict] = []
    for page in pages:
        for block in _pack_sentences(split_by_sentence(page["text"]), max_len):
            chunks.append(
                {
                    "text": block,
                    "source": page["source"],
                    "page": page["page"],
                }
            )
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
