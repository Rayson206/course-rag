"""
M2 第1步：文档解析（PDF -> 带页码的文本）
运行：uv run python -m course_rag.document_parser
"""
from pathlib import Path

import pdfplumber

PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "lecture.pdf"


def parse_pdf(path: str | Path) -> list[dict]:
    """提取 PDF 每页正文与表格，返回 [{'source': 文件名, 'page': 页码, 'text': 文本}]。"""
    pages: list[dict] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # 表格提取：pdfplumber 能识别线条分隔的表格，转成「单元格 | 单元格」文本
            for table in page.extract_tables() or []:
                for row in table:
                    cells = [(cell or "").replace("\n", " ").strip() for cell in row]
                    text += "\n" + " | ".join(cells)

            pages.append(
                {
                    "source": Path(path).name,
                    "page": page_no,
                    "text": text.strip(),
                }
            )
    return pages


if __name__ == "__main__":
    pages = parse_pdf(PDF_PATH)
    print(f"共解析 {len(pages)} 页\n")
    for p in pages:
        print(f"第 {p['page']} 页（来源 {p['source']}），{len(p['text'])} 字")
        print(f"  {p['text'][:60]}……\n")
