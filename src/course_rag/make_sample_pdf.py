"""
M2 准备：生成带页码的测试 PDF（模拟真实课件，每个小节一页）
运行：uv run python -m course_rag.make_sample_pdf
"""
import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from course_rag.chunking import DATA_PATH as TXT_PATH

PDF_PATH = Path(__file__).resolve().parents[2] / "data" / "lecture.pdf"

# 中文 Windows 常见字体，依次尝试
FONT_CANDIDATES = [
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def split_sections(text: str) -> list[tuple[str, str]]:
    """把讲义按 ## 标题切成小节。"""
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            body = "\n".join(buf).strip()
            if body:
                sections.append((heading, body))
            buf = []
            heading = line[3:].strip()
        elif not line.startswith("# "):
            buf.append(line)
    body = "\n".join(buf).strip()
    if body:
        sections.append((heading, body))
    return sections


def build() -> None:
    font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
    if font_path is None:
        raise SystemExit("未找到中文字体，无法生成 PDF")
    pdfmetrics.registerFont(TTFont("CN", font_path, subfontIndex=0))

    text = TXT_PATH.read_text(encoding="utf-8")
    sections = split_sections(text)

    c = canvas.Canvas(str(PDF_PATH), pagesize=A4)
    width, height = A4
    for page_no, (heading, body) in enumerate(sections, start=1):
        c.setFont("CN", 18)
        c.drawString(50, height - 70, heading)
        c.setFont("CN", 12)
        y = height - 115
        for line in textwrap.wrap(body, width=38):
            c.drawString(50, y, line)
            y -= 22
        # 页脚页码（测试来源标注用）
        c.setFont("CN", 10)
        c.drawCentredString(width / 2, 35, f"- {page_no} -")
        c.showPage()
    c.save()
    print(f"已生成测试 PDF：{PDF_PATH}（共 {len(sections)} 页）")


if __name__ == "__main__":
    build()
