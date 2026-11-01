"""Build the Japanese final report PDF from docs/final-report.md."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "final-report.md"
OUTPUT = ROOT / "output" / "pdf" / "physical-ai-goal-conditioned-final-report.pdf"


def _styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    base = getSampleStyleSheet()
    font = "HeiseiKakuGo-W5"
    return {
        "body": ParagraphStyle(
            "JapaneseBody",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.2,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=4 * mm,
            textColor=colors.HexColor("#243447"),
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading1"],
            fontName=font,
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#12355B"),
            spaceBefore=6 * mm,
            spaceAfter=4 * mm,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading2"],
            fontName=font,
            fontSize=12,
            leading=17,
            textColor=colors.HexColor("#1E6F8C"),
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#526777"),
            spaceAfter=5 * mm,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=10,
            leftIndent=5 * mm,
            rightIndent=5 * mm,
            backColor=colors.HexColor("#F1F5F8"),
            borderPadding=5,
        ),
        "toc": ParagraphStyle(
            "TOC",
            parent=base["BodyText"],
            fontName=font,
            fontSize=10.5,
            leading=17,
            leftIndent=5 * mm,
            textColor=colors.HexColor("#12355B"),
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.3,
            leading=9.5,
            alignment=TA_LEFT,
            wordWrap="CJK",
        ),
    }


def _clean_inline(text: str) -> str:
    return escape(re.sub(r"`([^`]+)`", r"\1", text))


def _table(lines: list[str], style: ParagraphStyle, available_width: float) -> Table:
    raw_rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        raw_rows.append(cells)
    raw_rows.pop(1)
    column_count = len(raw_rows[0])
    widths = [available_width / column_count] * column_count
    rows: list[list[object]] = []
    for row in raw_rows:
        rendered_row: list[object] = []
        for cell in row:
            image_match = re.fullmatch(r"!\[(.+)]\((.+)\)", cell)
            if image_match:
                path = (SOURCE.parent / image_match.group(2)).resolve()
                image = Image(str(path))
                scale = min(
                    (widths[0] - 8 * mm) / image.imageWidth,
                    55 * mm / image.imageHeight,
                )
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
                image.hAlign = "CENTER"
                rendered_row.append(
                    [image, Paragraph(_clean_inline(image_match.group(1)), STYLES["caption"])]
                )
            else:
                rendered_row.append(Paragraph(_clean_inline(cell), style))
        rows.append(rendered_row)
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12355B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB8C2")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _image_flowable(alt: str, relative_path: str, available_width: float) -> list[object]:
    path = (SOURCE.parent / relative_path).resolve()
    image = Image(str(path))
    scale = min(available_width / image.imageWidth, 100 * mm / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return [Spacer(1, 2 * mm), image, Paragraph(_clean_inline(alt), STYLES["caption"])]


def _parse_markdown(lines: list[str], available_width: float) -> list[object]:
    story: list[object] = []
    index = 0
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            story.append(Paragraph(_clean_inline("".join(paragraph_lines)), STYLES["body"]))
            paragraph_lines.clear()

    while index < len(lines):
        line = lines[index].rstrip("\n")
        if not line.strip():
            flush_paragraph()
            index += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(_clean_inline(line[3:]), STYLES["h2"]))
            index += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(_clean_inline(line[4:]), STYLES["h3"]))
            index += 1
            continue
        if line.startswith("```"):
            flush_paragraph()
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index].rstrip("\n"))
                index += 1
            index += 1
            story.append(Preformatted("\n".join(code), STYLES["code"]))
            story.append(Spacer(1, 4 * mm))
            continue
        if line.startswith("- "):
            flush_paragraph()
            items: list[ListItem] = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(
                    ListItem(
                        Paragraph(_clean_inline(lines[index][2:].strip()), STYLES["body"]),
                        leftIndent=5 * mm,
                    )
                )
                index += 1
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=8 * mm, bulletFontName="Helvetica"))
            story.append(Spacer(1, 2 * mm))
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index].rstrip("\n"))
                index += 1
            story.append(_table(table_lines, STYLES["table"], available_width))
            story.append(Spacer(1, 5 * mm))
            continue
        image_match = re.fullmatch(r"!\[(.+)]\((.+)\)", line.strip())
        if image_match:
            flush_paragraph()
            story.extend(_image_flowable(image_match.group(1), image_match.group(2), available_width))
            index += 1
            continue
        paragraph_lines.append(line.strip())
        index += 1
    flush_paragraph()
    return story


def _page(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E1E8"))
    canvas.line(22 * mm, 16 * mm, 188 * mm, 16 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526777"))
    canvas.drawString(22 * mm, 10 * mm, "Vision-Based Panda Grasping and Transport")
    canvas.drawRightString(188 * mm, 10 * mm, str(document.page))
    canvas.restoreState()


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=22 * mm,
        title="視覚AIを用いたPandaロボットアームの物体把持・搬送",
        author="Physical AI project",
    )
    story: list[object] = [Spacer(1, 25 * mm)]
    story.append(
        Paragraph(
            "視覚AIを用いた<br/>Pandaロボットアームの<br/>物体把持・搬送",
            ParagraphStyle(
                "CoverTitle",
                fontName="HeiseiKakuGo-W5",
                fontSize=23,
                leading=34,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#12355B"),
            ),
        )
    )
    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "Vision-Based Object Grasping and Transport<br/>with a Franka Panda",
            ParagraphStyle(
                "CoverEnglish",
                fontName="Helvetica",
                fontSize=13,
                leading=19,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#1E6F8C"),
            ),
        )
    )
    story.append(Spacer(1, 20 * mm))
    summary = Table(
        [
            ["AI output", "Robot control", "Paired physical success"],
            ["Grasp position XY", "Fixed state machine + IK", "15/24 -> 24/24"],
        ],
        colWidths=[54 * mm] * 3,
    )
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12355B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EAF2F6")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#12355B")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.white),
            ]
        )
    )
    story.append(summary)
    story.append(Spacer(1, 28 * mm))
    story.append(Paragraph("Physical AI応用1 最終課題", STYLES["caption"]))
    story.append(Paragraph("実験日: 2026-10-31 / 改訂日: 2026-11-01", STYLES["caption"]))
    story.append(PageBreak())
    story.append(Paragraph("目次", STYLES["h2"]))
    for title in [
        "1. 背景と研究目的",
        "2. 視覚AIによる位置推定と従来型ロボット制御",
        "3. GenesisとFranka Pandaの物理シミュレーション",
        "4. 自作CNNの構造と学習データ",
        "5. 学習データ量による位置推定精度の比較",
        "6. オンライン物理操作実験",
        "7. 把持失敗の原因分析",
        "8. 学習改善前後の物理操作成功率比較",
        "9. 考察・制約・今後の課題",
        "10. まとめ",
    ]:
        story.append(Paragraph(title, STYLES["toc"]))
    story.append(PageBreak())
    raw_lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    divider = next(index for index, line in enumerate(raw_lines) if line.strip() == "---")
    story.extend(_parse_markdown(raw_lines[divider + 1 :], document.width))
    document.build(story, onFirstPage=_page, onLaterPages=_page)
    return OUTPUT


STYLES = _styles()

if __name__ == "__main__":
    print(build())
