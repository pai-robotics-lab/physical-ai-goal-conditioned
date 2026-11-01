"""Build Japanese-font report diagrams from verified project data."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "final"
FONT_REGULAR = Path(r"C:\Windows\Fonts\meiryo.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\meiryob.ttc")

BACKGROUND = "#F6F9FB"
NAVY = "#12355B"
BLUE = "#1E6F8C"
GREEN = "#25855A"
LIGHT_BLUE = "#E6F1F5"
LIGHT_GREEN = "#E7F4EC"
LIGHT_ORANGE = "#FFF0DA"
GRAY = "#526777"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def _box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    fill: str = LIGHT_BLUE,
) -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=NAVY, width=3)
    x1, y1, _, _ = xy
    draw.text((x1 + 22, y1 + 18), title, font=_font(27, True), fill=NAVY)
    y = y1 + 65
    for line in lines:
        draw.text((x1 + 22, y), line, font=_font(21), fill=GRAY)
        y += 34


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = BLUE,
) -> None:
    draw.line((*start, *end), fill=color, width=6)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - 18 * direction, ey - 12), (ex - 18 * direction, ey + 12)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 12, ey - 18 * direction), (ex + 12, ey - 18 * direction)]
    draw.polygon(points, fill=color)


def build_system_overview() -> None:
    image = Image.new("RGB", (1800, 690), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text(
        (55, 35),
        "視覚AIと従来型制御のシステム構成",
        font=_font(40, True),
        fill=NAVY,
    )
    _box(draw, (60, 145, 380, 330), "入力1: RGB画像", ["Genesis固定カメラ", "640×480 → 160×120"])
    _box(
        draw,
        (60, 405, 380, 590),
        "入力2: 構造化ID",
        ["対象色ID → 視覚AI", "搬送先ID → CNN補助入力", "搬送先ID → 状態機械"],
    )
    _box(
        draw,
        (500, 215, 850, 520),
        "視覚AI",
        ["色条件付き空間CNN", "Attention + Embedding", "出力は把持位置XYのみ"],
        LIGHT_GREEN,
    )
    _box(
        draw,
        (970, 215, 1280, 520),
        "従来型制御",
        ["搬送先の既定座標を選択", "Panda IK・関節補間", "把持力・動作タイミング"],
        LIGHT_ORANGE,
    )
    _box(draw, (1400, 215, 1735, 520), "物理操作", ["把持 → 持ち上げ", "搬送 → 配置 → 解放", "Genesis物理接触のみ"], LIGHT_GREEN)
    _arrow(draw, (380, 238), (500, 300))
    _arrow(draw, (380, 495), (500, 430))
    draw.line((380, 535, 900, 535), fill=BLUE, width=6)
    _arrow(draw, (900, 535), (970, 475))
    _arrow(draw, (850, 368), (970, 368))
    _arrow(draw, (1280, 368), (1400, 368))
    draw.text(
        (490, 585),
        "AIは関節軌道を生成しない / 真値・Segmentation・scene IDは推論に不使用",
        font=_font(25, True),
        fill=GREEN,
    )
    image.save(OUTPUT / "system_architecture_ja.png")


def build_model_architecture() -> None:
    image = Image.new("RGB", (1900, 1200), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text(
        (55, 35),
        "49,366パラメータ 把持位置推定CNN",
        font=_font(40, True),
        fill=NAVY,
    )
    _box(draw, (55, 130, 410, 335), "RGB前処理", ["640×480 → 160×120", "Train mean/stdで正規化", "色コントラスト map: 1ch"])
    _box(draw, (510, 110, 925, 385), "空間CNN: 26,052 params", ["4×120×160", "Conv5 s2: 4→24, 60×80", "Conv3 s2: 24→36, 30×40", "Conv3 s1: 36→48, 30×40", "各層 BatchNorm + ReLU"], LIGHT_GREEN)
    _box(draw, (1030, 110, 1450, 385), "色条件Attention", ["対象色Embedding: 3×8", "Linear 8→48でquery化", "dot / √48 → softmax", "30×40の空間重み"], LIGHT_ORANGE)
    _box(draw, (1525, 110, 1845, 385), "特徴抽出", ["色map重心: 2次元", "Attention座標: 2", "局所特徴: 48", "全体平均特徴: 48"])
    _arrow(draw, (410, 235), (510, 235))
    _arrow(draw, (925, 235), (1030, 235))
    _arrow(draw, (1450, 235), (1525, 235))

    _box(
        draw,
        (90, 520, 455, 710),
        "構造化ID",
        ["対象色ID → 8次元", "搬送先ID → 4次元", "左右で教師把持XYは同一"],
    )
    _box(draw, (610, 480, 1050, 760), "特徴結合: 112次元", ["色map重心 2", "Attention座標 2", "局所48 + 全体48", "色Embedding 8", "搬送先Embedding 4"], LIGHT_GREEN)
    _box(draw, (1210, 480, 1580, 760), "回帰MLP: 22,850 params", ["Linear 112→128 + ReLU", "Dropout p=0.1", "Linear 128→64 + ReLU", "Linear 64→2 + Tanh"], LIGHT_ORANGE)
    _box(draw, (1660, 520, 1850, 710), "出力", ["正規化XY", "[-1, 1]", "→ world m"])
    _arrow(draw, (455, 615), (610, 615))
    _arrow(draw, (1050, 620), (1210, 620))
    _arrow(draw, (1580, 620), (1660, 620))
    _arrow(draw, (1685, 385), (930, 480))

    draw.rounded_rectangle((90, 860, 1810, 1110), radius=20, fill="#FFFFFF", outline=BLUE, width=3)
    draw.text((125, 885), "パラメータ内訳", font=_font(29, True), fill=NAVY)
    breakdown = [
        "空間CNN 26,052",
        "対象色Embedding 24",
        "搬送先Embedding 8",
        "query射影 432",
        "回帰MLP 22,850",
        "合計 49,366",
    ]
    for index, text in enumerate(breakdown):
        x = 125 + (index % 3) * 550
        y = 945 + (index // 3) * 75
        color = GREEN if text.startswith("合計") else GRAY
        draw.text((x, y), text, font=_font(25, text.startswith("合計")), fill=color)
    image.save(OUTPUT / "model_architecture_ja.png")


def build_dataset_example() -> None:
    labels = [
        json.loads(line)
        for line in (ROOT / "datasets" / "phase7" / "labels.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if '"scene_id": "scene_0001"' in line
    ]
    source = Image.open(ROOT / "datasets" / "phase7" / "images" / "scene_0001.png").convert(
        "RGB"
    )
    source = source.resize((800, 600), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1900, 1120), BACKGROUND)
    canvas.paste(source, (55, 145))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (55, 35),
        "同一RGB画像から生成する6つの構造化指示サンプル",
        font=_font(38, True),
        fill=NAVY,
    )
    draw.text(
        (55, 785),
        "scene_0001 / seed 70005 / split=train",
        font=_font(25, True),
        fill=NAVY,
    )
    draw.text(
        (55, 830),
        "画像は6サンプルで共通。対象色IDが把持対象、搬送先IDが既定搬送先を指定する。",
        font=_font(22),
        fill=GRAY,
    )
    draw.text((920, 125), "指示", font=_font(27, True), fill=NAVY)
    draw.text((1485, 125), "正解把持XY [m]", font=_font(27, True), fill=NAVY)
    colors = {"red_cube": "#C62828", "blue_cube": "#1558B0", "green_cube": "#147D3C"}
    for index, record in enumerate(labels):
        y = 190 + index * 125
        fill = "#FFFFFF" if index % 2 == 0 else "#ECF2F5"
        draw.rounded_rectangle((900, y - 20, 1830, y + 82), radius=12, fill=fill)
        draw.text(
            (930, y),
            record["instruction"],
            font=_font(23, True),
            fill=colors[record["target_class"]],
        )
        x_value, y_value = record["grasp_position_xy_world_m"]
        draw.text(
            (1490, y),
            f"({x_value:.6f}, {y_value:.6f})",
            font=_font(23),
            fill=GRAY,
        )
    draw.rounded_rectangle((55, 910, 1830, 1060), radius=15, fill=LIGHT_GREEN, outline=GREEN, width=3)
    draw.text(
        (85, 935),
        "教師: 物体中心のworld XY = phase 2で検証したtop-down把持中心",
        font=_font(25, True),
        fill=GREEN,
    )
    draw.text(
        (85, 990),
        "左右搬送先で教師把持XYは同一。異なる把持位置教師は1シーン3点。",
        font=_font(24),
        fill=GRAY,
    )
    draw.text(
        (85, 1025),
        "推論入力: RGB + 対象色ID + 搬送先ID（真値・mask・scene IDは不使用）",
        font=_font(21),
        fill=GRAY,
    )
    canvas.save(OUTPUT / "dataset_six_instructions_ja.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_system_overview()
    build_model_architecture()
    build_dataset_example()


if __name__ == "__main__":
    main()
