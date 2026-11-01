"""Compose a silent 1-3 minute final demo video from verified phase-8 recordings."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT / "recordings" / "phase8"
CARDS = RECORDINGS / "cards"
SEGMENTS = RECORDINGS / "segments"
OUTPUT = ROOT / "videos" / "physical-ai-goal-conditioned-final-demo.mp4"
FONT_REGULAR = Path(r"C:\Windows\Fonts\meiryo.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\meiryob.ttc")
FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        current = ""
        for character in raw_line:
            candidate = current + character
            if draw.textbbox((0, 0), candidate, font=font)[2] > width and current:
                lines.append(current)
                current = character
            else:
                current = candidate
        lines.append(current)
    return lines


def _card(path: Path, title: str, lines: list[str], image_path: Path | None = None) -> None:
    canvas = Image.new("RGB", (1280, 720), "#07131f")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1280, 18), fill="#31a354")
    draw.text((70, 55), title, font=_font(FONT_BOLD, 42), fill="#f7fbff")
    text_width = 600 if image_path else 1120
    y = 145
    body_font = _font(FONT_REGULAR, 28)
    for line in lines:
        for fitted in _fit_text(draw, line, body_font, text_width):
            draw.text((80, y), fitted, font=body_font, fill="#dce9f2")
            y += 46
        y += 10
    if image_path:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((540, 460), Image.Resampling.LANCZOS)
        x = 700 + (520 - image.width) // 2
        image_y = 150 + (460 - image.height) // 2
        canvas.paste(image, (x, image_y))
        draw.rectangle((x - 2, image_y - 2, x + image.width + 2, image_y + image.height + 2), outline="#7fcdbb", width=3)
    draw.text(
        (70, 670),
        "Vision-Based Panda Grasping and Transport",
        font=_font(FONT_REGULAR, 18),
        fill="#7fa6bf",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _run(arguments: list[str]) -> None:
    completed = subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def _still_segment(image: Path, duration: float, output: Path) -> None:
    _run(
        [
            str(FFMPEG), "-y", "-loop", "1", "-i", str(image), "-t", str(duration),
            "-vf", "fps=30,format=yuv420p", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", str(output),
        ]
    )


def _raw_segment(video: Path, label: str, output: Path) -> None:
    font = FONT_BOLD.as_posix().replace(":", r"\:")
    filter_value = (
        "scale=960:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:#07131f,fps=30,"
        f"drawtext=fontfile='{font}':text='{label}':"
        "x=40:y=35:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=12,"
        "format=yuv420p"
    )
    _run(
        [
            str(FFMPEG), "-y", "-i", str(video), "-vf", filter_value, "-an", "-c:v",
            "libx264", "-preset", "medium", "-crf", "20", str(output),
        ]
    )


def _comparison_segment(old_video: Path, improved_video: Path, output: Path) -> None:
    font = FONT_BOLD.as_posix().replace(":", r"\:")
    filter_value = (
        "[0:v]scale=640:480,setpts=PTS-STARTPTS[left];"
        "[1:v]scale=640:480,setpts=PTS-STARTPTS[right];"
        "[left][right]hstack=inputs=2,pad=1280:720:0:120:#07131f,"
        f"drawtext=fontfile='{font}':text='OLD MODEL - FAIL':x=55:y=45:fontsize=30:"
        "fontcolor=#ff8a80:box=1:boxcolor=black@0.55:boxborderw=10,"
        f"drawtext=fontfile='{font}':text='IMPROVED MODEL - SUCCESS':x=700:y=45:fontsize=30:"
        "fontcolor=#7fcdbb:box=1:boxcolor=black@0.55:boxborderw=10,"
        "fps=30,format=yuv420p"
    )
    _run(
        [
            str(FFMPEG), "-y", "-i", str(old_video), "-i", str(improved_video),
            "-filter_complex", filter_value, "-shortest", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", str(output),
        ]
    )


def build() -> Path:
    old_json = json.loads((RECORDINGS / "old_trial01.json").read_text(encoding="utf-8"))
    improved_json = json.loads(
        (RECORDINGS / "improved_trial01.json").read_text(encoding="utf-8")
    )
    CARDS.mkdir(parents=True, exist_ok=True)
    SEGMENTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _card(
        CARDS / "01_title.png",
        "視覚AIを用いたPandaロボットアームの物体把持・搬送",
        [
            "CNNがRGB画像と構造化IDから把持位置XYを推定",
            "固定状態機械・IK・把持力制御で物理操作",
            "追加評価として学習データ量と物理成功率を比較",
        ],
    )
    _card(
        CARDS / "02_system.png",
        "視覚AIと従来型制御の役割",
        [
            "入力: RGB + 対象色ID + 搬送先ID",
            "AI出力: ワールド座標の把持位置XYのみ",
            "49,366パラメータの小型CNN",
            "軌道・把持力・搬送先選択は従来型制御",
        ],
        ROOT / "docs" / "assets" / "final" / "improved_model_trial01_prediction.png",
    )
    _card(
        CARDS / "05_data.png",
        "学習データ量と位置推定精度",
        ["Train 80 -> 500 -> 2,000 scenes", "共通Test平均誤差 12.48 mm -> 2.14 mm"],
        ROOT / "docs" / "assets" / "final" / "data_scale_vs_error.png",
    )
    _card(
        CARDS / "06_physical.png",
        "同一24配置の物理操作評価",
        ["旧モデル: 15/24 = 62.5%", "改善モデル: 24/24 = 100%", "同一の固定制御器で比較"],
        ROOT / "docs" / "assets" / "final" / "physical_success_comparison.png",
    )
    _card(
        CARDS / "07_conclusion.png",
        "結論",
        [
            "データ増量だけで位置推定精度を改善",
            "旧モデルの把持失敗9件をすべて成功へ変更",
            "100%は固定カメラ・立方体・評価24配置に限定",
        ],
    )
    segments = [SEGMENTS / f"segment_{index:02d}.mp4" for index in range(1, 8)]
    _still_segment(CARDS / "01_title.png", 8, segments[0])
    _still_segment(CARDS / "02_system.png", 9, segments[1])
    _raw_segment(RECORDINGS / "improved_trial01.mp4", "VISION AI + FIXED ROBOT CONTROL", segments[2])
    _comparison_segment(
        RECORDINGS / "old_trial01.mp4", RECORDINGS / "improved_trial01.mp4", segments[3]
    )
    _still_segment(CARDS / "05_data.png", 10, segments[4])
    _still_segment(CARDS / "06_physical.png", 10, segments[5])
    _still_segment(CARDS / "07_conclusion.png", 9, segments[6])
    concat_file = SEGMENTS / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{segment.as_posix()}'" for segment in segments) + "\n",
        encoding="utf-8",
    )
    _run(
        [
            str(FFMPEG), "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", "-movflags", "+faststart", str(OUTPUT),
        ]
    )
    manifest = {
        "output": str(OUTPUT.resolve()),
        "old_recording": old_json,
        "improved_recording": improved_json,
        "segments": [str(segment.resolve()) for segment in segments],
        "silent": True,
        "model_or_controller_changed": False,
    }
    (RECORDINGS / "final_video_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return OUTPUT


if __name__ == "__main__":
    print(build())
