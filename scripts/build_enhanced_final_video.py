"""Build an annotated silent final demo from verified logs and recordings."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pai_goal_conditioned.goal_vision import project_world_to_pixel

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT / "recordings" / "phase8"
OUTPUT = ROOT / "videos" / "physical-ai-goal-conditioned-final-demo-enhanced.mp4"
TEMP_VIDEO = RECORDINGS / "enhanced_temp.mp4"
FONT_REGULAR = Path(r"C:\Windows\Fonts\meiryo.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\meiryob.ttc")
WIDTH, HEIGHT, FPS = 1280, 720, 30
BACKGROUND = "#07131f"
NAVY = "#12355b"
WHITE = "#f7fbff"
MUTED = "#b8cbd8"
GREEN = "#5ad18a"
RED = "#ff746c"
CYAN = "#60c7e8"
PURPLE = "#c59cff"

STAGE_LABELS = {
    "initial": "初期認識",
    "move_above": "対象へ接近",
    "descend": "把持高さへ下降",
    "close": "グリッパー閉・把持",
    "lift": "持ち上げ",
    "transport": "左搬送先へ搬送",
    "place": "下降・配置",
    "open": "グリッパー開・解放",
    "retreat": "退避",
    "settle": "静止確認",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        current = ""
        for character in raw_line:
            candidate = current + character
            if current and draw.textbbox((0, 0), candidate, font=font)[2] > width:
                lines.append(current)
                current = character
            else:
                current = candidate
        lines.append(current)
    return lines


def _base(title: str, subtitle: str = "") -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 16), fill="#31a354")
    draw.text((55, 38), title, font=_font(38, True), fill=WHITE)
    if subtitle:
        draw.text((58, 94), subtitle, font=_font(22), fill=MUTED)
    draw.text(
        (55, 682),
        "Goal-Conditioned Visual Robotic Manipulation",
        font=_font(17),
        fill="#789bb1",
    )
    return image


def _card(title: str, lines: list[str], image_path: Path | None = None) -> Image.Image:
    image = _base(title)
    draw = ImageDraw.Draw(image)
    text_width = 560 if image_path else 1120
    y = 145
    for line in lines:
        font = _font(28, line.startswith("強調:"))
        color = GREEN if line.startswith("強調:") else WHITE
        cleaned = line.removeprefix("強調:")
        for fitted in _fit(draw, cleaned, font, text_width):
            draw.text((70, y), fitted, font=font, fill=color)
            y += 43
        y += 12
    if image_path:
        content = Image.open(image_path).convert("RGB")
        content.thumbnail((570, 480), Image.Resampling.LANCZOS)
        x = 665 + (570 - content.width) // 2
        image_y = 135 + (480 - content.height) // 2
        image.paste(content, (x, image_y))
        draw.rounded_rectangle(
            (x - 4, image_y - 4, x + content.width + 4, image_y + content.height + 4),
            radius=6,
            outline=CYAN,
            width=3,
        )
    return image


def _draw_cross(draw: ImageDraw.ImageDraw, point: tuple[float, float], color: str, size: int = 13) -> None:
    x, y = point
    draw.line((x - size, y, x + size, y), fill=color, width=5)
    draw.line((x, y - size, x, y + size), fill=color, width=5)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def _draw_destination(draw: ImageDraw.ImageDraw, point: tuple[float, float], label: str) -> None:
    x, y = point
    draw.ellipse((x - 18, y - 18, x + 18, y + 18), outline=PURPLE, width=5)
    draw.text((x + 22, y - 16), label, font=_font(20, True), fill=PURPLE)


def _red_bbox(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    mask = (red > 100) & (red - green > 45) & (red - blue > 45) & (green < 130)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _stage_at(frame_index: int, stage_frames: dict[str, int]) -> str:
    cursor = 0
    for stage, count in stage_frames.items():
        cursor += count
        if frame_index < cursor:
            return stage
    return "settle"


class SequentialReader:
    def __init__(self, path: Path) -> None:
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open {path}")
        self.index = -1
        self.frame: np.ndarray | None = None

    def get(self, target_index: int) -> np.ndarray:
        while self.index < target_index:
            ok, frame = self.capture.read()
            if not ok:
                break
            self.index += 1
            self.frame = frame
        if self.frame is None:
            raise RuntimeError("Recording contains no frames")
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.capture.release()


def _goal_conditioning_card(improved_results: dict[str, Any], camera: dict[str, Any]) -> Image.Image:
    check = improved_results["instruction_sensitivity_check"]
    source = Image.open(
        ROOT / "outputs" / "phase7" / "physical" / "improved" / check["source_image"]
    ).convert("RGB")
    source = source.resize((720, 540), Image.Resampling.LANCZOS)
    image = _base(
        "同じRGB画像でも、対象色指示で把持点が変わる",
        "改善モデルの実推論ログ / destination = left_bin / ロボット未実行のオフライン可視化",
    )
    image.paste(source, (35, 145))
    draw = ImageDraw.Draw(image)
    colors = {"red_cube": RED, "blue_cube": "#5794ff", "green_cube": GREEN}
    names = {"red_cube": "赤", "blue_cube": "青", "green_cube": "緑"}
    predictions = check["predictions_by_target_class_m"]
    cube_z = 0.224945
    label_offsets = {
        "red_cube": (18, 4),
        "blue_cube": (18, -45),
        "green_cube": (18, -22),
    }
    for target_class, xy in predictions.items():
        pixel = project_world_to_pixel([xy[0], xy[1], cube_z], camera)
        display = (35 + pixel[0] * 720 / 640, 145 + pixel[1] * 540 / 480)
        _draw_cross(draw, display, colors[target_class], 15)
        label_offset = label_offsets[target_class]
        draw.text(
            (display[0] + label_offset[0], display[1] + label_offset[1]),
            f"{names[target_class]}指示",
            font=_font(21, True),
            fill=colors[target_class],
        )
    draw.rounded_rectangle((785, 145, 1245, 625), radius=18, fill="#102637", outline=CYAN, width=3)
    draw.text((815, 175), "入力する対象色ID", font=_font(27, True), fill=WHITE)
    y = 240
    for target_class in ("red_cube", "blue_cube", "green_cube"):
        xy = predictions[target_class]
        draw.text(
            (815, y),
            f"{names[target_class]} → ({xy[0]:.6f}, {xy[1]:.6f}) m",
            font=_font(22, True),
            fill=colors[target_class],
        )
        y += 78
    draw.text((815, 500), "3予測点はすべて異なる", font=_font(25, True), fill=GREEN)
    draw.text((815, 550), "真値・maskは推論入力に不使用", font=_font(20), fill=MUTED)
    return image


def _prediction_card(log: dict[str, Any], camera: dict[str, Any]) -> Image.Image:
    source = Image.open(
        ROOT / "outputs" / "phase7" / "physical" / "improved" / "images" / "trial_01_input.png"
    ).convert("RGB")
    source = source.resize((760, 570), Image.Resampling.LANCZOS)
    image = _base("AI入力と推定結果", "実行ログ: seed 90000 / improved Train 2,000 model")
    image.paste(source, (25, 130))
    draw = ImageDraw.Draw(image)
    scale_x, scale_y = 760 / 640, 570 / 480
    predicted = project_world_to_pixel([*log["predicted_xy_m"], 0.224945], camera)
    truth = project_world_to_pixel([*log["truth_xy_m"], 0.224945], camera)
    destination = project_world_to_pixel([0.42, 0.22, 0.225], camera)
    predicted_display = (25 + predicted[0] * scale_x, 130 + predicted[1] * scale_y)
    truth_display = (25 + truth[0] * scale_x, 130 + truth[1] * scale_y)
    destination_display = (25 + destination[0] * scale_x, 130 + destination[1] * scale_y)
    _draw_cross(draw, predicted_display, RED, 18)
    draw.ellipse(
        (
            truth_display[0] - 13,
            truth_display[1] - 13,
            truth_display[0] + 13,
            truth_display[1] + 13,
        ),
        outline=GREEN,
        width=5,
    )
    _draw_destination(draw, destination_display, "左搬送先")
    draw.rounded_rectangle((810, 130, 1250, 650), radius=18, fill="#102637", outline=CYAN, width=3)
    lines = [
        ("入力指示", WHITE, True),
        ("red cube to left bin", RED, True),
        ("対象: 赤立方体", WHITE, False),
        ("搬送先: 左 / (0.42, 0.22) m", PURPLE, False),
        ("CNN推定把持XY", WHITE, True),
        (f"({log['predicted_xy_m'][0]:.6f},", RED, False),
        (f" {log['predicted_xy_m'][1]:.6f}) m", RED, False),
        (f"評価誤差: {log['prediction_error_mm']:.2f} mm", GREEN, True),
        ("赤十字=AI予測 / 緑円=評価用真値", MUTED, False),
    ]
    y = 165
    for text, color, bold in lines:
        draw.text((840, y), text, font=_font(22, bold), fill=color)
        y += 49
    return image


def _annotate_action(
    rgb: np.ndarray,
    log: dict[str, Any],
    camera: dict[str, Any],
    source_index: int,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    video = Image.fromarray(rgb).resize((920, 690), Image.Resampling.LANCZOS)
    image.paste(video, (0, 15))
    draw = ImageDraw.Draw(image)
    stage = _stage_at(source_index, log["stage_frames"])
    bbox = _red_bbox(rgb)
    if bbox:
        x1, y1, x2, y2 = bbox
        scaled = (x1 * 920 / 640, 15 + y1 * 690 / 480, x2 * 920 / 640, 15 + y2 * 690 / 480)
        draw.rectangle(scaled, outline=RED, width=5)
        if stage in {"transport", "place", "open", "retreat", "settle"}:
            label_y = min(660, scaled[3] + 5)
        else:
            label_y = max(20, scaled[1] - 31)
        draw.text((scaled[0], label_y), "対象: 赤", font=_font(21, True), fill=RED)
    if stage in {"initial", "move_above", "descend", "close"}:
        point = project_world_to_pixel([*log["predicted_xy_m"], 0.224945], camera)
        _draw_cross(draw, (point[0] * 920 / 640, 15 + point[1] * 690 / 480), RED, 14)
    destination = project_world_to_pixel([0.42, 0.22, 0.225], camera)
    _draw_destination(
        draw,
        (destination[0] * 920 / 640, 15 + destination[1] * 690 / 480),
        "左搬送先",
    )
    draw.rectangle((920, 0, WIDTH, HEIGHT), fill="#0c2030")
    draw.text((945, 30), "実行中の指示", font=_font(25, True), fill=WHITE)
    draw.text((945, 75), "red cube", font=_font(27, True), fill=RED)
    draw.text((945, 112), "to left bin", font=_font(27, True), fill=PURPLE)
    draw.line((945, 160, 1250, 160), fill="#35556a", width=2)
    draw.text((945, 183), "CNN推定把持XY", font=_font(22, True), fill=WHITE)
    draw.text((945, 223), f"X {log['predicted_xy_m'][0]:.6f} m", font=_font(20), fill=CYAN)
    draw.text((945, 257), f"Y {log['predicted_xy_m'][1]:.6f} m", font=_font(20), fill=CYAN)
    draw.text((945, 307), "現在の動作状態", font=_font(22, True), fill=WHITE)
    draw.rounded_rectangle((940, 350, 1260, 415), radius=12, fill="#17425a")
    draw.text((960, 365), STAGE_LABELS[stage], font=_font(25, True), fill=GREEN)
    draw.text((945, 455), "制御: 状態機械 + IK", font=_font(20), fill=MUTED)
    draw.text((945, 490), "真値補正・再試行なし", font=_font(20), fill=MUTED)
    if stage == "settle":
        draw.rounded_rectangle((940, 555, 1260, 655), radius=12, fill="#145b3a")
        draw.text((960, 575), "結果: 配置成功", font=_font(28, True), fill=WHITE)
        draw.text((960, 615), "最終誤差 4.12 mm", font=_font(21), fill=GREEN)
    return image


def _comparison_frame(
    old_rgb: np.ndarray,
    improved_rgb: np.ndarray,
    old_log: dict[str, Any],
    improved_log: dict[str, Any],
    camera: dict[str, Any],
    source_index: int,
) -> Image.Image:
    image = _base(
        "同一seed・配置・指示で新旧モデルを比較",
        "共通指示: red cube to left bin / seed 90000 / 初期物体位置は同一",
    )
    old_view = Image.fromarray(old_rgb).resize((600, 450), Image.Resampling.LANCZOS)
    improved_view = Image.fromarray(improved_rgb).resize((600, 450), Image.Resampling.LANCZOS)
    image.paste(old_view, (20, 145))
    image.paste(improved_view, (660, 145))
    draw = ImageDraw.Draw(image)
    stage = _stage_at(source_index, old_log["stage_frames"])
    for rgb, offset, title, error, color in (
        (old_rgb, 20, "旧モデル", old_log["prediction_error_mm"], RED),
        (improved_rgb, 660, "改善モデル", improved_log["prediction_error_mm"], GREEN),
    ):
        draw.rectangle((offset, 145, offset + 600, 595), outline=color, width=4)
        draw.text((offset + 18, 160), title, font=_font(25, True), fill=color)
        draw.text((offset + 18, 198), f"把持位置誤差 {error:.2f} mm", font=_font(21, True), fill=WHITE)
        bbox = _red_bbox(rgb)
        if bbox:
            x1, y1, x2, y2 = bbox
            scaled = (
                offset + x1 * 600 / 640,
                145 + y1 * 450 / 480,
                offset + x2 * 600 / 640,
                145 + y2 * 450 / 480,
            )
            draw.rectangle(scaled, outline=RED, width=4)
    draw.rounded_rectangle((20, 610, 620, 670), radius=10, fill="#4d2024")
    draw.rounded_rectangle((660, 610, 1260, 670), radius=10, fill="#145b3a")
    if stage in {"lift", "transport", "place", "open", "retreat", "settle"}:
        old_status = "旧: 把持失敗（対象を持ち上げられない）"
        improved_status = "改善: 把持成功 → 搬送・配置成功"
    else:
        old_status = f"旧: {STAGE_LABELS[stage]}"
        improved_status = f"改善: {STAGE_LABELS[stage]}"
    draw.text((38, 624), old_status, font=_font(21, True), fill=WHITE)
    draw.text((678, 624), improved_status, font=_font(21, True), fill=WHITE)
    return image


def _write_still(writer: cv2.VideoWriter, image: Image.Image, duration: float) -> int:
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    count = round(duration * FPS)
    for _ in range(count):
        writer.write(frame)
    return count


def _write_action(
    writer: cv2.VideoWriter,
    reader: SequentialReader,
    log: dict[str, Any],
    camera: dict[str, Any],
    duration: float,
) -> int:
    count = round(duration * FPS)
    max_index = log["frame_count"] - 1
    for output_index in range(count):
        source_index = min(int(output_index / FPS * log["fps"]), max_index)
        frame = _annotate_action(reader.get(source_index), log, camera, source_index)
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    return count


def _write_comparison(
    writer: cv2.VideoWriter,
    old_reader: SequentialReader,
    improved_reader: SequentialReader,
    old_log: dict[str, Any],
    improved_log: dict[str, Any],
    camera: dict[str, Any],
    duration: float,
) -> int:
    count = round(duration * FPS)
    max_index = min(old_log["frame_count"], improved_log["frame_count"]) - 1
    for output_index in range(count):
        source_index = min(int(output_index / FPS * old_log["fps"]), max_index)
        frame = _comparison_frame(
            old_reader.get(source_index),
            improved_reader.get(source_index),
            old_log,
            improved_log,
            camera,
            source_index,
        )
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    return count


def _transcode() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(TEMP_VIDEO),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUTPUT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])


def build() -> Path:
    old_log = _load_json(RECORDINGS / "old_trial01.json")
    improved_log = _load_json(RECORDINGS / "improved_trial01.json")
    improved_results = _load_json(
        ROOT / "outputs" / "phase7" / "physical" / "improved" / "phase5_results.json"
    )
    phase3 = _load_json(ROOT / "configs" / "phase3_dataset.json")
    camera = phase3["camera"]
    if old_log["seed"] != improved_log["seed"]:
        raise ValueError("Comparison recordings must use the same seed")
    if (old_log["target_class"], old_log["destination_id"]) != (
        improved_log["target_class"],
        improved_log["destination_id"],
    ):
        raise ValueError("Comparison recordings must use the same instruction")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(TEMP_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {TEMP_VIDEO}")
    frame_counts: dict[str, int] = {}
    old_reader = SequentialReader(RECORDINGS / "old_trial01.mp4")
    improved_reader = SequentialReader(RECORDINGS / "improved_trial01.mp4")
    action_reader = SequentialReader(RECORDINGS / "improved_trial01.mp4")
    try:
        frame_counts["title"] = _write_still(
            writer,
            _card(
                "視覚・目標条件付きAIによる自律物体操作",
                [
                    "RGB画像 + 対象色ID + 搬送先ID → 把持位置XY",
                    "AI予測を状態機械・IKへ渡し、Pandaが物理接触で操作",
                    "学習データ拡張による把持性能改善を同一条件で評価",
                ],
            ),
            6,
        )
        frame_counts["roles"] = _write_still(
            writer,
            _card(
                "Goal-conditioned AIと従来制御の役割",
                [
                    "視覚AI: 指示された対象物を選び、把持XYを推定",
                    "下位制御: 搬送先IDから固定目標を選び、IK・状態機械で実行",
                    "ニューラルネットワークは関節軌道を生成しない",
                    "強調:推論入力に真値・Segmentation・scene IDは不使用",
                ],
                ROOT / "docs" / "assets" / "final" / "system_architecture_ja.png",
            ),
            8,
        )
        frame_counts["goal_conditioning"] = _write_still(
            writer, _goal_conditioning_card(improved_results, camera), 12
        )
        frame_counts["prediction"] = _write_still(
            writer, _prediction_card(improved_log, camera), 8
        )
        frame_counts["improved_action"] = _write_action(
            writer, action_reader, improved_log, camera, 12
        )
        frame_counts["comparison"] = _write_comparison(
            writer, old_reader, improved_reader, old_log, improved_log, camera, 12
        )
        frame_counts["data_scaling"] = _write_still(
            writer,
            _card(
                "学習データ量と共通Test平均位置誤差",
                [
                    "Train scenes: 80 → 500 → 2,000",
                    "平均誤差: 12.48 → 3.40 → 2.14 mm",
                    "モデル構造・学習方法は固定",
                ],
                ROOT / "docs" / "assets" / "final" / "data_scale_vs_error.png",
            ),
            10,
        )
        frame_counts["physical_result"] = _write_still(
            writer,
            _card(
                "同一24配置の指示達成率",
                [
                    "旧モデル: 15/24 = 62.5%",
                    "改善モデル: 24/24 = 100%",
                    "同じseed・物体配置・対象色・搬送先で比較",
                    "強調:100%は評価した24配置に限定",
                ],
                ROOT / "docs" / "assets" / "final" / "physical_success_comparison.png",
            ),
            10,
        )
        frame_counts["conclusion"] = _write_still(
            writer,
            _card(
                "結論",
                [
                    "Goal conditioningで同一画像から指示対象を選択",
                    "データ増量で把持位置誤差を12.48 → 2.14 mmへ改善",
                    "同一24配置の物理成功率を62.5% → 100%へ改善",
                    "固定カメラ・3色立方体・構造化指示・開ループ評価",
                ],
            ),
            10,
        )
    finally:
        writer.release()
        old_reader.close()
        improved_reader.close()
        action_reader.close()
    _transcode()
    total_frames = sum(frame_counts.values())
    manifest = {
        "output": str(OUTPUT.resolve()),
        "duration_seconds": total_frames / FPS,
        "resolution": [WIDTH, HEIGHT],
        "fps": FPS,
        "silent": True,
        "frame_counts": frame_counts,
        "comparison": {
            "same_seed": old_log["seed"] == improved_log["seed"],
            "same_instruction": True,
            "instruction": "red cube to left bin",
            "old_prediction_error_mm": old_log["prediction_error_mm"],
            "improved_prediction_error_mm": improved_log["prediction_error_mm"],
            "old_success": old_log["placement_success"],
            "improved_success": improved_log["placement_success"],
        },
        "goal_conditioning": improved_results["instruction_sensitivity_check"],
        "sources": [
            str((RECORDINGS / "old_trial01.json").resolve()),
            str((RECORDINGS / "improved_trial01.json").resolve()),
            str(
                (
                    ROOT
                    / "outputs"
                    / "phase7"
                    / "physical"
                    / "improved"
                    / "phase5_results.json"
                ).resolve()
            ),
        ],
        "simulation_rerun": False,
        "model_or_controller_changed": False,
    }
    (RECORDINGS / "enhanced_video_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return OUTPUT


if __name__ == "__main__":
    print(build())
