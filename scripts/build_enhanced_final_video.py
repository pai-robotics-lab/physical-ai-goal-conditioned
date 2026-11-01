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
    "transport": "搬送先へ搬送",
    "place": "下降・配置",
    "open": "グリッパー開・解放",
    "retreat": "退避",
    "settle": "静止確認",
}

TARGET_STYLES = {
    "red_cube": {"ja": "赤", "en": "red cube", "color": RED},
    "blue_cube": {"ja": "青", "en": "blue cube", "color": "#5794ff"},
    "green_cube": {"ja": "緑", "en": "green cube", "color": GREEN},
}
DESTINATION_STYLES = {
    "left_bin": {"ja": "左搬送先", "en": "to left bin", "xy": [0.42, 0.22]},
    "right_bin": {"ja": "右搬送先", "en": "to right bin", "xy": [0.68, 0.22]},
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
        "Vision-Based Panda Grasping and Transport",
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


def _target_bbox(rgb: np.ndarray, target_class: str) -> tuple[int, int, int, int] | None:
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    if target_class == "red_cube":
        mask = (red > 100) & (red - green > 45) & (red - blue > 45) & (green < 130)
    elif target_class == "blue_cube":
        mask = (blue > 90) & (blue - red > 35) & (blue - green > 35)
    elif target_class == "green_cube":
        mask = (green > 80) & (green - red > 30) & (green - blue > 20)
    else:
        raise ValueError(f"Unsupported target class: {target_class}")
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


def _target_color_condition_card(
    improved_results: dict[str, Any], camera: dict[str, Any]
) -> Image.Image:
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
    source = source.resize((720, 540), Image.Resampling.LANCZOS)
    image = _base("AI入力と推定結果", "実行ログ: seed 90000 / improved Train 2,000 model")
    image.paste(source, (45, 125))
    draw = ImageDraw.Draw(image)
    scale_x, scale_y = 720 / 640, 540 / 480
    predicted = project_world_to_pixel([*log["predicted_xy_m"], 0.224945], camera)
    truth = project_world_to_pixel([*log["truth_xy_m"], 0.224945], camera)
    destination = project_world_to_pixel([0.42, 0.22, 0.225], camera)
    predicted_display = (45 + predicted[0] * scale_x, 125 + predicted[1] * scale_y)
    truth_display = (45 + truth[0] * scale_x, 125 + truth[1] * scale_y)
    destination_display = (45 + destination[0] * scale_x, 125 + destination[1] * scale_y)
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
    trial_number: int,
    trial_count: int,
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    video = Image.fromarray(rgb).resize((920, 690), Image.Resampling.LANCZOS)
    image.paste(video, (0, 15))
    draw = ImageDraw.Draw(image)
    stage = _stage_at(source_index, log["stage_frames"])
    target = TARGET_STYLES[log["target_class"]]
    destination_style = DESTINATION_STYLES[log["destination_id"]]
    bbox = _target_bbox(rgb, log["target_class"])
    if bbox:
        x1, y1, x2, y2 = bbox
        scaled = (x1 * 920 / 640, 15 + y1 * 690 / 480, x2 * 920 / 640, 15 + y2 * 690 / 480)
        draw.rectangle(scaled, outline=target["color"], width=5)
        if stage in {"transport", "place", "open", "retreat", "settle"}:
            label_y = min(660, scaled[3] + 5)
        else:
            label_y = max(20, scaled[1] - 31)
        draw.text(
            (scaled[0], label_y),
            f"対象: {target['ja']}",
            font=_font(21, True),
            fill=target["color"],
        )
    if stage in {"initial", "move_above", "descend", "close"}:
        point = project_world_to_pixel([*log["predicted_xy_m"], 0.224945], camera)
        _draw_cross(
            draw,
            (point[0] * 920 / 640, 15 + point[1] * 690 / 480),
            target["color"],
            14,
        )
    destination_xy = destination_style["xy"]
    destination = project_world_to_pixel([*destination_xy, 0.225], camera)
    _draw_destination(
        draw,
        (destination[0] * 920 / 640, 15 + destination[1] * 690 / 480),
        destination_style["ja"],
    )
    draw.rectangle((920, 0, WIDTH, HEIGHT), fill="#0c2030")
    draw.text(
        (945, 24),
        f"独立試行 {trial_number}/{trial_count}",
        font=_font(24, True),
        fill=GREEN,
    )
    draw.text((945, 62), f"seed {log['seed']} / 別シーン初期化", font=_font(17), fill=MUTED)
    draw.line((945, 98, 1250, 98), fill="#35556a", width=2)
    draw.text((945, 118), target["en"], font=_font(27, True), fill=target["color"])
    draw.text((945, 155), destination_style["en"], font=_font(27, True), fill=PURPLE)
    draw.line((945, 198, 1250, 198), fill="#35556a", width=2)
    draw.text((945, 218), "CNN推定把持XY", font=_font(22, True), fill=WHITE)
    draw.text((945, 255), f"X {log['predicted_xy_m'][0]:.6f} m", font=_font(20), fill=CYAN)
    draw.text((945, 288), f"Y {log['predicted_xy_m'][1]:.6f} m", font=_font(20), fill=CYAN)
    draw.text((945, 333), "現在の動作状態", font=_font(22, True), fill=WHITE)
    draw.rounded_rectangle((940, 373, 1260, 438), radius=12, fill="#17425a")
    draw.text((960, 388), STAGE_LABELS[stage], font=_font(25, True), fill=GREEN)
    draw.text((945, 475), "制御: 状態機械 + IK", font=_font(20), fill=MUTED)
    draw.text((945, 510), "真値補正・再試行なし", font=_font(20), fill=MUTED)
    if stage == "settle":
        draw.rounded_rectangle((940, 570, 1260, 675), radius=12, fill="#145b3a")
        draw.text((960, 590), "結果: 配置成功", font=_font(28, True), fill=WHITE)
        draw.text(
            (960, 632),
            f"最終誤差 {log['final_placement_error_mm']:.2f} mm",
            font=_font(21),
            fill=GREEN,
        )
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
        bbox = _target_bbox(rgb, "red_cube")
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
    trial_number: int,
    trial_count: int,
) -> int:
    count = round(duration * FPS)
    max_index = log["frame_count"] - 1
    for output_index in range(count):
        source_index = min(int(output_index / FPS * log["fps"]), max_index)
        frame = _annotate_action(
            reader.get(source_index),
            log,
            camera,
            source_index,
            trial_number,
            trial_count,
        )
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
    phase3 = _load_json(ROOT / "configs" / "phase3_dataset.json")
    camera = phase3["camera"]
    trial_specs = [
        ("improved_trial01", 90000, "red_cube", "left_bin"),
        ("montage_seed90007_red_right", 90007, "red_cube", "right_bin"),
        ("montage_seed90014_blue_left", 90014, "blue_cube", "left_bin"),
        ("montage_seed90023_green_right", 90023, "green_cube", "right_bin"),
    ]
    trials: list[dict[str, Any]] = []
    for basename, seed, target_class, destination_id in trial_specs:
        log_path = RECORDINGS / f"{basename}.json"
        video_path = RECORDINGS / f"{basename}.mp4"
        log = _load_json(log_path)
        actual = (log["seed"], log["target_class"], log["destination_id"])
        expected = (seed, target_class, destination_id)
        if actual != expected:
            raise ValueError(f"Unexpected trial metadata in {log_path}: {actual} != {expected}")
        if not log["placement_success"]:
            raise ValueError(f"Montage source did not pass placement evaluation: {log_path}")
        if log["control_change"] or log["truth_used_for_control"]:
            raise ValueError(f"Montage source changed control or used truth: {log_path}")
        trials.append({"log": log, "log_path": log_path, "video_path": video_path})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(TEMP_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {TEMP_VIDEO}")
    frame_counts: dict[str, int] = {}
    readers = [SequentialReader(trial["video_path"]) for trial in trials]
    try:
        frame_counts["title"] = _write_still(
            writer,
            _card(
                "視覚AIを用いたPandaロボットアームの物体把持・搬送",
                [
                    "CNN: RGB画像 + 構造化ID → 把持位置XY",
                    "従来型制御: 搬送先選択・状態機械・IK・把持力",
                    "追加評価: 学習データ量と位置精度・物理成功率を比較",
                ],
            ),
            5,
        )
        frame_counts["roles"] = _write_still(
            writer,
            _card(
                "視覚AIによる位置推定と従来型制御の役割",
                [
                    "視覚AI: 対象色に応じた物体選択と把持位置XY推定",
                    "従来型制御: 搬送先選択・状態機械・IK・把持力",
                    "CNNは関節軌道・把持力・動作タイミングを生成しない",
                    "強調:推論入力に真値・Segmentation・scene IDは不使用",
                ],
                ROOT / "docs" / "assets" / "final" / "system_architecture_ja.png",
            ),
            6,
        )
        frame_counts["independent_trial_notice"] = _write_still(
            writer,
            _card(
                "4つの独立試行を順に表示",
                [
                    "各試行は別seedでシーンとPandaを初期化",
                    "90000: 赤→左 / 90007: 赤→右",
                    "90014: 青→左 / 90023: 緑→右",
                    "強調:同一シーン内の連続仕分けではない",
                ],
            ),
            4,
        )
        for index, (reader, trial) in enumerate(zip(readers, trials), start=1):
            log = trial["log"]
            key = f"independent_trial_{index}_seed_{log['seed']}"
            frame_counts[key] = _write_action(
                writer,
                reader,
                log,
                camera,
                log["duration_seconds"],
                index,
                len(trials),
            )
        frame_counts["data_scaling"] = _write_still(
            writer,
            _card(
                "位置推定精度: 共通Testの平均XY誤差",
                [
                    "オフライン回帰指標（小さいほど正確）",
                    "Train scenes: 80 → 500 → 2,000",
                    "平均誤差: 12.48 → 3.40 → 2.14 mm",
                    "モデル構造・制御器は固定",
                ],
                ROOT / "docs" / "assets" / "final" / "data_scale_vs_error.png",
            ),
            7,
        )
        frame_counts["physical_result"] = _write_still(
            writer,
            _card(
                "物理成功率: 把持から配置までの結果",
                [
                    "位置誤差とは異なるend-to-end物理指標",
                    "旧モデル: 15/24 = 62.5%",
                    "改善モデル: 24/24 = 100%",
                    "強調:100%は評価した24配置に限定",
                ],
                ROOT / "docs" / "assets" / "final" / "physical_success_comparison.png",
            ),
            7,
        )
        frame_counts["conclusion"] = _write_still(
            writer,
            _card(
                "結論",
                [
                    "4独立試行で3色・左右搬送先・異なる初期配置を実演",
                    "CNNの出力は把持位置XY / 動作は従来型制御",
                    "位置推定誤差と物理成功率は別の評価指標",
                    "強調:固定カメラ・構造化指示・開ループ評価",
                ],
            ),
            7,
        )
    finally:
        writer.release()
        for reader in readers:
            reader.close()
    _transcode()
    total_frames = sum(frame_counts.values())
    manifest = {
        "output": str(OUTPUT.resolve()),
        "duration_seconds": total_frames / FPS,
        "resolution": [WIDTH, HEIGHT],
        "fps": FPS,
        "silent": True,
        "frame_counts": frame_counts,
        "montage": {
            "trial_count": len(trials),
            "each_trial_is_independent_scene": True,
            "continuous_same_scene_sorting": False,
            "all_recordings_passed": all(trial["log"]["placement_success"] for trial in trials),
            "trials": [trial["log"] for trial in trials],
        },
        "sources": [str(trial["log_path"].resolve()) for trial in trials]
        + [
            str((ROOT / "outputs/phase7/physical/improved/phase5_results.json").resolve()),
        ],
        "backup_88_second_video": str(
            (ROOT / "videos/physical-ai-goal-conditioned-final-demo-88s-backup.mp4").resolve()
        ),
        "additional_recordings_created": 3,
        "model_or_controller_changed": False,
    }
    (RECORDINGS / "enhanced_video_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return OUTPUT


if __name__ == "__main__":
    print(build())
