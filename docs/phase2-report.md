# フェーズ2 固定座標Pick and Place 検証報告

実施日: 2026-10-31

## 結論

フェーズ2は合格した。Genesis 1.2.1のCPU物理で、Franka Pandaが50 mm立方体を二指の
物理接触で把持し、持ち上げ、280 mm搬送し、目標位置へ解放した。独立初期化10試行の
全試行で全工程に成功し、8/10の合格基準を満たした。フェーズ3以降は未着手である。

## 実装

- 状態機械: 初期姿勢、対象上方、把持高さ、指閉、持ち上げ、搬送、配置下降、指開、退避、静置。
- 腕: フェーズ1と同じ下向き姿勢 `(0, 1, 0, 0)` のIKと7関節位置制御。
- 指: 開時は各指0.04 mの位置制御、閉時は各指-1.0 Nの力制御。
- 軌道: 各状態間の関節目標をsmoothstep補間し、閉把持中は毎step指力を再指令。
- 接触: Panda、立方体、テーブルの摩擦係数を1.0とした。立方体密度は500 kg/m³。
- 独立性: 試行ごとに別Pythonプロセスを起動し、GenesisとSceneを新規初期化した。
- 物体操作: 立方体はScene構築時に初期配置した後、位置・姿勢を直接変更していない。
  実行中の取得値は成功判定と記録だけに使った。

指制御はGenesis公式の
[`IK_motion_planning_grasp.py`](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/tutorials/IK_motion_planning_grasp.py)
および
[`franka_grasp_rigid_cube.py`](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/sap_coupling/franka_grasp_rigid_cube.py)
と同じ、腕の位置制御と指の力制御を分離する方法を採用した。

## 固定条件

| 項目 | 値 |
|---|---|
| バックエンド | Genesis 1.2.1 CPU |
| 物理step | 0.01 s、substeps 2 |
| テーブル天面 | z=0.200 m |
| 立方体サイズ | 0.050 x 0.050 x 0.050 m |
| 初期中心 | (0.550, -0.120, 0.225) m |
| 目標中心 | (0.550, 0.160, 0.225) m |
| 把持hand高さ | 0.335 m |
| 持上・搬送hand高さ | 0.530 m |
| 配置水平許容誤差 | 0.050 m |
| 静置時間 | 2.00 s、200 step |
| 乱数seed | 2400～2409 |

全設定は `configs/phase2_pick_place.json`、実行時スナップショットは
`outputs/phase2/phase2_config_snapshot.json` に保存した。

## 成功判定

- 把持: 閉指後も立方体が把持位置の水平35 mm以内にある。
- 持ち上げ: 立方体中心が支持高さより80 mm以上上昇する。
- 搬送: 持ち上げ状態を維持し、目標水平位置の60 mm以内へ到達する。
- 配置: 解放後、目標水平位置の50 mm以内かつ支持高さ誤差12 mm以内にある。
- 安定: 2秒静置後の並進速度0.025 m/s以下、角速度0.25 rad/s以下。

代表試行では、立方体中心zは把持前0.2247 mから持ち上げ後0.4194 mへ上昇した。
搬送終了時の中心は `(0.5553, 0.1555, 0.4232)` mであり、支持面上を滑らせた結果ではない。
解放・静置後は `(0.5532, 0.1588, 0.2249)` mとなった。

## 10試行結果

| 試行 | seed | 把持 | 持上 | 搬送 | 配置 | 静止 | 誤差 (mm) | 時間 (s) | 全工程 |
|---:|---:|---|---|---|---|---|---:|---:|---|
| 1 | 2400 | PASS | PASS | PASS | PASS | PASS | 3.412 | 35.141 | PASS |
| 2 | 2401 | PASS | PASS | PASS | PASS | PASS | 3.412 | 33.851 | PASS |
| 3 | 2402 | PASS | PASS | PASS | PASS | PASS | 3.412 | 182.805 | PASS |
| 4 | 2403 | PASS | PASS | PASS | PASS | PASS | 3.412 | 182.473 | PASS |
| 5 | 2404 | PASS | PASS | PASS | PASS | PASS | 3.412 | 181.590 | PASS |
| 6 | 2405 | PASS | PASS | PASS | PASS | PASS | 3.412 | 188.528 | PASS |
| 7 | 2406 | PASS | PASS | PASS | PASS | PASS | 3.412 | 187.389 | PASS |
| 8 | 2407 | PASS | PASS | PASS | PASS | PASS | 3.412 | 181.677 | PASS |
| 9 | 2408 | PASS | PASS | PASS | PASS | PASS | 3.412 | 184.115 | PASS |
| 10 | 2409 | PASS | PASS | PASS | PASS | PASS | 3.412 | 193.680 | PASS |

- 全工程成功: 10/10
- 失敗段階: なし
- 配置水平誤差: 最小・平均・最大すべて3.412 mm
- 静置後並進速度: 全試行 `9.08e-7 m/s`
- 静置後角速度: 全試行 `1.46e-5 rad/s`
- 1試行時間: 33.851～193.680秒、平均155.125秒

実行時間差は別プロセスでのGenesis/Quadrants初期化とJITキャッシュ挙動による。固定初期状態、
CPU物理、同一制御ではseed差による物理結果の差は観測されず、再現性は10/10だった。

## 成果物

- 実装: `src/pai_goal_conditioned/phase2_pick_place.py`
- 設定: `configs/phase2_pick_place.json`
- 集計: `outputs/phase2/phase2_results.json`、`phase2_results.csv`
- 試行別記録: `outputs/phase2/trials/trial_01.json`～`trial_10.json`
- 代表画像: `outputs/phase2/images/trial_01_lift.png`、`trial_01_final.png`
- 単体テスト: `tests/test_phase2_pick_place.py`

`outputs/`は再生成可能な評価生成物としてGit管理外である。

## 判定と次段階

フェーズ2は完了と判定する。フェーズ3へ進行可能な状態だが、今回の作業範囲に従い、
RGB画像・教師データ生成には着手しない。

## 最終品質確認

```powershell
.\.venv-sim\python.exe -m ruff check src tests
.\.venv-sim\python.exe -m pytest -q
.\.venv-sim\python.exe -m pai_goal_conditioned.check_env --mode sim
.\.venv-sim\python.exe -m pai_goal_conditioned.smoke_test
```

- Ruff: All checks passed
- pytest: 5 passed
- sim環境確認: PASS（Python 3.10.22、PyTorch 2.8.0+cpu、Genesis 1.2.1）
- CPU smoke test: PASS（`z=1.000 -> 0.544 m`、12.940秒）
