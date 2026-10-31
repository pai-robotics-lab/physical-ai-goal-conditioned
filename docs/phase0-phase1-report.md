# フェーズ0・1 検証報告

実施日: 2026-10-30～2026-10-31

## 結論

フェーズ0とフェーズ1は合格した。Windows 10上でGenesis 1.2.1のCPU物理、GTX 1050上で
CUDA版PyTorchが動作し、Franka Pandaは5つの目標姿勢へIKと関節位置制御で到達した。
固定座標Pick and Placeを扱うフェーズ2へ進める。

## 環境

| 項目 | シミュレーション環境 | 学習環境 |
|---|---|---|
| Prefix | `.venv-sim` | `.venv-train` |
| Python | 3.10.22 | 3.10.22 |
| PyTorch | 2.8.0+cpu | 2.8.0+cu126 |
| torchvision | 使用しない | 0.23.0+cu126 |
| Genesis | 1.2.1 | 実行に使用しない |
| バックエンド | CPU | CUDA 12.6 |
| GPU | 不使用 | NVIDIA GeForce GTX 1050、4GB |

学習環境は、約2.9GBのCUDA wheel取得が回線断で長時間化したため、既存の動作確認済み
GPU環境を変更せず、Conda cloneで新しい独立prefixとして作成した。clone元に含まれる
Genesis 0.3.14は学習環境の実行経路では使用しない。通常の新規構築手順は
`setup-train.ps1`に記録した。

## フェーズ0

### CPUシミュレーション

球の自由落下を30 step、`dt=0.01 s`で実行した。

- 初期高さ: 1.000 m
- 最終高さ: 0.544 m
- 結果: PASS
- 初回JITコンパイル込み実行時間: 172.418秒
- キャッシュ後の再実行時間: 13.268秒

2回目以降はプロジェクト内`.cache/`を再利用する。今回の再実行では約13秒まで短縮した。
キャッシュはGit管理しない。

### CUDA学習環境

GTX 1050上で3x4行列と4x2行列の積、および8枚の32x32 RGB画像に対するConv2dを
実行し、GPU同期後に結果を確認した。

- `torch.cuda.is_available()`: `True`
- GPU: NVIDIA GeForce GTX 1050
- CUDA runtime: 12.6
- Compute Capability: 6.1
- VRAM: 4,294,836,224 bytes
- 演算結果の総和: 522.0
- Conv2d出力shape: `(8, 8, 30, 30)`
- 結果: PASS

## 公式サンプルとの互換性

Genesis公式mainブランチの`examples/rigid/ik_franka.py`が使用する主要APIを、インストールした
Genesis 1.2.1で照合した。

- `gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")`
- `entity.get_link("hand")`
- `entity.inverse_kinematics(..., return_error=True)`
- `entity.control_dofs_position(...)`
- `scene.step()`

これらは1.2.1に存在し、今回の検証で実動した。mainサンプルはIK解の可視化に
`set_qpos()`を使うが、公式コメントに従い、実制御検証ではIK解を関節目標として渡し、
`control_dofs_position()`と`scene.step()`で追従させた。

## フェーズ1

共通目標姿勢は下向き姿勢のwxyz quaternion `(0, 1, 0, 0)`とした。合格基準は位置誤差
10 mm以下、姿勢誤差3度以下とした。

| 目標 | 位置 (m) | 位置誤差 (mm) | 姿勢誤差 (deg) | IK時間 (s) | 制御時間 (s) | 結果 |
|---|---|---:|---:|---:|---:|---|
| front_low | (0.45, 0.00, 0.30) | 5.081 | 0.675 | 6.757 | 2.074 | PASS |
| left_low | (0.48, 0.14, 0.34) | 5.801 | 0.734 | 0.004 | 1.995 | PASS |
| right_low | (0.48, -0.14, 0.34) | 5.843 | 0.740 | 0.003 | 2.002 | PASS |
| left_high | (0.40, 0.12, 0.46) | 5.011 | 0.635 | 0.003 | 2.004 | PASS |
| right_high | (0.40, -0.12, 0.46) | 4.584 | 0.621 | 0.003 | 1.990 | PASS |

画像付き5姿勢検証の総時間は75.704秒だった。各姿勢を300 step制御し、最終姿勢のPNGを
`outputs/phase1/images/`へ保存した。

## 警告と問題点

Panda MJCF読込時に次の警告を確認した。

- 2本のfinger tendonをjoint actuatorとして近似する。
- MJCFのneutral `qpos0`が一部関節制限を超える。
- constraint solverのtime constantが0.005から0.01へ調整される。
- neutral姿勢で自己衝突するgeometry pairがフィルタされる。

今回の5姿勢では制御失敗や数値発散はなく、すべて合格した。ただしフェーズ2では、指関節の
tendon近似と接触安定性が把持成功率へ影響する可能性があるため、最重要リスクとして扱う。

## フェーズ2への判定

進行可。次の順序で固定座標Pick and Placeを検証する。

1. 指を開いた状態でpre-grasp姿勢へ移動する。
2. 垂直に下降する。
3. 位置制御で指を閉じる。
4. 物体を持ち上げ、物体高さで把持成功を判定する。
5. 固定された箱位置へ搬送し、開放する。
6. 10試行中8回以上をGo条件とする。

AI、カメラ認識、ランダム配置はフェーズ2の合格後に開始する。

