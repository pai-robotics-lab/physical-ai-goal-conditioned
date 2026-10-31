# フェーズ4 Goal-conditioned視覚AI 学習・オフライン評価報告

実施日: 2026-10-31

## 結論

フェーズ4は合格した。RGB画像、対象色ID、搬送先IDだけを入力する小型モデルをGTX 1050で
学習し、未知Test 10シーン・60サンプルで平均XY誤差15.250 mmを得た。暫定目標20 mm以下を
達成した。TestはValidationによるモデル固定後に1回だけ評価し、調整には使用していない。

## データと入力制約

- フェーズ3の100シーン、600サンプルをそのまま使用した。
- Train/Validation/Testは80/10/10シーン、480/60/60サンプル。
- 同一シーン・画像の異なる指示は同じsplitに維持した。
- 入力は160 x 120 RGB、対象色ID、搬送先IDのみ。
- 物体真値、Segmentation、scene ID、把持ラベルは推論入力に使用しない。
- RGB mean/stdとXY min/maxはTrain splitだけから算出した。

Train由来XY範囲は `x=[0.404241, 0.699629] m`、`y=[-0.217919, 0.078037] m` で、
各軸を `[-1, 1]` に正規化した。RGB mean/std、画像サイズ、XY範囲は
`models/phase4/normalization.json`に保存した。

## モデル構造

本モデルは49,366 trainable parametersの小型ネットワークである。

1. RGBをTrain mean/stdで正規化する。
2. 対象色IDを使い、RGB内で対象色チャンネルが他2色より強い連続コントラストマップを計算する。
3. RGBとコントラストマップの4チャンネルを小型CNNへ入力し、30 x 40空間特徴を得る。
4. 色Embeddingをqueryとする空間Attention、色マップ重心、局所特徴、全体特徴を抽出する。
5. それらと対象色Embedding・搬送先Embeddingを結合する。
6. MLPとtanhで正規化XYを回帰し、Train範囲へ逆正規化する。

色コントラストマップはRGBと指示色IDだけから計算でき、Segmentationや真値ではない。学習と
推論で同じ`goal_vision.py`の前処理・モデルを使う。

| モデル | 入力 | パラメータ数 |
|---|---|---:|
| full | RGB + 対象色ID + 搬送先ID | 49,366 |
| instruction_only | 対象色ID + 搬送先ID | 10,082 |
| image_only | RGB | 1,114,266 |

## 学習条件

- 環境: `.venv-train`、PyTorch 2.8.0+cu126、GTX 1050 4GB
- optimizer: AdamW、learning rate 0.001、weight decay 0.0001
- loss: Smooth L1、beta 0.1
- batch size: 32
- 最大epoch: 100
- Early Stopping: Validation Loss、patience 15
- LR scheduler: ReduceLROnPlateau、patience 5、factor 0.5
- seed: 4100
- 重み: PyTorch state_dictのみを保存

| モデル | 最良Epoch | 実行Epoch | 学習時間 (s) | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|
| full | 47 | 62 | 30.819 | 201.8 MiB | 304.0 MiB |
| instruction_only | 4 | 19 | 3.420 | 84.4 MiB | 106.0 MiB |
| image_only | 3 | 18 | 8.536 | 216.4 MiB | 258.0 MiB |

3モデルの合計学習時間は42.775秒だった。

## ValidationとTest

### 本モデル

| 指標 | Validation | Test |
|---|---:|---:|
| 平均誤差 (mm) | 18.844 | 15.250 |
| 中央値 (mm) | 18.277 | 13.446 |
| P90 (mm) | 29.497 | 28.665 |
| 10 mm以内 | 13.3% | 33.3% |
| 20 mm以内 | 56.7% | 71.7% |

### Test色別・搬送先別平均誤差

| 区分 | 平均誤差 (mm) |
|---|---:|
| red_cube | 14.301 |
| blue_cube | 14.187 |
| green_cube | 17.263 |
| left_bin | 15.174 |
| right_bin | 15.326 |

搬送先別の差は0.152 mmで、XYと無関係な搬送先IDによる偏りは小さい。

## ベースライン比較

| モデル | Validation平均 (mm) | Test平均 (mm) | Test中央値 (mm) | Test P90 (mm) |
|---|---:|---:|---:|---:|
| full | 18.844 | 15.250 | 13.446 | 28.665 |
| instruction_only | 127.755 | 126.526 | 135.817 | 182.831 |
| image_only | 125.408 | 127.825 | 133.813 | 184.011 |

指示のみではシーンごとのランダム位置が分からず、画像のみでは同一画像に3つある正解のどれを
選ぶか分からない。本モデルが両者を結合することで約112 mm以上改善した。

## 同一画像で色指示を変更した評価

Testの同一画像・同一搬送先における3色間の平均pairwise変化は次のとおり。

- 真値: 201.765 mm
- full予測: 196.967 mm
- full変化ベクトル誤差: 18.203 mm
- instruction_only予測: 33.074 mm
- image_only予測: 0.000 mm

fullモデルは同じRGBでも色指示に応じて別の物体位置へ予測を移動できた。画像のみモデルは
指示を受け取らないため、3色で同じ点を返した。

## 過学習

本モデルの最良Epoch 47ではTrainとValidation Lossが近く、その後Validationは約0.043～0.045、
Trainは約0.034～0.040で推移した。小さなgeneralization gapはあるが、Validationが継続悪化する
深刻な過学習は見られない。Early StoppingによりEpoch 62で終了した。

## 可視化

- `outputs/phase4/visualizations/loss_curves.png`
- `outputs/phase4/visualizations/test_error_distribution.png`
- `outputs/phase4/visualizations/baseline_comparison.png`
- `outputs/phase4/visualizations/prediction_examples/`
- `outputs/phase4/visualizations/same_image_color_conditioning.png`

予測・真値の画像重畳には、フェーズ3に保存したcamera position、lookat、up、vertical FOV、
解像度からpinhole外部・内部行列相当を計算した。単体テストで既知ワールド点の投影値が
Genesis segmentation中心と約1～2 pixel以内で一致することを確認した。

## 成果物

- モデル: `src/pai_goal_conditioned/goal_vision.py`
- 学習・評価: `src/pai_goal_conditioned/phase4_train.py`
- 設定: `configs/phase4_training.json`
- テスト: `tests/test_goal_vision.py`
- state_dict: `models/phase4/*_best_state_dict.pt`
- 正規化: `models/phase4/normalization.json`
- 学習記録: `outputs/phase4/training_summary.json`、`histories/`
- 評価: `outputs/phase4/evaluation_summary.json`、`predictions/`
- 比較: `outputs/phase4/baseline_comparison.csv`
- 可視化: `outputs/phase4/visualizations/`

`models/`と`outputs/`はGit管理外である。学習済み重みや大容量生成物はGitへ追加していない。

## 判定

未知シーンTest平均15.250 mmで暫定20 mm目標を満たしたため、フェーズ5以降へ進行可能と
判断する。ただし今回はフェーズ4で終了し、Panda実行、オンライン統合、閉ループ制御には
着手しない。

## 最終品質確認

```powershell
.\.venv-sim\python.exe -m ruff check src tests
.\.venv-sim\python.exe -m pytest -q
.\.venv-sim\python.exe -m pai_goal_conditioned.check_env --mode sim
.\.venv-sim\python.exe -m pai_goal_conditioned.smoke_test
.\.venv-train\python.exe -m pai_goal_conditioned.check_env --mode train
```

- Ruff: All checks passed
- pytest: 12 passed
- sim環境確認: PASS
- CPU smoke test: PASS（`z=1.000 -> 0.544 m`、11.771秒）
- train環境確認: PASS（PyTorch 2.8.0+cu126、GTX 1050、CUDA Conv2d）
