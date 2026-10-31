# フェーズ7 Goal-conditioned視覚AI精度改善・物理性能比較報告

実施日: 2026-10-31

## 結論

フェーズ4と同じ49,366パラメータ構造・学習方法を保ち、Train scene数だけを
80、500、2,000へ増やして比較した。共通未使用Test 250シーン・1,500サンプルで、
平均XY誤差は12.478、3.402、2.137 mmとなった。Train 2,000モデルは目標10 mm以下を
達成した。

旧フェーズ4モデルとTrain 2,000モデルを、同じ未使用24配置で物理評価した。旧モデルは
15/24（62.5%）、改善モデルは24/24（100%）の指示達成となり、目標90%程度を達成した。
旧モデルが把持失敗した9条件はすべて改善モデルで成功し、逆方向の退行はなかった。

## データセット

フェーズ3生成器を設定変更なしの物理・カメラ条件で再利用した。新規seed 70000以降で
2,500シーンを生成し、2,000 Train、250 Validation、250 Testへシーン単位で分割した。

- シーン: 2,500
- RGB画像: 2,500
- ラベル: 15,000（3物体 x 2搬送先）
- Train/Validation/Test: 2,000/250/250シーン、12,000/1,500/1,500サンプル
- 全ラベル有効、split漏洩0
- 最小可視画素367
- 無効初回配置40件を理由・seed付きで除外して再生成
- 既存フェーズ4～6評価seedとの重複0
- 生成時間1,048.561秒、速度2.384 scenes/s、ディスク約92.8 MB

80と500の比較条件は2,000 Train sceneから固定seed 7177で選んだ入れ子部分集合であり、
`80 subset 500 subset 2000`を満たす。scene IDは
`outputs/phase7/offline/train_scene_subsets.json`へ保存した。Validation/Testは3モデルで完全に
共通である。

## 学習条件

- モデル: フェーズ4 Fullモデル、49,366パラメータ
- 画像: 160 x 120 RGB
- 指示: 対象色ID + 搬送先ID
- optimizer: AdamW、learning rate 0.001、weight decay 0.0001
- loss: Smooth L1、beta 0.1
- batch size: 32
- max epoch: 100
- Early Stopping: Validation Loss、patience 15
- seed: 7100、部分集合seed: 7177
- GTX 1050 4GB、PyTorch 2.8.0+cu126

モデル構造、optimizer、loss、batch size、Early Stoppingは固定し、データ量だけを変更した。
正規化は各Train部分集合だけから独立に計算した。Testはモデル固定後に別ステージで1回だけ
評価した。

| Train scenes | Samples | Best epoch | Epochs | 学習時間 (s) | Peak allocated | Peak reserved |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 480 | 87 | 100 | 129.098 | 201.8 MiB | 304.0 MiB |
| 500 | 3,000 | 97 | 100 | 368.327 | 201.8 MiB | 304.0 MiB |
| 2,000 | 12,000 | 49 | 64 | 791.547 | 202.0 MiB | 304.0 MiB |

合計学習時間は1,288.973秒だった。

## 共通Testオフライン評価

| モデル | 平均 (mm) | 中央値 | P90 | 10 mm以内 | 15 mm以内 | 20 mm以内 |
|---|---:|---:|---:|---:|---:|---:|
| 旧phase4 80 | 14.034 | 13.230 | 24.632 | 34.1% | 59.9% | 78.9% |
| 新規80 | 12.478 | 11.799 | 22.157 | 40.4% | 66.9% | 85.7% |
| 新規500 | 3.402 | 3.014 | 6.096 | 99.3% | 99.9% | 100% |
| 新規2,000 | 2.137 | 1.976 | 3.666 | 100% | 100% | 100% |

旧モデルはフェーズ4の既存state_dictを新規共通Testへ適用した参考値である。データ量だけの主比較は
新規80/500/2,000の3モデルである。

### X/Y方向別平均絶対誤差

| モデル | X・直交方向 (mm) | Y・開閉方向 (mm) |
|---|---:|---:|
| 旧phase4 80 | 9.134 | 8.887 |
| 新規80 | 8.096 | 7.821 |
| 新規500 | 1.807 | 2.441 |
| 新規2,000 | 1.274 | 1.456 |

Train 2,000のX/Y P90絶対誤差は2.747/2.906 mmで、フェーズ6の代表配置で観測した
±10 mm前後の把持許容範囲より十分小さい。

### Train 2,000の色別平均誤差

| 色 | 平均誤差 (mm) |
|---|---:|
| red | 1.955 |
| blue | 2.136 |
| green | 2.321 |

## ペア物理操作評価

seed 90000～90023の24配置を使用した。6種類の対象色・搬送先組合せを各4回含む。
2モデル間の全物体初期位置差は最大0.0 mmであり、推論モデル以外の条件は同一だった。

| 指標 | 旧phase4 | 改善2,000 |
|---|---:|---:|
| 正しい対象把持 | 15/24 | 24/24 |
| 持ち上げ | 15/24 | 24/24 |
| 搬送 | 15/24 | 24/24 |
| 配置 | 15/24 | 24/24 |
| 指示達成 | 15/24 (62.5%) | 24/24 (100%) |
| 平均推定誤差 | 13.059 mm | 2.190 mm |
| 中央値 | 13.832 mm | 1.794 mm |
| P90 | 22.024 mm | 3.393 mm |
| X平均絶対誤差 | 8.124 mm | 1.288 mm |
| Y平均絶対誤差 | 8.481 mm | 1.470 mm |
| 成功時平均配置誤差 | 9.773 mm | 8.870 mm |
| 成功時最大配置誤差 | 29.785 mm | 32.609 mm |

旧モデルの9失敗はすべて把持段階だった。旧成功群の平均推定誤差は9.267 mm、失敗群は
19.378 mmだった。失敗群のX/Y平均絶対誤差は11.836/12.111 mmで、成功群の
5.897/6.302 mmより大きい。

改善モデルは旧失敗9件をすべて成功へ変え、旧成功からの退行は0件だった。色別では旧モデルが
red 4/8、blue 7/8、green 4/8、改善モデルは全色8/8だった。成功後の配置精度は両モデルで
同程度であり、成功率向上の主因は把持XY推定の改善である。

平均推論時間は旧0.0117秒、改善0.0106秒、平均総試行時間は12.537/12.347秒で、モデル構造を
維持したため推論・操作速度の悪化はなかった。

## 可視化

- `outputs/phase7/offline/visualizations/data_scale_vs_error.png`
- `outputs/phase7/physical_comparison/physical_success_comparison.png`
- `outputs/phase7/physical/old/overlays/`、`actions/`
- `outputs/phase7/physical/improved/overlays/`、`actions/`

同じ試行1で旧モデルは把持失敗、改善モデルは赤立方体の持ち上げに成功した。旧誤差13.945 mm、
改善誤差3.440 mmである。

## 成果物

- データ設定: `configs/phase7_dataset.json`
- 学習設定: `configs/phase7_training.json`
- 物理設定: `configs/phase7_physical_old.json`、`phase7_physical_improved.json`
- 学習・評価: `src/pai_goal_conditioned/phase7_train.py`
- 物理比較: `src/pai_goal_conditioned/phase7_compare.py`
- テスト: `tests/test_phase7_train.py`、`tests/test_phase7_compare.py`
- データ: `datasets/phase7/`
- モデル: `models/phase7/train_80/`、`train_500/`、`train_2000/`
- オフライン評価: `outputs/phase7/offline/`
- 物理評価: `outputs/phase7/physical/`、`physical_comparison/`

## 判定

データ増量だけで共通Test平均誤差を12.478 mmから2.137 mmへ改善し、ペア物理成功率を
62.5%から100%へ改善した。位置誤差10 mm以下、物理成功率90%程度の両目標を達成したため、
フェーズ7は合格と判断できる。今回はフェーズ7で終了し、閉ループ制御、追加学習、LLM、
フェーズ8以降には進まない。

## 最終品質確認

- Ruff: All checks passed
- pytest: 25 passed
- sim環境: Python 3.10.22、PyTorch 2.8.0+cpu、Genesis 1.2.1、PASS
- train環境: PyTorch 2.8.0+cu126、GTX 1050、CUDA演算PASS
- CPU smoke test: PASS（`z=1.000 -> 0.544 m`、135.884秒）

