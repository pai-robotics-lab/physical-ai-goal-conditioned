# フェーズ5 Goal-conditioned AIオンライン自律操作 検証報告

実施日: 2026-10-31

## 結論

フェーズ4の49,366パラメータFullモデルをGenesis 1.2.1 CPU物理へ統合し、未知配置10試行を
実行した。RGB画像、対象色ID、搬送先IDだけから把持XYを推定し、真値補正や再試行なしで
7/10試行（70%）の指示達成に成功した。推定誤差は平均14.911 mm、中央値14.798 mm、
最大42.423 mmで、オフライン評価の暫定目標20 mm以下をオンライン試行でも平均値として
満たした。

3失敗はすべて把持段階であり、持ち上げ後の搬送・配置失敗はなかった。フェーズ2の固定座標・
単一物体10/10とは条件が異なるため、直接的な性能差とは扱わない。統合経路は成立したが、
開ループ操作の信頼性は改善余地があり、フェーズ6へ進む前に把持許容幅の改善を推奨する。

## 実装

- `.venv-sim`のCPU版PyTorchでstate_dictと`normalization.json`を読み込む。
- GenesisカメラのRGB配列へ、フェーズ4と共通のresize・RGB正規化を適用する。
- 推論APIはRGB、対象色ID、搬送先IDだけを受け取る。
- ワークスペース外の予測は物理動作前に棄却する。
- フェーズ2の上方接近、下降、指力制御、持ち上げ、搬送、配置、解放、退避を再利用する。
- 物体真値は推論後の独立評価と可視化だけに保存し、制御計画へ渡さない。
- 再認識、自動リトライ、真値補正、物体の直接移動は行わない。

カメラ、テーブル、50 mm立方体、色、Panda、物理設定はフェーズ3の教師データ生成条件と
一致させた。各試行の開始時だけseedに基づく配置へリセットし、動作開始後は物理接触だけで
操作した。

## 評価条件

- スモーク: seed 51000～51002、3試行
- 本評価: seed 52000～52009、10試行
- 対象色: red 4、blue 3、green 3試行
- 搬送先: left 5、right 5試行
- 推論: PyTorch CPU、単発開ループ
- 配置許容誤差: 50 mm
- 解放後静止確認: フェーズ2と同じ物理step・速度条件

スモークは3/3成功、平均推定誤差10.933 mmだったため、本評価へ進んだ。

## 10試行結果

| 試行 | 指示 | 推定誤差 (mm) | 正対象 | 持上 | 搬送 | 配置 | 指示達成 | 配置誤差 (mm) | 失敗段階 |
|---:|---|---:|---|---|---|---|---|---:|---|
| 1 | red -> left | 17.424 | NG | NG | NG | NG | NG | 483.196 | grasp |
| 2 | blue -> right | 5.692 | OK | OK | OK | OK | OK | 19.067 | - |
| 3 | green -> left | 20.352 | OK | OK | OK | OK | OK | 13.360 | - |
| 4 | red -> right | 13.349 | OK | OK | OK | OK | OK | 4.847 | - |
| 5 | blue -> left | 42.423 | NG | NG | NG | NG | NG | 320.169 | grasp |
| 6 | green -> right | 7.790 | OK | OK | OK | OK | OK | 4.091 | - |
| 7 | red -> left | 6.184 | OK | OK | OK | OK | OK | 8.118 | - |
| 8 | blue -> right | 2.649 | OK | OK | OK | OK | OK | 4.627 | - |
| 9 | green -> left | 16.246 | OK | OK | OK | OK | OK | 1.920 | - |
| 10 | red -> right | 17.003 | NG | NG | NG | NG | NG | 482.901 | grasp |

失敗試行の配置誤差は、対象が初期位置付近に残った結果であり、配置性能の値としては扱わない。
成功7試行の配置誤差は平均8.004 mm、中央値4.847 mm、最大19.067 mmで全件50 mm以内だった。

## 推定誤差と物理成否

| 区分 | 件数 | 平均誤差 (mm) | 中央値 (mm) | 範囲 (mm) |
|---|---:|---:|---:|---:|
| 成功 | 7 | 10.323 | 7.790 | 2.649～20.352 |
| 失敗 | 3 | 25.617 | 17.424 | 17.003～42.423 |

42.423 mmの最大誤差は明確な把持失敗となった。一方、20.352 mmでも成功し、17 mm前後で
2件失敗したため、ユークリッド距離だけでは成否を完全には決められない。二指グリッパーに
対する誤差方向、物体位置によるIK・接触条件、閉把持時の指と立方体の相対位置も影響する。

## 処理時間

- 平均推論時間: 0.0105秒
- 平均物理操作時間: 12.221秒
- 平均試行時間: 12.731秒
- 成功試行の平均: 11.595秒
- 失敗試行の平均: 15.380秒
- シーン構築・初回JITを含む全体: 278.960秒

試行1は初回制御コンパイルを含むため、他試行より長い。モデル推論時間は物理操作時間に対して
無視できる程度だった。

## 指示条件と入力分離

試行1の同じRGB、同じleft搬送先に対して色IDだけを変えた場合、予測点の色間距離は
120.378～223.009 mmで、3色すべて異なる把持点となった。

単体テストでは次を確認した。

- ファイル画像とGenesisのメモリ内RGBで前処理結果が一致する。
- phase4 state_dictをCPUで読み込み、有限なXYを返す。
- 推論APIに真値、Segmentation、scene ID引数が存在しない。
- 制御計画の把持XYがAI予測値と一致し、真値補正されない。
- ワークスペース外予測を棄却する。

## 可視化と機械可読結果

- `outputs/phase5/phase5_results.json`
- `outputs/phase5/phase5_results.csv`
- `outputs/phase5/images/`: 推論入力RGB
- `outputs/phase5/overlays/`: AI予測点と評価専用真値の重畳
- `outputs/phase5/actions/`: 持ち上げ時・最終状態
- `outputs/phase5/visualizations/prediction_error_vs_success.png`

重畳画像は赤がAI予測、緑が評価専用真値である。ワールド点から画像点への変換には
フェーズ3・4と同じカメラ外部・内部パラメータを使用した。

## 作成・変更ファイル

- `src/pai_goal_conditioned/phase5_online.py`
- `src/pai_goal_conditioned/goal_vision.py`
- `configs/phase5_online.json`
- `tests/test_phase5_online.py`
- `tests/test_goal_vision.py`
- `docs/phase5-report.md`
- `README.md`
- `AGENTS.md`
- `worklogs/2026-10-31.md`

## 問題点と改善候補

今回の範囲では追加学習や閉ループ回復を行っていない。次の改善候補は、学習データ増量、
画面端・ワークスペース端の重点サンプリング、予測不確実性の推定、把持方向別の許容誤差解析、
視覚による再認識と1回限定の補正である。これらはフェーズ5の評価条件には後付けしていない。

## 判定

Goal-conditioned AIからGenesis物理操作までの開ループ統合は成立した。ただし指示達成率70%で
3件の把持失敗が残るため、実運用相当の信頼性を要求するならフェーズ6へ直ちに進むより、
上記改善を別途計画してから進むことを推奨する。今回はフェーズ5で終了し、フェーズ6以降は
実装しない。

## 最終品質確認

```powershell
.\.venv-sim\python.exe -m ruff check src tests
.\.venv-sim\python.exe -m pytest -q
.\.venv-sim\python.exe -m pai_goal_conditioned.check_env --mode sim
.\.venv-sim\python.exe -m pai_goal_conditioned.smoke_test
```

- Ruff: All checks passed
- pytest: 17 passed
- sim環境: Python 3.10.22、PyTorch 2.8.0+cpu、Genesis 1.2.1、PASS
- CPU smoke test: PASS（`z=1.000 -> 0.544 m`、10.756秒）

