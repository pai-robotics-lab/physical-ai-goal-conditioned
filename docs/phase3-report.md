# フェーズ3 Goal-conditioned視覚AI用教師データ生成 検証報告

実施日: 2026-10-31

## 結論

フェーズ3は合格した。Genesis 1.2.1 CPUで1つのSceneを再利用し、固定カメラから赤・青・緑
の立方体を撮影して100シーン、600 Goal-conditionedサンプルを生成した。全ラベルが有効で、
Train/Validation/Test間のシーン・画像漏洩はない。CNN学習や推論統合には着手していない。

## データ生成設計

- 1シーンに赤・青・緑の50 mm立方体を1個ずつ配置する。
- 到達可能範囲 `x=[0.40, 0.70] m`、`y=[-0.22, 0.08] m` からseed付きで配置する。
- 物体中心間を100 mm以上離し、重なりを除外する。
- 配置後40 physics step進め、支持高さと並進速度を検査する。
- 640 x 480固定カメラからRGBとQA用link segmentationを同時取得する。
- 物体ごとにleft/right binの2指示を作り、1画像あたり6サンプルとする。
- 把持・搬送は実行せず、Scene構築とレンダラJITを100シーンで共有する。

セグメンテーションは対象物の可視性検査と確認用画像の作成だけに使う。学習入力として保存
するのはRGB画像であり、深度、マスク、シミュレーション真値を入力には含めない。

## 教師ラベル

各JSONLサンプルに次を保存した。

- RGB画像の相対パス
- `red cube to left bin`形式の指示
- 対象クラス
- 把持位置XYワールド座標
- 搬送先IDと固定搬送先座標
- シーンID、seed、split
- カメラ位置、注視点、up、FOV、解像度
- 対象物中心の真値XYZ
- ラベル有効性、可視画素数、無効理由
- 物理把持へ変換するhand姿勢とフェーズ2検証情報

CSVには学習・集計で使う主要な平坦化列も保存した。

## フェーズ2との把持整合性

学習対象のXYは、物理静置後の立方体中心XYそのものとした。実行時は次の固定変換で、
フェーズ2で10/10成功した上方把持姿勢へ変換できる。

- 接近方向: 鉛直上方からのtop-down
- hand quaternion (wxyz): `(0, 1, 0, 0)`
- hand target: `(grasp_x, grasp_y, cube_center_z + 0.110 m)`
- pregrasp target: `(grasp_x, grasp_y, cube_center_z + 0.255 m)`

従ってモデルは実行困難な抽象点ではなく、検証済み把持姿勢へ一意に変換できるXYを学習する。

## 小規模検証

修正後の最終10シーン検証結果は次のとおり。

- シーン: 10
- サンプル: 60
- Train/Validation/Test: 8/1/1シーン
- 有効ラベル: 60/60
- 失敗配置: 0
- 可視画素: 536～834 pixels/object
- 分割漏洩: 0
- キャッシュ後生成時間: 22.685秒
- 生成速度: 0.441 scenes/s、2.645 samples/s

Genesis 1.2.1の非colorized segmentation値が `link_idx + 1` であることを確認画像から特定し、
そのオフセットを反映した後に全物体の矩形と画像上の実物体が一致することを確認した。

## 100シーン本生成結果

| 項目 | 結果 |
|---|---:|
| RGBシーン数 | 100 |
| 教師サンプル数 | 600 |
| 1シーンあたりサンプル | 6 |
| Train | 80 scenes / 480 samples |
| Validation | 10 scenes / 60 samples |
| Test | 10 scenes / 60 samples |
| 有効ラベル | 600 / 600 |
| XYラベル最大誤差 | 0.0 m |
| 分割漏洩 | 0 |
| 可視画素 | 500～906 pixels/object |
| 生成時間 | 60.526秒 |
| 生成速度 | 1.652 scenes/s、9.913 samples/s |
| ディスク使用量 | 3,888,593 bytes（約3.89 MB） |

無効な初回配置は3件あった。scene 27の赤は可視画素不足・支持高さ・速度、scene 67と89の
青は支持高さ・速度条件で除外した。各失敗はseedと理由をmanifestへ保存し、次seedで再配置
して100有効シーンを生成した。

## 同一画像でのGoal条件

`scene_0000.png`から次の6サンプルを生成した。

| 指示 | 把持XY (m) | 搬送先ID |
|---|---|---|
| red cube to left bin | (0.50579, -0.16439) | left_bin |
| red cube to right bin | (0.50579, -0.16439) | right_bin |
| blue cube to left bin | (0.66966, -0.13594) | left_bin |
| blue cube to right bin | (0.66966, -0.13594) | right_bin |
| green cube to left bin | (0.43893, -0.07506) | left_bin |
| green cube to right bin | (0.43893, -0.07506) | right_bin |

色の指示を変えると同じRGB画像でも把持XYが変わり、搬送先の語を変えると把持XYを維持した
まま搬送先IDが変わる。Goal-conditioned学習に必要な対応が明示されている。

## 品質・再現性

- 把持XYと対象真値XYの最大誤差は0.0 m。
- 100画像すべてで3物体が可視画素閾値100を超えた。
- 画像・シーン単位でsplitを固定し、同一画像の指示違いは必ず同じsplitに入る。
- 10シーン検証と100シーン本生成の先頭10 PNGはSHA-256が10/10一致した。
- 固定カメラ、物体seed、実配置、失敗再試行seed、設定スナップショットを保存した。
- 色が明瞭で全対象が見えており、RGBと指示だけから対象色と把持位置を同定できる。
- 搬送先は指示中のleft/right IDから決まり、IDと固定ワールド座標の対応を設定に保存した。

## 成果物

- 実装: `src/pai_goal_conditioned/phase3_dataset.py`
- 設定: `configs/phase3_dataset.json`
- テスト: `tests/test_phase3_dataset.py`
- RGB: `datasets/phase3/images/scene_0000.png`～`scene_0099.png`
- オーバーレイ: `datasets/phase3/overlays/`（代表5シーン）
- 教師ラベル: `datasets/phase3/labels.jsonl`、`labels.csv`
- シーン真値: `datasets/phase3/scenes.jsonl`
- 分割: `datasets/phase3/splits/*_scenes.txt`
- 集計と失敗記録: `datasets/phase3/manifest.json`
- 設定スナップショット: `datasets/phase3/phase3_config_snapshot.json`

`datasets/`は再生成可能な大容量生成物としてGit管理外である。

## 判定

フェーズ4「視覚AIの学習」へ進行可能と判断する。ただし今回の作業範囲に従い、CNN学習、
モデル選定、推論、Panda統合には進まない。

## 最終品質確認

```powershell
.\.venv-sim\python.exe -m ruff check src tests
.\.venv-sim\python.exe -m pytest -q
.\.venv-sim\python.exe -m pai_goal_conditioned.check_env --mode sim
.\.venv-sim\python.exe -m pai_goal_conditioned.smoke_test
```

- Ruff: All checks passed
- pytest: 9 passed
- sim環境確認: PASS（Python 3.10.22、PyTorch 2.8.0+cpu、Genesis 1.2.1）
- CPU smoke test: PASS（`z=1.000 -> 0.544 m`、12.085秒）
