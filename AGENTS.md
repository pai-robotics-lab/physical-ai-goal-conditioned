# AGENTS.md

このファイルは、このリポジトリで作業するAIエージェント向けの共通指示です。
適用範囲はリポジトリ全体です。

## プロジェクト概要

- Physical AI応用1講座の最終課題として、「視覚AIを用いたPandaロボットアームの
  物体把持・搬送」を開発する。CNNは把持位置XYを推定し、IK・軌道・把持力・
  搬送先選択は従来型制御が担う。
- 既存のゴールキーパープロジェクトとは完全に独立させる。
- Pythonコードは `src/pai_goal_conditioned/`、テストは `tests/` に置く。
- 標準シミュレーションはGenesis 1.2.1のCPUバックエンドを使用する。
- 小型CNN学習だけを独立したCUDA版PyTorch環境で実行する。

## 現在の進捗

- フェーズ0（独立環境）、フェーズ1（Panda IK）、フェーズ2（固定座標Pick and Place）、
  フェーズ3（構造化指示付き教師データ生成）、フェーズ4（視覚AI学習・オフライン評価）、
  フェーズ5（開ループオンライン自律操作）、フェーズ6（把持誤差診断）、
  フェーズ7（データ増量・精度改善・ペア物理比較）、フェーズ8（最終成果物・公開準備）は完了。
- フェーズ2はGenesis CPUの独立初期化10試行で10/10成功、配置水平誤差3.41 mm。
- フェーズ3は100シーン、600サンプルを生成し、ラベル有効600/600、分割漏洩0。
- フェーズ4のTest平均XY誤差は15.25 mmで、暫定目標20 mmを達成。
- フェーズ5は未知配置10試行中7回指示達成。平均推定誤差14.91 mm、失敗3件は把持段階。
- フェーズ6で失敗3件を3/3再現し、診断用真値把持は3/3成功。代表配置の観測成功区間は
  X=-10～+15 mm、Y=-15～+10 mmだが、各点1試行の暫定値である。
- フェーズ7はTrain 2,000モデルで共通Test平均2.14 mm、ペア物理24/24成功。旧モデルは
  同条件15/24で、データ増量による改善を確認した。
- 新テーマに合わせた最終レポート、提出用動画、公開用README、第三者ライセンス通知を
  作成済み。Data Scalingは追加の性能評価として扱う。
- パッケージ名や一部の履歴資料に残る`goal_conditioned`は実装履歴を示す名称であり、
  CNNが搬送先まで含むロボット制御方策を学習したことを意味しない。
- GitHub Push、動画アップロード、講座提出はこの更新では実施しない。

## 環境

- `.venv-sim`: Python 3.10、Genesis 1.2.1、CPU版PyTorch。
- `.venv-train`: Python 3.10、CUDA版PyTorch。Genesisの実行には使用しない。
- 既存のConda環境や別プロジェクトの仮想環境を変更しない。
- GTX 1050の制約から、GenesisのCUDA物理を標準経路にしない。
- 環境構築・実行にはプロジェクト内prefixのPythonを明示して使用する。

## 実装規約

- Ruffの行長100、Python 3.10互換に従う。
- Genesis初期化前にQuadrants/GsTaichiキャッシュを `.cache/` に設定する。
- 標準テストはヘッドレスで実行できるようにする。
- 乱数シード、実験条件、誤差、実行時間、失敗理由を機械可読形式で保存する。
- シミュレーション真値はデータ生成と評価に限定し、最終推論の制御入力に使わない。
- APIキーや個人情報をコード、ログ、動画、レポートへ含めない。

## 検証

通常の変更後は次を実行する。

```powershell
.\.venv-sim\python.exe -m ruff check src tests
.\.venv-sim\python.exe -m pytest -q
.\.venv-sim\python.exe -m pai_goal_conditioned.check_env --mode sim
.\.venv-sim\python.exe -m pai_goal_conditioned.smoke_test
```

CUDA学習環境を変更した場合は次も実行する。

```powershell
.\.venv-train\python.exe -m pai_goal_conditioned.check_env --mode train
```

## 生成物

- `.venv-sim/`、`.venv-train/`、`.cache/`、`artifacts/`、`datasets/`、`models/`、
  `outputs/`、`recordings/`、`videos/` はGit管理しない。
- 学習済みモデルや大容量動画をGitへ追加しない。
- GitHub公開や外部提出は、ユーザーの明示的な依頼なしに行わない。

## 作業ログ

- 作業日は `worklogs/YYYY-MM-DD.md` に、実施内容、環境、結果、判断、問題、次の作業を記録する。
- 長いコマンド出力を貼らず、再現に必要なコマンドと結果の要点を残す。

