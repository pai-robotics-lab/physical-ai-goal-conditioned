# GitHub公開・講座提出前チェックリスト

確認日: 2026-10-31

公開予定ファイル数: 67

## 公開予定

- `.gitattributes`、`.gitignore`、`AGENTS.md`、`LICENSE`、`README.md`、`THIRD_PARTY_NOTICES.md`
- `pyproject.toml`、`setup-sim.ps1`、`setup-train.ps1`
- `configs/*.json`
- `src/pai_goal_conditioned/*.py`
- `tests/*.py`
- `scripts/*.py`
- `docs/*.md`、`docs/assets/final/*.png`
- `output/pdf/physical-ai-goal-conditioned-final-report.pdf`
- `worklogs/*.md`

## Git管理から除外

- 仮想環境: `.venv-sim/`、`.venv-train/`
- キャッシュ: `.cache/`、`tmp/`
- 生成データ・モデル・実験出力: `datasets/`、`models/`、`outputs/`、`artifacts/`
- 動画・収録: `recordings/`、`videos/`、`*.mp4`
- 学習重み: `*.pt`

## 完了した確認

- [x] README、最終レポート、図表にユーザー名・絶対ローカルパスを記載していない
- [x] 公開候補へAPIキー、パスワード、認証トークンを記載していない
- [x] モデル重み、仮想環境、キャッシュ、大容量データ、動画をGit除外した
- [x] Genesis 1.2.1、Pandaアセット、PyTorch等のライセンス情報をローカル配布物で確認した
- [x] 第三者ライセンス情報を`THIRD_PARTY_NOTICES.md`へ記録した
- [x] 自作ソースコードのMIT Licenseを`LICENSE`へ追加した
- [x] Git Author/Committerを公開用アカウントのGitHub noreplyアドレスへ統一した
- [x] PDFの日本語表示、ページレイアウト、図表を目視確認した
- [x] 動画を全フレームデコードし、解像度・再生時間・代表フレームを確認した

## 公開前に所有者が判断・実施する項目

- [ ] 動画を公開先へアップロードし、READMEの動画URL欄を更新する
- [ ] GitHubで公開対象の最終diffとコミット作者情報を確認する
- [ ] 講座規約が生成データ・モデル・Genesis画面の公開を許可するか確認する
- [ ] 必要なら動画へ音声、字幕、BGMを追加し、その素材ライセンスを確認する

## Physical AI応用1 提出チェックリスト

- OmniCampusアカウントID: `140354`

### 説明資料PDF

- [x] 確定済み13ページの最終PDFを本文変更なしで提出用ファイルへコピーした
- [x] 提出ファイル名を`説明資料_PAI最終課題_140354.pdf`とした
- [x] コピー元と提出用PDFのSHA-256一致を確認した
- [ ] OmniCampusのPDFアップロード欄へ`説明資料_PAI最終課題_140354.pdf`をアップロードする
- [ ] アップロード後にファイル名と13ページすべてが正常に閲覧できることを確認する

### YouTube動画

- [x] 提出対象が88.00秒の改訂版`physical-ai-goal-conditioned-final-demo-enhanced.mp4`であることを確認した
- [ ] YouTubeの正式タイトルを`PAI最終課題_140354`としてアップロードする
- [ ] 「この動画は子ども向けですか？」を「いいえ」にする
- [ ] 年齢制限を「いいえ」にする
- [ ] 公開範囲を「公開」または「限定公開」にする
- [ ] 「公開」の場合もインスタントプレミア公開・プレミア公開を使用しない
- [ ] 公開日時を予約する場合は提出締切までに視聴可能になる日時を設定する
- [ ] 採点完了までは動画を削除せず、「非公開」へ変更しない
- [ ] YouTube URLはアップロード後に記録する（現時点では未設定。仮URLを作らない）

### GitHub

- [x] 公開先候補を`pai-robotics-lab/physical-ai-goal-conditioned`とした
- [ ] GitHubリポジトリをPublicで作成する
- [ ] 初回コミット、Remote設定、Push後にリポジトリURLを記録する
- [ ] 採点に必要な期間はPublicを維持する

### OmniCampus入力・送信前確認

- [ ] レポートのタイトル欄へ`140354.pdf`と入力する
- [ ] Q1: YouTubeの動画URLを入力する
- [ ] Q2: YouTubeの公開設定として「公開」または「限定公開」を入力する
- [ ] Q3: PublicにしたGitHubリポジトリのURLを入力する
- [ ] Q4: 提出物に個人情報が含まれないことを確認し、`確認しました`と入力する
- [ ] PDFアップロード欄に提出用PDFが添付されていることを確認する
- [ ] PDFアップロード後のプレビューを確認し、必要ならダウンロードして正常に開けることを確認する
- [ ] シークレットウィンドウでYouTube URLを開き、ログアウト状態で再生できることを確認する
- [ ] シークレットウィンドウでGitHub URLを開き、ログアウト状態でPublic閲覧できることを確認する
- [ ] Q1〜Q4、PDF、YouTube URL、GitHub URLを送信直前に再確認する
- [ ] 「提出する」を押した後は、完了表示までリロードやページ移動をしない
- [ ] `レポートが提出されました`の表示と提出日時を確認し、受付記録を保存する
- [ ] 再提出時は最後にアップロードした内容だけが採点対象になることを確認する

GitHub Push、公開設定変更、YouTubeアップロード、講座提出はこのフェーズでは実施していない。
