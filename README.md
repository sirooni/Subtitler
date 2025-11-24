# Subtitler

動画ファイルから音声を抽出し、WhisperXを用いて文字起こしを行い、SRT字幕ファイルとDaVinci Resolve用のFCPXMLファイルを生成するツールです。

## 必要要件

- Python 3.8+
- ffmpeg
- ffprobe
- CUDA (GPUを使用する場合)

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

```bash
python subtitler.py <動画ファイルパス> [オプション]
```

### オプション

- `--skip-transcribe`: 文字起こしをスキップし、既存の `segments.json` を使用します。
- `--skip-align`: アラインメントをスキップし、既存の `aligned.json` を使用します。
- `--skip-tokenize`: 形態素解析をスキップし、既存の `tokens.json` を使用します。

### 出力

実行すると `outputs/<動画ファイル名>/` ディレクトリに以下のファイルが生成されます。

- `audio.wav`: 抽出された音声
- `<動画ファイル名>.srt`: SRT形式の字幕ファイル
- `<動画ファイル名>.fcpxml`: DaVinci Resolve (Text+) 用のタイムラインファイル
- 中間JSONファイル (`segments.json`, `aligned.json`, `tokens.json`)

## DaVinci Resolve へのインポート

1. DaVinci Resolve を開き、プロジェクトを作成または開きます。
2. メニューから **File > Import > Timeline...** を選択します。
3. 生成された `.fcpxml` ファイルを選択します。
4. 設定ダイアログが表示されたら、必要に応じて設定を確認し **OK** をクリックします。
5. タイムラインに字幕が編集可能なクリップとして配置されます。
