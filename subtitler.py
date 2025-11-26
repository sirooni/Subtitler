import os
import sys
import json
import winsound
import subprocess
import whisperx
from sudachipy import tokenizer
from sudachipy import dictionary

import xml.etree.ElementTree as ET
from fractions import Fraction

# ===== 設定 =====
language = "ja"
whisperx_model = "large-v3"
device = "cpu"  # GPU: "cuda", CPU: "cpu"
compute_type = "float32"  # float16で失敗する場合はfloat32
min_chars = 5  # 1字幕の最低文字数
max_chars = 20  # 1字幕の最大文字数
min_duration = 1.0  # 最低表示時間(秒)
gap_threshold = 0.25  # セリフ切れ目の閾値(秒)
long_token_threshold = 0.1  # 文節の閾値(秒)

# Sudachi設定
sudachi_tokenizer = dictionary.Dictionary().create()
sudachi_mode = tokenizer.Tokenizer.SplitMode.C


# ===== 音声抽出 =====
def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        audio_path,
    ]
    subprocess.run(cmd, check=True)


# ===== 動画情報取得 =====
def get_video_info(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,duration,width,height",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        r_frame_rate = stream["r_frame_rate"]
        num, den = map(int, r_frame_rate.split("/"))
        fps = num / den
        duration = float(stream.get("duration", 0))
        width = int(stream.get("width", 1920))
        height = int(stream.get("height", 1080))
        return {
            "fps": fps,
            "frame_duration_num": den,
            "frame_duration_den": num,
            "duration_sec": duration,
            "width": width,
            "height": height,
        }
    except Exception as e:
        print(f"Warning: Could not get video info via ffprobe: {e}")
        # Default to 30fps 1080p
        return {
            "fps": 30.0,
            "frame_duration_num": 1001,
            "frame_duration_den": 30000,
            "duration_sec": 3600.0,
            "width": 1920,
            "height": 1080,
        }


# ===== WhisperXで文字起こし =====
def transcribe_whisperx(audio_path):
    audio = whisperx.load_audio(audio_path)
    model = whisperx.load_model(whisperx_model, device, compute_type=compute_type)
    result = model.transcribe(audio, language=language)
    return result


# ===== アラインメント =====
def align_segments(audio_path, segments):
    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    aligned_result = whisperx.align(
        segments, align_model, metadata, audio_path, device=device
    )
    return aligned_result


# ===== Sudachiで日本語分割 =====
def tokenize_japanese(aligned_segments, save_json=None):
    tokens = []
    
    # 全ての文字とタイムスタンプのペアリストを作成
    chars = []
    for seg in aligned_segments:
        if "words" not in seg:
            continue
        for w in seg["words"]:
            if "word" not in w:
                continue
            word_text = w["word"]
            start = w.get("start")
            end = w.get("end")
            
            # 単語の時間が取得できない場合は、セグメントの時間を使うなどの補完も考えられるが
            # ここではNoneのままとして、後続処理でハンドリングする
            
            # 単語を文字に分解してリストに追加
            # 時間は単語全体で均等割り...ではなく、単語全体で同じ時間を持つとする（簡易実装）
            # または、文字数で割ることも可能だが、WhisperXの精度次第。
            # ここでは「その文字が含まれる単語の開始・終了時間」を保持する。
            for c in word_text:
                chars.append({"char": c, "start": start, "end": end})

    # 全テキストを結合してSudachiで解析
    full_text = "".join([c["char"] for c in chars])
    sudachi_tokens = sudachi_tokenizer.tokenize(full_text, sudachi_mode)

    # 解析結果をcharsリストにマッピング
    char_idx = 0
    for m in sudachi_tokens:
        surface = m.surface()
        length = len(surface)
        
        if length == 0:
            continue

        # 対応する文字範囲の時間を取得
        token_start = None
        token_end = None
        
        # マッピング範囲のチェック
        if char_idx + length <= len(chars):
            # 開始時間は最初の文字のstart
            # 終了時間は最後の文字のend
            # ただしNoneが含まれる可能性を考慮
            
            # 有効な開始時間を探す
            for i in range(length):
                s = chars[char_idx + i]["start"]
                if s is not None:
                    token_start = s
                    break
            
            # 有効な終了時間を探す（後ろから）
            for i in range(length - 1, -1, -1):
                e = chars[char_idx + i]["end"]
                if e is not None:
                    token_end = e
                    break
        
        tokens.append({"surface": surface, "start": token_start, "end": token_end})
        char_idx += length

    # JSON保存
    if save_json:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)

    return tokens



# ===== SRT読み込み =====
def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize newlines
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    blocks = content.strip().split("\n\n")

    srt_lines = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) < 3:
            continue

        # Index (skip)
        # index = lines[0]

        # Timestamp
        time_line = lines[1]
        if " --> " not in time_line:
            continue
        
        start_str, end_str = time_line.split(" --> ")
        
        def parse_time(t_str):
            t_str = t_str.strip()
            # HH:MM:SS,mmm
            parts = t_str.replace(",", ".").split(":")
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s

        start = parse_time(start_str)
        end = parse_time(end_str)

        # Text
        text = "\n".join(lines[2:])
        
        srt_lines.append({
            "start": start,
            "end": end,
            "text": text
        })

    return srt_lines


# ===== SRT整形 =====
def format_srt_with_tokens(
    tokens,
    min_chars=5,
    max_chars=20,
    min_duration=2.0,
    gap_threshold=0.5,
    long_token_threshold=0.2,
):
    srt_lines = []
    idx = 1
    current_text = ""
    current_start = None
    current_end = None

    for token in tokens:
        text = token["surface"]
        start = token["start"]
        end = token["end"]

        # 無効データスキップ
        if start is None or end is None:
            continue

        # 新しいブロックを開始する条件
        should_split = False
        if current_text and (start - current_end > gap_threshold):
            should_split = True
        if len(current_text) + len(text) > max_chars:
            should_split = True

        if should_split:
            # 既存の字幕ブロックを追加
            if current_text:
                duration = max(current_end - current_start, min_duration)
                srt_lines.append(
                    {
                        "index": idx,
                        "start": current_start,
                        "end": current_start + duration,
                        "text": current_text.strip(),
                    }
                )
                idx += 1
            # 新しいブロック開始
            current_text = text
            current_start = start
            current_end = end
        else:
            if not current_text:
                current_start = start
            current_text += text
            current_end = end

        is_phrase_finished = (
            end - start
        ) >= long_token_threshold or text in "。、！？!?"
        over_min_chars = len(current_text) >= min_chars
        if is_phrase_finished and over_min_chars:
            if current_text:
                duration = max(current_end - current_start, min_duration)
                srt_lines.append(
                    {
                        "index": idx,
                        "start": current_start,
                        "end": current_start + duration,
                        "text": current_text.strip(),
                    }
                )
                idx += 1
            # 新しいブロック開始
            current_text = ""
            current_start = start
            current_end = end

    # 最後のブロックを追加
    if current_text:
        duration = max(current_end - current_start, min_duration)
        srt_lines.append(
            {
                "index": idx,
                "start": current_start,
                "end": current_start + duration,
                "text": current_text.strip(),
            }
        )

    return srt_lines


# ===== SRT書き込み =====
def write_srt(srt_lines, srt_path):
    def format_timestamp(seconds):
        if seconds is None:
            return "00:00:00,000"
        if seconds < 0:
            seconds = 0
        ms = int((seconds - int(seconds)) * 1000)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for line in srt_lines:
            text = line["text"].strip()
            # 改行コードを削除（念のため）
            text = text.replace("\n", "").replace("\r", "")
            
            f.write(f"{line['index']}\n")
            f.write(
                f"{format_timestamp(line['start'])} --> {format_timestamp(line['end'])}\n"
            )
            f.write(f"{text}\n\n")


# ===== FCPXML書き込み =====
def write_fcpxml(srt_lines, fcpxml_path, video_info):
    fps = video_info["fps"]
    frame_duration_num = video_info["frame_duration_num"]
    frame_duration_den = video_info["frame_duration_den"]
    width = video_info["width"]
    height = video_info["height"]
    duration_sec = video_info["duration_sec"]

    frame_duration_str = f"{frame_duration_num}/{frame_duration_den}s"
    total_frames = int(duration_sec * fps)
    total_duration_str = f"{total_frames * frame_duration_num}/{frame_duration_den}s"

    fcpxml = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(fcpxml, "resources")
    format_elem = ET.SubElement(
        resources,
        "format",
        id="r1",
        name=f"FFVideoFormat{width}x{height}p{fps}",
        frameDuration=frame_duration_str,
        width=str(width),
        height=str(height),
    )
    effect_elem = ET.SubElement(
        resources,
        "effect",
        id="r2",
        name="Basic Title",
        uid=".../Titles.localized/Bumper: Opener.localized/Basic Title.localized/Basic Title.moti",
    )

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="Subtitles")
    project = ET.SubElement(event, "project", name="Subtitle Project")
    sequence = ET.SubElement(
        project,
        "sequence",
        format="r1",
        duration=total_duration_str,
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence, "spine")

    # Base gap to hold titles
    base_gap = ET.SubElement(
        spine,
        "gap",
        name="Gap",
        offset="0s",
        duration=total_duration_str,
        start="0s",
    )

    # Add titles
    for i, line in enumerate(srt_lines):
        start_sec = line["start"]
        end_sec = line["end"]
        text = line["text"]

        # Calculate frames
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        duration_frame = end_frame - start_frame

        start_str = f"{start_frame * frame_duration_num}/{frame_duration_den}s"
        duration_str = f"{duration_frame * frame_duration_num}/{frame_duration_den}s"
        offset_str = start_str  # Offset relative to the parent gap (which starts at 0)

        title = ET.SubElement(
            base_gap,
            "title",
            name=f"Subtitle {i+1}",
            lane="1",
            offset=offset_str,
            ref="r2",
            duration=duration_str,
            start=start_str,
        )
        
        # Param for text content (standard Text+ / Basic Title structure)
        # Note: The exact structure for "Text+" vs "Basic Title" can vary.
        # "Basic Title" is safer for general compatibility.
        # To make it "Text+", we might need a different effect UID, but Basic Title is usually editable.
        
        text_elem = ET.SubElement(title, "text")
        text_style = ET.SubElement(text_elem, "text-style", ref=f"ts{i+1}")
        text_style.text = text

        text_style_def = ET.SubElement(title, "text-style-def", id=f"ts{i+1}")
        text_style_val = ET.SubElement(
            text_style_def,
            "text-style",
            font="Hiragino Kaku Gothic ProN",
            fontSize="50",
            fontFace="W3",
            fontColor="1 1 1 1",
            alignment="center",
        )

    tree = ET.ElementTree(fcpxml)
    ET.indent(tree, space="  ", level=0)
    tree.write(fcpxml_path, encoding="utf-8", xml_declaration=True)


import argparse

# ===== メイン処理 =====
def main():
    parser = argparse.ArgumentParser(description="Video to Subtitle/FCPXML Converter")
    parser.add_argument("video_file", help="Path to the input video file")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription and use existing segments.json")
    parser.add_argument("--skip-align", action="store_true", help="Skip alignment and use existing aligned.json")
    parser.add_argument("--skip-tokenize", action="store_true", help="Skip tokenization and use existing tokens.json")
    parser.add_argument("--import-srt", action="store_true", help="Skip all processing and import from an existing SRT file")
    parser.add_argument("--srt-path", help="Path to the SRT file to import (used with --import-srt). Defaults to outputs/<base>/<base>.srt")

    args = parser.parse_args()

    video_path = args.video_file

    base = os.path.splitext(os.path.basename(video_path))[0]
    outputs_dir = os.path.join("outputs", base)
    os.makedirs(outputs_dir, exist_ok=True)
    audio_path = os.path.join(outputs_dir, "audio.wav")
    srt_path = os.path.join(outputs_dir, f"{base}.srt")
    fcpxml_path = os.path.join(outputs_dir, f"{base}.fcpxml")
    seg_json = os.path.join(outputs_dir, "segments.json")
    aligned_json = os.path.join(outputs_dir, "aligned.json")
    token_json = os.path.join(outputs_dir, "tokens.json")

    # 動画情報取得
    print("[0/5] Getting video info...")
    video_info = get_video_info(video_path)
    print(f"      FPS: {video_info['fps']}, Size: {video_info['width']}x{video_info['height']}")

    if args.import_srt:
        import_path = args.srt_path if args.srt_path else srt_path
        if not os.path.exists(import_path):
             print(f"Error: SRT file not found at {import_path}")
             sys.exit(1)
             
        print(f"[1-4/5] Importing SRT from {import_path}...")
        srt_lines = parse_srt(import_path)
    else:
        print("[1/5] Extracting audio...")
        extract_audio(video_path, audio_path)
        winsound.Beep(1000, 200)

        if args.skip_transcribe and os.path.exists(seg_json):
            print("[2/5] Skipping transcription, loading segments.json...")
            with open(seg_json, "r", encoding="utf-8") as f:
                segments = json.load(f)["segments"]
        else:
            print("[2/5] Transcribing with WhisperX...")
            result = transcribe_whisperx(audio_path)
            segments = result["segments"]
            with open(seg_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        winsound.Beep(1000, 200)

        if args.skip_align and os.path.exists(aligned_json):
            print("[3/5] Skipping alignment, loading aligned.json...")
            with open(aligned_json, "r", encoding="utf-8") as f:
                aligned_result = json.load(f)
        else:
            print("[3/5] Aligning segments...")
            aligned_result = align_segments(audio_path, segments)
            with open(aligned_json, "w", encoding="utf-8") as f:
                json.dump(aligned_result, f, ensure_ascii=False, indent=2)
        winsound.Beep(1000, 200)

        if args.skip_tokenize and os.path.exists(token_json):
            print("[4/5] Skipping tokenization, loading tokens.json...")
            with open(token_json, "r", encoding="utf-8") as f:
                tokens = json.load(f)
        else:
            print("[4/5] Tokenizing with Sudachi B-mode...")
            tokens = tokenize_japanese(aligned_result["segments"], save_json=token_json)
        winsound.Beep(1000, 200)

        print(f"[5/5] Formatting and writing SRT to {srt_path} ...")
        srt_lines = format_srt_with_tokens(
            tokens, min_chars, max_chars, min_duration, gap_threshold, long_token_threshold
        )
        write_srt(srt_lines, srt_path)
    
    print(f"      Writing FCPXML to {fcpxml_path} ...")
    write_fcpxml(srt_lines, fcpxml_path, video_info)
    
    winsound.Beep(1000, 200)

    print("Done.")


if __name__ == "__main__":
    main()
