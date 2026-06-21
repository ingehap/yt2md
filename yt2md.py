#!/usr/bin/env python3
"""
yt2md - Transcribe the spoken audio of a YouTube video into a Markdown (.md) file.

Pipeline (every component is open source, runs locally, needs no API key):
  1. Download the best audio stream with yt-dlp and extract it to 16 kHz mono
     WAV using FFmpeg.
  2. Transcribe the audio locally with faster-whisper, the CTranslate2
     reimplementation of OpenAI Whisper (~4x faster, less memory, same accuracy).
     No audio leaves your machine.
  3. Write a Markdown file: a YAML front-matter metadata block followed by the
     transcript (timestamped segments by default, or continuous prose).

Usage
-----
    python yt2md.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python yt2md.py URL -m turbo -o talk.md
    python yt2md.py URL --model large-v3 --language en --no-timestamps
    python yt2md.py URL --task translate          # translate non-English speech to English

Prerequisites
-------------
    pip install -U yt-dlp faster-whisper
    FFmpeg must be installed and on PATH (https://ffmpeg.org/download.html):
      macOS:          brew install ffmpeg
      Debian/Ubuntu:  sudo apt install ffmpeg
      Windows:        winget install Gyan.FFmpeg   (or: choco install ffmpeg)

Model choices for --model (accuracy vs. speed / size trade-off)
---------------------------------------------------------------
    tiny, base, small (default), medium, large-v3   standard Whisper sizes
    turbo  (= large-v3-turbo)                       ~large-v3 accuracy, much faster; best on a GPU
    distil-large-v3                                 English-focused, fast
The first time a model is used it is downloaded automatically from the Hugging
Face Hub (needs internet). Larger models are more accurate but slower and use
more memory/disk. On a CPU, "small" is a sensible default; with an NVIDIA GPU,
"turbo" or "large-v3" are recommended. Note: "turbo" is trained for transcription
only, so use a non-turbo model (e.g. large-v3) with --task translate.

References
----------
  faster-whisper (SYSTRAN), Apache-2.0:   https://github.com/SYSTRAN/faster-whisper
  Whisper large-v3-turbo release notes:   https://github.com/openai/whisper/discussions/2363
  yt-dlp:                                 https://github.com/yt-dlp/yt-dlp
  yt-dlp format selection:                https://github.com/yt-dlp/yt-dlp#format-selection
  FFmpeg:                                 https://ffmpeg.org/
  Whisper paper, Radford et al. 2022:     https://arxiv.org/abs/2212.04356
  Silero VAD (used to suppress silence):  https://github.com/snakers4/silero-vad
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import re
import shutil
import sys
import tempfile

__version__ = "1.0.0"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def hms(seconds: float) -> str:
    """Format a number of seconds as HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_filename(name: str) -> str:
    """Turn an arbitrary title into a filesystem-safe base name."""
    name = re.sub(r'[\\/:*?"<>|\n\r\t]+', "_", name or "").strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    return (name or "transcript")[:120]


def check_ffmpeg() -> None:
    """Abort early with a clear message if FFmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ERROR: FFmpeg was not found on your PATH. It is required to extract "
            "audio.\n"
            "  macOS:          brew install ffmpeg\n"
            "  Debian/Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows:        winget install Gyan.FFmpeg\n"
            "See https://ffmpeg.org/download.html"
        )


def resolve_device(device_arg: str, compute_arg: str) -> tuple[str, str]:
    """Pick the compute device and quantization type.

    'auto' uses CUDA if a GPU is visible to CTranslate2, otherwise CPU.
    Default compute type is float16 on GPU and int8 on CPU (fast, low memory,
    negligible accuracy loss).
    """
    device = device_arg
    if device_arg == "auto":
        device = "cpu"
        try:
            import ctranslate2  # installed as a dependency of faster-whisper

            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
        except Exception:
            device = "cpu"

    compute = compute_arg
    if compute_arg == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


# --------------------------------------------------------------------------- #
# Step 1 - download + extract audio
# --------------------------------------------------------------------------- #
def download_audio(url: str, tmpdir: str, quiet: bool) -> tuple[str, dict]:
    """Download the best audio and extract it to 16 kHz mono WAV.

    Returns (wav_path, info_dict).
    """
    try:
        import yt_dlp
    except ImportError:
        sys.exit(
            "ERROR: yt-dlp is not installed.  Install it with:\n"
            "    pip install -U yt-dlp"
        )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "noplaylist": True,          # a URL with &list=... grabs only the one video
        "quiet": quiet,
        "no_warnings": quiet,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"},
        ],
        # Resample to 16 kHz mono (what Whisper expects) to keep the file small.
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:  # type: ignore[attr-defined]
        sys.exit(
            f"ERROR: yt-dlp could not download the video.\n  {exc}\n"
            "Common causes: the video is private, removed, region-locked, or "
            "age-restricted; or yt-dlp is out of date.\n"
            "Try updating it:  pip install -U yt-dlp\n"
            "For age-restricted videos, see yt-dlp's --cookies-from-browser option."
        )

    if info is None:
        sys.exit("ERROR: yt-dlp returned no video information.")

    # The extracted file is <id>.wav; fall back to whatever was produced.
    video_id = info.get("id", "audio")
    wav_path = os.path.join(tmpdir, f"{video_id}.wav")
    if not os.path.exists(wav_path):
        candidates = glob.glob(os.path.join(tmpdir, f"{video_id}.*"))
        wavs = [c for c in candidates if c.lower().endswith(".wav")]
        if wavs:
            wav_path = wavs[0]
        elif candidates:
            wav_path = candidates[0]  # faster-whisper can decode other formats too
        else:
            sys.exit("ERROR: no audio file was produced after download.")
    return wav_path, info


# --------------------------------------------------------------------------- #
# Step 2 - transcribe
# --------------------------------------------------------------------------- #
def transcribe_audio(
    audio_path: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
    task: str,
    beam_size: int,
    vad: bool,
    verbose: bool,
):
    """Run faster-whisper. Returns (segments_list, info)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "ERROR: faster-whisper is not installed.  Install it with:\n"
            "    pip install -U faster-whisper"
        )

    if verbose:
        print(
            f"[yt2md] Loading model '{model_size}' on {device} "
            f"(compute_type={compute_type}). First use downloads it from "
            f"Hugging Face.",
            file=sys.stderr,
        )

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        sys.exit(
            f"ERROR: could not load Whisper model '{model_size}' on device "
            f"'{device}' with compute_type '{compute_type}'.\n  {exc}\n"
            "If you are on CPU, try: --device cpu --compute-type int8\n"
            "If model download failed, check your internet connection."
        )

    segments, info = model.transcribe(
        audio_path,
        language=language,        # None -> auto-detect
        task=task,                # 'transcribe' or 'translate'
        beam_size=beam_size,
        vad_filter=vad,           # Silero VAD trims silence and reduces hallucinations
    )

    # segments is a lazy generator; iterating runs the transcription.
    collected = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            collected.append((seg.start, seg.end, text))
        if verbose:
            print(f"  [{hms(seg.start)} -> {hms(seg.end)}] {text}", file=sys.stderr)
    return collected, info


# --------------------------------------------------------------------------- #
# Step 3 - write Markdown
# --------------------------------------------------------------------------- #
def _yaml_escape(value: str) -> str:
    return str(value).replace('"', '\\"')


def build_markdown(segments, info, run_meta: dict, timestamps: bool) -> str:
    """Assemble the full Markdown document as a string."""
    title = info.get("title") or info.get("id") or "Untitled"
    uploader = info.get("uploader") or info.get("channel") or "unknown"
    url = info.get("webpage_url") or run_meta.get("url", "")
    video_id = info.get("id", "")
    duration = info.get("duration")
    duration_str = hms(duration) if duration else "unknown"

    upload_date = info.get("upload_date")  # 'YYYYMMDD'
    if upload_date and len(upload_date) == 8:
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}"
    else:
        upload_date = "unknown"

    lang = run_meta.get("detected_language") or run_meta.get("language") or "unknown"
    lang_prob = run_meta.get("language_probability")
    lang_prob_str = f"{lang_prob:.2f}" if isinstance(lang_prob, float) else "n/a"

    front = [
        "---",
        f'title: "{_yaml_escape(title)}"',
        f'source_url: "{_yaml_escape(url)}"',
        f'video_id: "{_yaml_escape(video_id)}"',
        f'uploader: "{_yaml_escape(uploader)}"',
        f"duration: \"{duration_str}\"",
        f"upload_date: \"{upload_date}\"",
        f'detected_language: "{lang}"',
        f"language_probability: {lang_prob_str}",
        f'asr_engine: "faster-whisper (CTranslate2 / OpenAI Whisper)"',
        f'transcription_model: "{run_meta["model"]}"',
        f'device: "{run_meta["device"]}"',
        f'compute_type: "{run_meta["compute_type"]}"',
        f'task: "{run_meta["task"]}"',
        f'generated_utc: "{run_meta["generated_utc"]}"',
        f'generated_by: "yt2md.py v{__version__}"',
        "---",
        "",
        f"# {title}",
        "",
        f"[Source]({url}) - uploaded by {uploader} - duration {duration_str}",
        "",
        "## Transcript",
        "",
    ]

    body_lines: list[str] = []
    if timestamps:
        # One segment per entry, blank-line separated so it renders unambiguously.
        for start, _end, text in segments:
            body_lines.append(f"`[{hms(start)}]` {text}")
            body_lines.append("")
    else:
        # Continuous prose: start a new paragraph on a pause > 2 s.
        paragraph: list[str] = []
        prev_end = None
        for start, end, text in segments:
            if prev_end is not None and (start - prev_end) > 2.0 and paragraph:
                body_lines.append(" ".join(paragraph))
                body_lines.append("")
                paragraph = []
            paragraph.append(text)
            prev_end = end
        if paragraph:
            body_lines.append(" ".join(paragraph))
            body_lines.append("")

    if not segments:
        body_lines.append("_(No speech was detected in the audio.)_")
        body_lines.append("")

    return "\n".join(front + body_lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yt2md.py",
        description=(
            "Transcribe the spoken audio of a YouTube video to a Markdown file "
            "using yt-dlp + faster-whisper (fully local, open source)."
        ),
        epilog=(
            "Models (--model): tiny, base, small, medium, large-v3, turbo, "
            "distil-large-v3.\n"
            "References: faster-whisper https://github.com/SYSTRAN/faster-whisper | "
            "yt-dlp https://github.com/yt-dlp/yt-dlp | "
            "Whisper https://arxiv.org/abs/2212.04356"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", help="YouTube video URL (or bare 11-character video ID)")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output .md path (default: <video-title>.md in the current directory)",
    )
    parser.add_argument(
        "-m", "--model", default="small",
        help="Whisper model size/name (default: small). E.g. turbo, large-v3.",
    )
    parser.add_argument(
        "-l", "--language", default=None,
        help="Force a language code (e.g. en, no, de). Default: auto-detect.",
    )
    parser.add_argument(
        "--task", choices=["transcribe", "translate"], default="transcribe",
        help="'transcribe' keeps the spoken language; 'translate' renders English.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Compute device (default: auto-detect GPU, else CPU).",
    )
    parser.add_argument(
        "--compute-type", default="auto",
        help="CTranslate2 compute type, e.g. int8, float16, int8_float16 "
             "(default: auto).",
    )
    parser.add_argument(
        "--beam-size", type=int, default=5,
        help="Beam size for decoding (default: 5).",
    )
    parser.add_argument(
        "--no-timestamps", dest="timestamps", action="store_false",
        help="Write continuous prose instead of per-segment timestamps.",
    )
    parser.add_argument(
        "--no-vad", dest="vad", action="store_false",
        help="Disable the Silero voice-activity-detection filter.",
    )
    parser.add_argument(
        "--keep-audio", action="store_true",
        help="Keep the extracted WAV file next to the output .md.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print progress (including each segment) to stderr.",
    )
    parser.add_argument("--version", action="version", version=f"yt2md.py {__version__}")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    check_ffmpeg()

    device, compute_type = resolve_device(args.device, args.compute_type)
    if args.task == "translate" and args.model == "turbo":
        print(
            "[yt2md] Warning: the 'turbo' model is not trained for translation. "
            "Consider --model large-v3 with --task translate.",
            file=sys.stderr,
        )

    tmpdir = tempfile.mkdtemp(prefix="yt2md_")
    try:
        print(f"[yt2md] Downloading audio: {args.url}", file=sys.stderr)
        wav_path, info = download_audio(args.url, tmpdir, quiet=not args.verbose)

        print(f"[yt2md] Transcribing with '{args.model}' on {device} ...",
              file=sys.stderr)
        segments, tinfo = transcribe_audio(
            wav_path,
            model_size=args.model,
            device=device,
            compute_type=compute_type,
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
            vad=args.vad,
            verbose=args.verbose,
        )

        run_meta = {
            "url": args.url,
            "model": args.model,
            "device": device,
            "compute_type": compute_type,
            "task": args.task,
            "language": args.language,
            "detected_language": getattr(tinfo, "language", None),
            "language_probability": getattr(tinfo, "language_probability", None),
            "generated_utc": _dt.datetime.now(_dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        markdown = build_markdown(segments, info, run_meta, timestamps=args.timestamps)

        # Decide output path.
        if args.output:
            out_path = args.output
            if not out_path.lower().endswith(".md"):
                out_path += ".md"
        else:
            base = safe_filename(info.get("title") or info.get("id") or "transcript")
            out_path = f"{base}.md"

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(markdown)

        words = sum(len(t.split()) for _, _, t in segments)
        print(
            f"[yt2md] Done. {len(segments)} segments, ~{words} words -> {out_path}",
            file=sys.stderr,
        )

        if args.keep_audio:
            dest = os.path.splitext(out_path)[0] + ".wav"
            try:
                shutil.copy2(wav_path, dest)
                print(f"[yt2md] Kept audio: {dest}", file=sys.stderr)
            except OSError as exc:
                print(f"[yt2md] Could not keep audio: {exc}", file=sys.stderr)

        return 0
    except KeyboardInterrupt:
        print("\n[yt2md] Interrupted.", file=sys.stderr)
        return 130
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
