I'll research the current best-in-class open tools first (these libraries change often, 
especially anything touching YouTube), then build and sanity-check the script.Confirmed 
faster-whisper is the actively maintained standard and supports the `turbo` model. Now 
checking the audio-download side (yt-dlp) and the lightweight caption alternative.Good 
— yt-dlp is confirmed current (updated within weeks) and remains the standard. One more 
check on the lightweight caption-fetch alternative, then I'll build the script.I have 
what I need: the pipeline is yt-dlp (audio download) + FFmpeg (extraction) + faster-whisper 
(local ASR), all open and actively maintained. I'll build the script, then verify what 
the sandbox allows (syntax, CLI, import surface). Note: I can't run a full end-to-end 
test here because YouTube and Hugging Face aren't reachable from this sandbox — but the 
code paths and API names are verifiable.Now I'll verify the script: syntax, the CLI surface 
(works without heavy deps thanks to lazy imports), then install the dependencies from PyPI 
to confirm the import paths and API names are correct.CLI and lazy imports work. Now 
installing the two dependencies from PyPI to verify the import paths and API signatures 
I used are correct.Import paths and API signatures all check out (yt-dlp 2026.06.09, 
ctranslate2 4.8.0, and `transcribe()` accepts `language`/`task`/`beam_size`/`vad_filter`). 

Reviewing the data flow, I caught a bug: I passed the faster-whisper info object to the 
Markdown builder, but that function reads video metadata (title, uploader, etc.) which 
lives on the yt-dlp dict. Let me fix the two info sources and then render synthetic output 
to confirm the format.Output renders correctly in all three cases — timestamped, prose 
(note the paragraph break correctly triggered by the 3.7 s pause), and the no-speech fallback. 
YAML front-matter is properly escaped, and duration/date formatting is right.

Now I'll add a `requirements.txt` and place both files in the output directory.Done. The 
script and its `requirements.txt` are above.

What it does: it performs genuine speech-to-text on the audio (not caption-scraping), in 
three local stages — yt-dlp downloads the best audio stream, FFmpeg extracts it to 16 kHz 
mono WAV, and faster-whisper transcribes it. Everything runs on your machine with no API 
key, and the output is a Markdown file with a YAML front-matter metadata block (title, 
source URL, uploader, duration, detected language, model, etc.) followed by the transcript.

Why these tools: yt-dlp is the actively maintained fork of youtube-dl and the de facto 
standard for YouTube downloads, relying on FFmpeg for audio extraction. It receives frequent 
updates to stay compatible with YouTube's changes and, paired with FFmpeg, can extract and 
convert audio. faster-whisper is the CTranslate2 reimplementation of OpenAI Whisper; it runs 
up to 4x faster than the original implementation while using less memory, with the same accuracy.

Setup and use:
```
pip install -U -r requirements.txt     # or: pip install -U yt-dlp faster-whisper
# plus FFmpeg on PATH (brew/apt/winget install ffmpeg)

python yt2md.py "https://www.youtube.com/watch?v=VIDEO_ID"
python yt2md.py URL -m turbo -o talk.md
python yt2md.py URL --model large-v3 --language no --no-timestamps
```

On model choice: the default is `small` (good CPU balance, ~500 MB). For higher accuracy use 
`large-v3`, or `turbo` (large-v3-turbo). Turbo delivers accuracy similar to large-v3 at 
significantly higher speed, and because its encoder is as heavy as large-v3 but its decoder is 
much lighter, it benefits strongly from GPU acceleration — so `turbo` is the sweet spot if you 
have an NVIDIA GPU. One caveat the script warns about: turbo was fine-tuned on transcription 
data excluding translation, so it is not expected to perform well at translation; use `large-v3` 
with `--task translate` instead. The Silero VAD filter is on by default, which trims silence 
and reduces Whisper's tendency to hallucinate during quiet passages.

What I verified vs. couldn't: I confirmed the script compiles, the CLI works, the import paths 
and the `WhisperModel.transcribe` signature (`language`/`task`/`beam_size`/`vad_filter`) are 
correct against the currently installed libraries (yt-dlp 2026.06.09, faster-whisper with 
ctranslate2 4.8.0), and that the Markdown renders correctly. I could not run a full end-to-end 
transcription here because this sandbox can't reach YouTube or the Hugging Face model host — so 
the first real run on your machine will download the chosen model (needs internet), then work 
offline thereafter.

A lighter alternative, if you only ever need videos that already have captions: the 
`youtube-transcript-api` library fetches existing captions in milliseconds without downloading 
audio or a model. I deliberately did not build the script around it, both because you asked to 
transcribe the spoken words and because YouTube now blocks most requests from cloud-provider 
IP ranges (AWS, GCP, Azure, etc.), so caption-fetching tends to fail off a residential connection 
and depends on captions existing at all. The ASR approach is the robust, source-independent one.

References:
- faster-whisper (SYSTRAN, Apache-2.0): https://github.com/SYSTRAN/faster-whisper
- Whisper large-v3-turbo release notes: https://github.com/openai/whisper/discussions/2363
- yt-dlp: https://github.com/yt-dlp/yt-dlp
- FFmpeg: https://ffmpeg.org/
- Whisper paper (Radford et al., 2022), arXiv:2212.04356: https://arxiv.org/abs/2212.04356
- Silero VAD: https://github.com/snakers4/silero-vad
- youtube-transcript-api (the alternative): https://pypi.org/project/youtube-transcript-api/

