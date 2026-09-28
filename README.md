# meetscribe-server

Lean GPU speech backend for [MeetScribe](https://github.com/koekaverna/meetscribe). It serves the
subset of the [speaches](https://github.com/speaches-ai/speaches) API that MeetScribe uses, with
current library versions and nothing else in the image:

| Endpoint | Engine |
|---|---|
| `POST /v1/audio/transcriptions` | faster-whisper 1.2 / CTranslate2 4.8 (`BatchedInferencePipeline`, Silero VAD v6) |
| `POST /v1/audio/diarization` | onnx-diarization: pyannote `segmentation_community_1` + WeSpeaker ResNet34-LM + VBx, ONNX Runtime CUDA. Whole-file, no VAD. |
| `POST /v1/audio/speech/embedding` | WeSpeaker ResNet34-LM, 256-d |
| `GET /health`, `GET /v1/models` | |

Request/response shapes match speaches (OpenAI `verbose_json` for transcription, `{duration, segments[{start,end,speaker}]}`
for diarization, OpenAI embeddings list for speaker embeddings), so MeetScribe needs no changes.

## Run (Docker, NVIDIA GPU)

```bash
docker compose up -d --build
curl -fsS http://127.0.0.1:8000/health
```

Models are read from / downloaded into the external volume `hf-hub-cache`
(`/home/ubuntu/.cache/huggingface/hub`). The container fails fast at startup if CUDA is not
usable by both CTranslate2 and ONNX Runtime (`REQUIRE_GPU=true`).

## Behaviour worth knowing

- **GPU concurrency** is bounded per task (`MAX_CONCURRENT_*`); extra requests queue. A queued request
  whose client has already disconnected (MeetScribe times out after 600 s and retries) is dropped with
  HTTP 499 instead of being processed for nobody.
- **GPU memory stays flat.** ONNX Runtime sessions run with `arena_extend_strategy=kSameAsRequested` and
  arena shrinkage after every run, so the process sits at roughly the model weights (~2 GB on an RTX 4080)
  between requests instead of accumulating reserved memory until CUDA OOM.
- **Transcription segments follow pauses.** VAD speech chunks are grouped into clips of at most
  `STT_VAD_MAX_SPEECH_S` seconds of *timeline* (faster-whisper 1.1 / speaches semantics), and each clip is
  one output segment, so long silences never end up inside a segment.
- **Diarization is VAD-free** and works on the whole file, exactly like the speaches endpoint did.

## Transcribing a track as clips

`POST /v1/audio/transcriptions/clips` takes one track and a list of clips and decodes them in GPU
batches. It replaces one upload per chunk: one decode, one queue slot and real batching per track.

Form fields: `file`, `clips` (JSON list of `{"start": s, "end": s, "speaker": "..."}`, seconds on
the track timeline), and optionally `model`, `language`, `pad_ms` (0..2000), `vad`, `temperature`,
`prompt`, `hotwords`.

```json
{
  "task": "transcribe", "language": "ru", "duration": 2066.1, "clips": 249,
  "segments": [{"clip_index": 0, "speaker": "A", "start": 22.9, "end": 28.1, "text": "...",
                "avg_logprob": -0.21, "no_speech_prob": 0.02, "compression_ratio": 1.4,
                "id": 1, "seek": 2295, "tokens": [], "temperature": 0.0}],
  "failed_clips": [{"clip_index": 6, "reason": "empty"}]
}
```

- Segments are matched to clips by `clip_index` and sorted by it, then by `start`.
- A clip of up to 30 s gives at most one segment. A longer clip is split on pauses and gives several.
- VAD runs on every clip separately and trims it to the speech inside, as for a separate upload,
  so the text matches the per-chunk path (measured on two sessions: 616 of 616 segments kept,
  12454 vs 12456 words).
- A clip in neither list was decoded and produced no text.
- `failed_clips` reasons: `out_of_range`, `empty` (shorter than 50 ms), `no_speech` (only with
  `STT_VAD_EMPTY_FALLBACK=false`), `error: <Exception>` (decoding broke; everything decoded before
  it is still returned, only these are worth retrying).
- If the client disconnects, decoding stops and the request ends with 499.

Measured on an RTX 4080 with the medium model, 677 clips, 84 minutes of speech:

| Path | Real-time factor | Host GPU memory peak |
|---|---|---|
| one upload per chunk, 3 in flight | 0.031 | not measured |
| clips, `STT_BATCH_SIZE=8` | 0.020 | not measured |
| clips, `STT_BATCH_SIZE=16` | 0.013 | 9.9 GB |
| clips, `STT_BATCH_SIZE=32` | 0.012 | 12.3 GB |

The memory figures are for the whole card, which held about 4.4 GB before the run. The default stays
at 8: two transcriptions and a diarization may run at once on a 16 GB card.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `STT_MODEL` | `Systran/faster-whisper-medium` | Whisper model loaded at startup |
| `STT_ALLOWED_MODELS` | empty | Comma-separated model ids a request may name besides `STT_MODEL`. Naming one swaps it in (one Whisper model stays resident); any other id gets 404 |
| `STT_VAD_EMPTY_FALLBACK` | `true` | If VAD hears nothing in an upload or clip of at most 30 s, transcribe it whole instead of returning nothing |
| `STT_CLIP_PAD_MS` | `0` | Default padding around clips of `/v1/audio/transcriptions/clips` |
| `STT_COMPUTE_TYPE` | `default` | CTranslate2 compute type (`float16`, `int8_float16`, ...) |
| `STT_DEVICE` / `STT_DEVICE_INDEX` | `auto` / `0` | |
| `STT_NUM_WORKERS`, `STT_BATCH_SIZE`, `STT_BEAM_SIZE` | `1`, `8`, `5` | |
| `STT_VAD_FILTER` | `true` | Silero VAD before decoding (`STT_VAD_MIN_SILENCE_MS=160`, `STT_VAD_MAX_SPEECH_S=30`) |
| `SEG_MODEL` / `EMB_MODEL` | `fedirz/segmentation_community_1` / `Wespeaker/wespeaker-voxceleb-resnet34-LM` | Diarization + embedding models (ONNX) |
| `DIARIZATION_EMBEDDING_BATCH_SIZE` | `32` | |
| `MAX_CONCURRENT_STT` / `MAX_CONCURRENT_DIARIZATION` / `MAX_CONCURRENT_EMBEDDING` | `2` / `1` / `2` | GPU concurrency limits per task; extra requests wait |
| `REQUIRE_GPU` | `true` | Refuse to start on CPU |
| `PRELOAD` | `true` | Load all models at startup |
| `LOG_LEVEL` | `info` | |

## Benchmarks

- `bench/replay_sessions.py` replays stored MeetScribe sessions and compares with the stored
  transcripts. `--model` picks the Whisper model, `--clips` uses the clips endpoint, `--pad-ms`
  sets its padding. It shows differences, not correctness: the stored text is machine output.
- `bench/groundtruth/` measures WER against public Russian data with human references. See its
  README. Change the model or decoding settings only after a run there.

Results of both contain audio-derived text and are git-ignored.

## Development

```bash
uv sync                     # CPU onnxruntime on Windows/macOS, GPU build on Linux
uv run pytest               # unit tests, no models
uv run pytest -m integration   # downloads faster-whisper-tiny + the two ONNX models, runs on CPU
uv run ruff check src tests && uv run mypy src
```
