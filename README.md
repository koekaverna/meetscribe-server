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

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `STT_MODEL` | `Systran/faster-whisper-medium` | Whisper model loaded at startup; a request naming another CT2 Whisper model swaps it in |
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

## Development

```bash
uv sync                     # CPU onnxruntime on Windows/macOS, GPU build on Linux
uv run pytest               # unit tests, no models
uv run pytest -m integration   # downloads faster-whisper-tiny + the two ONNX models, runs on CPU
uv run ruff check src tests && uv run mypy src
```
