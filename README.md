# meetscribe-server

GPU speech backend for [MeetScribe](https://github.com/koekaverna/meetscribe): Whisper
transcription, speaker diarization and speaker embeddings behind a small HTTP API.

It started as a replacement for [speaches](https://github.com/speaches-ai/speaches) and keeps the
request and response shapes of the endpoints MeetScribe used there, so it is a drop-in backend.
Everything MeetScribe does not call is left out: the image has no PyTorch, no TTS and no UI.

| Endpoint | What it does | Engine |
|---|---|---|
| `POST /v1/audio/transcriptions` | Transcribe one upload | faster-whisper 1.2 on CTranslate2, batched decoding, Silero VAD |
| `POST /v1/audio/transcriptions/clips` | Transcribe many clips of one track in GPU batches | same |
| `POST /v1/audio/diarization` | Who spoke when, whole file, no VAD | onnx-diarization: pyannote `segmentation_community_1`, WeSpeaker ResNet34-LM, VBx clustering |
| `POST /v1/audio/speech/embedding` | 256-d speaker embedding of an upload | WeSpeaker ResNet34-LM |
| `GET /health`, `GET /v1/models` | Liveness and loaded models | |

Interactive API docs are served at `/docs`.

## Requirements

- NVIDIA GPU with a recent driver and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  About 3 GB of GPU memory with the default models.
- Docker with Compose.

CPU-only runs work for development and tests (`REQUIRE_GPU=false`), not for real workloads.

## Run

```bash
docker volume create hf-hub-cache
docker compose up -d --build
curl -fsS http://127.0.0.1:8000/health
```

Models are downloaded from Hugging Face on first start into the `hf-hub-cache` volume, about
2 GB. The container refuses to start if CTranslate2 or ONNX Runtime cannot use CUDA, instead of
silently falling back to CPU.

The server has **no authentication**. `compose.yaml` binds it to `127.0.0.1`. Put it behind a
reverse proxy with access control before exposing it to a network.

## API

All endpoints take `multipart/form-data` with the audio in `file`. Any format FFmpeg can decode
is accepted; audio is resampled to 16 kHz mono.

### Transcription

```bash
curl -F file=@meeting.wav -F language=ru -F response_format=verbose_json \
  http://127.0.0.1:8000/v1/audio/transcriptions
```

Fields: `model`, `language`, `response_format` (`json`, `text`, `verbose_json`), `prompt`,
`hotwords`, `temperature`, `timestamp_granularities[]` (`segment`, `word`), `vad`.

`verbose_json` follows the OpenAI shape. Every segment carries `avg_logprob`, `no_speech_prob`
and `compression_ratio`.

### Transcription of clips

One track plus a list of clips, for clients that already know where speech is, for example from
diarization. Compared with one upload per clip this decodes the file once, takes one queue slot
and lets the GPU batch.

```bash
curl -F file=@track.wav -F language=ru \
  -F 'clips=[{"start": 22.9, "end": 28.1, "speaker": "Anna"}, {"start": 30.0, "end": 41.5, "speaker": "Boris"}]' \
  http://127.0.0.1:8000/v1/audio/transcriptions/clips
```

Fields: `clips` (JSON list, seconds on the track timeline, `speaker` optional), `model`,
`language`, `pad_ms` (0..2000), `vad`, `prompt`, `hotwords`, `temperature`.

```json
{
  "task": "transcribe", "language": "ru", "duration": 2066.1, "clips": 2,
  "segments": [
    {"clip_index": 0, "speaker": "Anna", "start": 22.9, "end": 28.1, "text": "...",
     "avg_logprob": -0.21, "no_speech_prob": 0.02, "compression_ratio": 1.4,
     "id": 1, "seek": 2290, "tokens": [50365], "temperature": 0.0}
  ],
  "failed_clips": [{"clip_index": 1, "reason": "error: RuntimeError"}]
}
```

- Match segments to clips by `clip_index`. Segments are sorted by it, then by `start`.
- A clip of up to 30 s gives at most one segment. A longer clip is split on pauses and gives
  several with the same `clip_index`.
- VAD runs on every clip separately and trims it to the speech inside, so the text is the same
  as for a separately uploaded clip.
- A clip that appears in neither list was decoded and produced no text.
- `failed_clips` reasons: `out_of_range`, `empty` (shorter than 50 ms), `no_speech` (only with
  `STT_VAD_EMPTY_FALLBACK=false`) and `error: <Exception>`. Only the last one is worth retrying.
  Clips decoded before an error are still returned.

### Diarization

```bash
curl -F file=@meeting.wav http://127.0.0.1:8000/v1/audio/diarization
```

Returns `{"duration": 2066.1, "segments": [{"start": 0.5, "end": 4.2, "speaker": "SPEAKER_00"}]}`.
Optional fields: `min_speakers`, `max_speakers`, `response_format` (`json`, `rttm`).

### Speaker embedding

```bash
curl -F file=@sample.wav http://127.0.0.1:8000/v1/audio/speech/embedding
```

Returns an OpenAI embeddings list with one 256-d vector. Compare vectors with cosine similarity.

## Behaviour worth knowing

- **Requests queue, they do not pile up on the GPU.** Concurrency is bounded per task
  (`MAX_CONCURRENT_*`). A queued request whose client has already disconnected is dropped with
  HTTP 499 instead of being processed for nobody.
- **GPU memory stays flat.** ONNX Runtime's arena never shrinks by itself; here it is shrunk after
  every run. Without this the process grew to the full 16 GB of the card over a few days and then
  failed with CUDA out-of-memory.
- **Segments follow pauses.** VAD speech chunks are grouped into clips of at most 30 s of
  timeline, and each clip is one segment, so a long silence never ends up inside a segment.
- **Diarization sees the whole file.** There is no VAD in front of the segmentation model.
- **Known Whisper hallucinations are dropped.** On silence and noise Whisper emits subtitle
  credits and video sign-offs. With large-v3-turbo these have `no_speech_prob` 0.0 and an ordinary
  `avg_logprob`, so only the text gives them away. The phrase list is Russian and deliberately
  narrow; see [`hallucinations.py`](src/meetscribe_server/hallucinations.py).
- **One Whisper model is resident.** A request may name another model only if it is listed in
  `STT_ALLOWED_MODELS`; naming it swaps the resident model and stalls concurrent requests for the
  load time.
- **Quiet speech is not lost to VAD.** If VAD hears nothing in an upload or clip of at most 30 s,
  it is transcribed whole.

## Configuration

Environment variables, also read from `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `STT_MODEL` | `Systran/faster-whisper-medium` | Whisper model loaded at startup, any CTranslate2 conversion on Hugging Face. `compose.yaml` sets `deepdml/faster-whisper-large-v3-turbo-ct2` |
| `STT_ALLOWED_MODELS` | empty | Comma-separated model ids a request may name besides `STT_MODEL`. Any other id gets 404 |
| `STT_COMPUTE_TYPE` | `default` | CTranslate2 compute type: `float16`, `int8_float16`, ... |
| `STT_DEVICE`, `STT_DEVICE_INDEX` | `auto`, `0` | |
| `STT_BATCH_SIZE` | `8` | Clips decoded together. 16 is about 1.5x faster and needs several GB more GPU memory |
| `STT_BEAM_SIZE` | `5` | |
| `STT_NUM_WORKERS`, `STT_CPU_THREADS` | `1`, `0` | |
| `STT_VAD_FILTER` | `true` | Silero VAD before decoding. A request can override it with `vad` |
| `STT_VAD_MIN_SILENCE_MS`, `STT_VAD_MAX_SPEECH_S` | `160`, `30` | |
| `STT_VAD_EMPTY_FALLBACK` | `true` | Transcribe a short upload or clip whole when VAD hears nothing in it |
| `STT_DROP_KNOWN_HALLUCINATIONS` | `true` | Drop segments that consist only of a known hallucinated phrase |
| `STT_CLIP_PAD_MS` | `0` | Default padding around clips of the clips endpoint |
| `SEG_MODEL`, `SEG_MODEL_FILE` | `fedirz/segmentation_community_1`, `model.onnx` | Segmentation model for diarization |
| `EMB_MODEL`, `EMB_MODEL_FILE` | `Wespeaker/wespeaker-voxceleb-resnet34-LM`, `voxceleb_resnet34_LM.onnx` | Speaker embedding model |
| `DIARIZATION_EMBEDDING_BATCH_SIZE` | `32` | |
| `MAX_CONCURRENT_STT`, `MAX_CONCURRENT_DIARIZATION`, `MAX_CONCURRENT_EMBEDDING` | `2`, `1`, `2` | Requests of each kind that may use the GPU at once |
| `REQUIRE_GPU` | `true` | Refuse to start without CUDA |
| `PRELOAD` | `true` | Load all models at startup |
| `LOG_LEVEL` | `info` | |

## Benchmarks

Measured results and the reasoning behind the defaults are in [docs/benchmarks.md](docs/benchmarks.md).
In short, for Russian speech: large-v3-turbo has 3 points lower WER than medium and is 1.7x
faster; beam size, compute type and sequential decoding make no measurable difference; a global
prompt or hotword list makes things worse.

- [`bench/groundtruth`](bench/groundtruth) measures WER against public Russian speech with human
  references.
- [`bench/replay_sessions.py`](bench/replay_sessions.py) replays sessions stored by MeetScribe
  through a server and compares with the stored transcripts. It shows what a change does to real
  meetings, not whether the result is correct.

## Development

```bash
uv sync                          # CPU onnxruntime on Windows and macOS, GPU build on Linux
uv run pytest                    # unit tests, no models needed
uv run pytest -m integration     # downloads faster-whisper-tiny and the ONNX models, runs on CPU
uv run ruff check . && uv run ruff format --check . && uv run mypy src
REQUIRE_GPU=false uv run uvicorn --factory meetscribe_server.main:create_app --port 8000
```

## License

[MIT](LICENSE).

Third-party components have their own terms: the Whisper models, the pyannote segmentation model
and the WeSpeaker model are downloaded from Hugging Face at runtime under the licenses stated on
their model pages. The `onnx-diarization` package is published on PyPI without a license
statement; check with its author before redistributing it.
