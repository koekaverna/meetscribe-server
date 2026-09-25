"""Integration tests against real models (downloaded from Hugging Face, run on CPU).

uv run pytest -m integration
"""

import math

import pytest
from fastapi.testclient import TestClient

from meetscribe_server.config import Settings
from meetscribe_server.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    settings = Settings(
        stt_model="Systran/faster-whisper-tiny",
        stt_device="cpu",
        stt_compute_type="int8",
        require_gpu=False,
        preload=True,
        log_level="info",
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_models_loaded(client: TestClient) -> None:
    models = client.get("/v1/models").json()["data"]
    assert all(m["loaded"] for m in models), models


def test_transcription_verbose_json(client: TestClient, wav_bytes: bytes) -> None:
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
        data={
            "model": "Systran/faster-whisper-tiny",
            "language": "ru",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["language"] == "ru"
    assert body["duration"] == pytest.approx(3.0, abs=0.1)
    assert isinstance(body["segments"], list)
    for seg in body["segments"]:
        assert {"start", "end", "text", "no_speech_prob", "avg_logprob"} <= set(seg)


def test_embedding_is_256d_unit_like(client: TestClient, wav_bytes: bytes) -> None:
    r = client.post(
        "/v1/audio/speech/embedding",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
        data={"model": "Wespeaker/wespeaker-voxceleb-resnet34-LM"},
    )
    assert r.status_code == 200, r.text
    emb = r.json()["data"][0]["embedding"]
    assert len(emb) == 256
    assert all(math.isfinite(x) for x in emb)
    # Same input twice -> identical vector (deterministic ONNX inference)
    r2 = client.post(
        "/v1/audio/speech/embedding",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
        data={"model": "Wespeaker/wespeaker-voxceleb-resnet34-LM"},
    )
    assert r2.json()["data"][0]["embedding"] == pytest.approx(emb, abs=1e-5)


def test_diarization_shape(client: TestClient, wav_bytes: bytes) -> None:
    r = client.post(
        "/v1/audio/diarization",
        files={"file": ("a.wav", wav_bytes, "audio/wav")},
        data={"model": "fedirz/segmentation_community_1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["duration"] == pytest.approx(3.0, abs=0.1)
    for seg in body["segments"]:
        assert seg["speaker"].startswith("SPEAKER_")
        # the last 10 s segmentation window is zero-padded, so ends may slightly overshoot the audio
        assert 0 <= seg["start"] <= seg["end"] <= body["duration"] + 0.5
