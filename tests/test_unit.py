"""Unit tests: response shaping and the app surface that needs no models."""

import io
from types import SimpleNamespace

import numpy as np
from fastapi import UploadFile
from fastapi.testclient import TestClient
from pyannote.core import Annotation, Segment

from meetscribe_server.audio import decode_upload
from meetscribe_server.config import Settings
from meetscribe_server.main import create_app
from meetscribe_server.routers.diarization import annotation_to_segments
from meetscribe_server.routers.stt import build_verbose_response, segments_to_text
from meetscribe_server.schemas import CreateEmbeddingResponse, EmbeddingObject


def _settings() -> Settings:
    return Settings(require_gpu=False, preload=False, log_level="warning")


def test_decode_upload_wav(wav_bytes: bytes) -> None:
    upload = UploadFile(file=io.BytesIO(wav_bytes), filename="a.wav")
    data = decode_upload(upload)
    assert data.dtype == np.float32
    assert data.ndim == 1
    assert data.size == 3 * 16000


def test_decode_upload_rejects_garbage() -> None:
    from fastapi import HTTPException

    upload = UploadFile(file=io.BytesIO(b"not audio at all"), filename="a.wav")
    try:
        decode_upload(upload)
    except HTTPException as e:
        assert e.status_code == 400
    else:
        raise AssertionError("expected HTTPException")


def _segment(i: int, start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=i,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=[1, 2],
        temperature=None,
        avg_logprob=-0.2,
        compression_ratio=1.1,
        no_speech_prob=0.05,
        words=[SimpleNamespace(start=start, end=end, word=text.strip())],
    )


def test_verbose_response_matches_openai_shape() -> None:
    segments = [_segment(0, 0.0, 1.5, " Привет"), _segment(1, 1.5, 3.0, " мир")]
    info = SimpleNamespace(language="ru", duration=3.0)
    res = build_verbose_response(segments, info, word_timestamps=False)  # type: ignore[arg-type]
    body = res.model_dump()
    assert body["text"] == "Привет мир"
    assert body["language"] == "ru"
    assert body["duration"] == 3.0
    assert body["words"] is None
    seg = body["segments"][0]
    assert set(seg) == {
        "id",
        "seek",
        "start",
        "end",
        "text",
        "tokens",
        "temperature",
        "avg_logprob",
        "compression_ratio",
        "no_speech_prob",
    }
    assert seg["temperature"] == 0.0  # None from the batched pipeline is normalised
    assert seg["no_speech_prob"] == 0.05
    assert segments_to_text(segments) == "Привет мир"  # type: ignore[arg-type]

    with_words = build_verbose_response(segments, info, word_timestamps=True)  # type: ignore[arg-type]
    assert [w.word for w in with_words.words or []] == ["Привет", "мир"]


def test_annotation_to_segments_first_label_wins() -> None:
    ann = Annotation()
    ann[Segment(0.0, 2.0)] = "SPEAKER_00"
    ann[Segment(2.0, 4.0)] = "SPEAKER_01"
    ann[Segment(3.0, 5.0)] = "SPEAKER_00"  # overlap
    segs = annotation_to_segments(ann)
    # itersegments() yields the stored (possibly overlapping) segments in time order, as speaches did
    assert [(s.start, s.end, s.speaker) for s in segs] == [
        (0.0, 2.0, "SPEAKER_00"),
        (2.0, 4.0, "SPEAKER_01"),
        (3.0, 5.0, "SPEAKER_00"),
    ]


def test_embedding_response_shape() -> None:
    body = CreateEmbeddingResponse(data=[EmbeddingObject(embedding=[0.1] * 256)], model="m").model_dump()
    assert body["object"] == "list"
    assert body["data"][0]["object"] == "embedding"
    assert len(body["data"][0]["embedding"]) == 256
    assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


def test_health_and_models_without_loading() -> None:
    with TestClient(create_app(_settings())) as client:
        assert client.get("/health").json() == {"message": "OK"}
        models = client.get("/v1/models").json()
        assert [m["task"] for m in models["data"]] == [
            "automatic-speech-recognition",
            "speaker-segmentation",
            "speaker-embedding",
        ]
        assert all(m["loaded"] is False for m in models["data"])


def test_unknown_diarization_model_is_404(wav_bytes: bytes) -> None:
    with TestClient(create_app(_settings())) as client:
        r = client.post(
            "/v1/audio/diarization",
            files={"file": ("a.wav", wav_bytes, "audio/wav")},
            data={"model": "somebody/else"},
        )
        assert r.status_code == 404
        r = client.post(
            "/v1/audio/speech/embedding",
            files={"file": ("a.wav", wav_bytes, "audio/wav")},
            data={"model": "somebody/else"},
        )
        assert r.status_code == 404


def test_stream_is_rejected(wav_bytes: bytes) -> None:
    with TestClient(create_app(_settings())) as client:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", wav_bytes, "audio/wav")},
            data={"model": "x", "stream": "true"},
        )
        assert r.status_code == 400
