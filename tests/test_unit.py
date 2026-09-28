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
    res = build_verbose_response(segments, "ru", 3.0, word_timestamps=False)  # type: ignore[arg-type]
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

    with_words = build_verbose_response(segments, "ru", 3.0, word_timestamps=True)  # type: ignore[arg-type]
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


# --- clip planning (/v1/audio/transcriptions/clips) ---


def test_upload_clips_falls_back_to_whole_short_upload() -> None:
    from meetscribe_server.routers.stt import upload_clips

    assert upload_clips([], 4.0, 30.0, empty_fallback=True) == ([{"start": 0.0, "end": 4.0}], True)
    assert upload_clips([], 4.0, 30.0, empty_fallback=False) == ([], False)
    assert upload_clips([], 45.0, 30.0, empty_fallback=True) == ([], False)  # long silence stays silent
    clips, fallback = upload_clips([{"start": 16000, "end": 32000}], 4.0, 30.0, empty_fallback=True)
    assert clips == [{"start": 1.0, "end": 2.0}] and fallback is False


def test_parse_clips_validation() -> None:
    from fastapi import HTTPException

    from meetscribe_server.routers.stt import parse_clips

    clips = parse_clips('[{"start": 1, "end": 2.5, "speaker": "A"}, {"start": 3, "end": 4}]')
    assert [(c.index, c.start, c.end, c.speaker) for c in clips] == [(0, 1.0, 2.5, "A"), (1, 3.0, 4.0, None)]
    for bad in ["nope", "[]", "{}", "[1]", '[{"start": "a", "end": 2}]']:
        try:
            parse_clips(bad)
        except HTTPException as e:
            assert e.status_code == 422
        else:
            raise AssertionError(f"expected 422 for {bad}")


def test_plan_clips_padding_vad_and_unique_seek() -> None:
    from meetscribe_server.routers.stt import ClipRequest, plan_clips, seek_of, speech_lookup

    reqs = [
        ClipRequest(0, 10.0, 12.0, "A"),
        ClipRequest(1, 10.0, 13.0, "B"),  # same start as clip 0: overlapping speakers
        ClipRequest(2, 50.0, 80.0, "A"),  # already 30 s: padding must not grow it
        ClipRequest(3, 200.0, 205.0, "A"),  # beyond the end of the file
        ClipRequest(4, 99.5, 101.0, "A"),  # clamped to duration
    ]
    planned, failed = plan_clips(reqs, duration_s=100.0, pad_s=0.2, find_speech=None, empty_fallback=True)
    spans = {req.index: clip for req, clip in planned}
    assert abs(spans[0]["start"] - 9.8) < 1e-9 and abs(spans[0]["end"] - 12.2) < 1e-9
    assert spans[2] == {"start": 50.0, "end": 80.0}
    assert spans[4]["end"] == 100.0
    assert [(f.clip_index, f.reason) for f in failed] == [(3, "out_of_range")]
    seeks = [seek_of(clip["start"]) for _, clip in planned]
    assert len(set(seeks)) == len(seeks)

    speech = speech_lookup([{"start": int(10.5 * 16000), "end": int(11.5 * 16000)}])
    planned, failed = plan_clips(reqs[:1] + reqs[2:3], 100.0, 0.0, speech, empty_fallback=False)
    assert [(req.index, clip) for req, clip in planned] == [(0, {"start": 10.5, "end": 11.5})]
    assert [(f.clip_index, f.reason) for f in failed] == [(2, "no_speech")]
    planned, failed = plan_clips(reqs[2:3], 100.0, 0.0, speech, empty_fallback=True)
    assert planned[0][1] == {"start": 50.0, "end": 80.0} and failed == []


def test_model_outside_allow_list_is_404(wav_bytes: bytes) -> None:
    settings = Settings(require_gpu=False, preload=False, log_level="warning", stt_allowed_models="a/b, c/d")
    assert settings.allowed_stt_models() == {"Systran/faster-whisper-medium", "a/b", "c/d"}
    with TestClient(create_app(settings)) as client:
        for path, extra in (
            ("/v1/audio/transcriptions", {}),
            ("/v1/audio/transcriptions/clips", {"clips": '[{"start": 0, "end": 1}]'}),
        ):
            r = client.post(path, files={"file": ("a.wav", wav_bytes, "audio/wav")}, data={"model": "x/y", **extra})
            assert r.status_code == 404, path
            assert "not enabled" in r.json()["detail"]


def test_plan_clips_splits_clips_longer_than_whisper_window() -> None:
    from meetscribe_server.routers.stt import ClipRequest, plan_clips, speech_lookup

    sr = 16000
    chunks = [
        {"start": 10 * sr, "end": 35 * sr},
        {"start": 36 * sr, "end": 60 * sr},
        {"start": 62 * sr, "end": 70 * sr},
    ]
    speech = speech_lookup(chunks)
    planned, failed = plan_clips([ClipRequest(0, 5.0, 75.0, "A")], 100.0, 0.2, speech, empty_fallback=True)
    assert failed == []
    assert [clip for _, clip in planned] == [
        {"start": 10.0, "end": 35.0},
        {"start": 36.0, "end": 60.0},
        {"start": 62.0, "end": 70.0},
    ]
    assert all(req.index == 0 for req, _ in planned)

    planned, failed = plan_clips([ClipRequest(0, 5.0, 75.0, "A")], 100.0, 0.0, None, empty_fallback=True)
    assert [clip for _, clip in planned] == [
        {"start": 5.0, "end": 35.0},
        {"start": 35.0, "end": 65.0},
        {"start": 65.0, "end": 75.0},
    ]
