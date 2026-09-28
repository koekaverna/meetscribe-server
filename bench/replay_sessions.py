"""Replay stored MeetScribe sessions through a backend and compare with the transcripts in the DB.

The stored `session_segments` were produced by the previous backend through the real MeetScribe
pipeline. This script reproduces that pipeline (diarization -> speaker identification -> chunking ->
transcription -> hallucination filter, or whole-file transcription for named tracks) against a
backend URL and reports text agreement (WER), speaker agreement, segment counts and speed.

    uv run --with jiwer python bench/replay_sessions.py --server http://127.0.0.1:8000 \
        --db E:/meetscribe/data/meetscribe.db --data E:/meetscribe/data SESSION_ID [SESSION_ID ...]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sqlite3
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import jiwer

# --- MeetScribe pipeline parameters (data/config.yaml + config.py defaults) ---
DIAR_MODEL = "fedirz/segmentation_community_1"
EMB_MODEL = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
STT_MODEL = "Systran/faster-whisper-medium"
LANGUAGE = "ru"
MAX_GAP_MS = 2000
MAX_CHUNK_MS = 30000
NO_SPEECH_PROB_THRESHOLD = 0.5
AVG_LOGPROB_THRESHOLD = -0.25
EMB_THRESHOLD = 0.6
EMB_MIN_THRESHOLD = 0.45
EMB_CONFIDENT_GAP = 0.2
EMB_MIN_DURATION_MS = 1500
MAX_INFLIGHT = 3


@dataclass
class Seg:
    start_ms: int
    end_ms: int
    speaker: str | None = None
    text: str = ""

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class Timing:
    diar_audio_s: float = 0.0
    diar_wall_s: float = 0.0
    emb_calls: int = 0
    emb_wall_s: float = 0.0
    stt_chunk_audio_s: float = 0.0
    stt_chunk_wall_s: float = 0.0  # sum of per-request latencies
    stt_chunk_elapsed_s: float = 0.0  # wall clock with MAX_INFLIGHT workers
    stt_file_audio_s: float = 0.0
    stt_file_wall_s: float = 0.0
    chunks: int = 0
    filtered_segments: int = 0
    kept_segments: int = 0


# --- helpers copied from meetscribe.pipeline (same semantics) ---


def merge_close_segments(segments: list[Seg], max_gap_ms: int, max_chunk_ms: int) -> list[Seg]:
    if not segments:
        return []
    merged: list[Seg] = []
    cur = Seg(segments[0].start_ms, segments[0].end_ms, segments[0].speaker)
    for seg in segments[1:]:
        gap = seg.start_ms - cur.end_ms
        duration = seg.end_ms - cur.start_ms
        if gap <= max_gap_ms and duration <= max_chunk_ms and seg.speaker == cur.speaker:
            cur.end_ms = seg.end_ms
        else:
            merged.append(cur)
            cur = Seg(seg.start_ms, seg.end_ms, seg.speaker)
    merged.append(cur)
    return merged


def find_speaker(start_ms: int, end_ms: int, segments: list[Seg]) -> str:
    best, best_overlap = "Unknown", 0
    for seg in segments:
        overlap = max(0, min(end_ms, seg.end_ms) - max(start_ms, seg.start_ms))
        if overlap > best_overlap:
            best_overlap, best = overlap, seg.speaker or "Unknown"
    return best


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def identify(embedding: list[float], voiceprints: dict[str, list[float]]) -> tuple[str | None, float]:
    if not voiceprints:
        return None, -1.0
    scores = sorted(((cosine(embedding, vp), name) for name, vp in voiceprints.items()), reverse=True)
    best_sim, best_name = scores[0]
    second = scores[1][0] if len(scores) > 1 else -1.0
    if best_sim >= EMB_THRESHOLD:
        return best_name, best_sim
    if best_sim >= EMB_MIN_THRESHOLD and best_sim - second >= EMB_CONFIDENT_GAP:
        return best_name, best_sim
    return None, best_sim


class Wav:
    def __init__(self, path: Path) -> None:
        with wave.open(str(path), "rb") as wf:
            self.rate = wf.getframerate()
            self.width = wf.getsampwidth()
            self.channels = wf.getnchannels()
            self.frames = wf.readframes(wf.getnframes())
        self.duration_ms = len(self.frames) * 1000 // (self.rate * self.width * self.channels)

    def slice(self, start_ms: int, end_ms: int) -> bytes:
        fs = self.width * self.channels
        a, b = start_ms * self.rate // 1000 * fs, end_ms * self.rate // 1000 * fs
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.width)
            wf.setframerate(self.rate)
            wf.writeframes(self.frames[a:b])
        return buf.getvalue()


# --- backend client ---


class Backend:
    def __init__(self, url: str, model: str = STT_MODEL, clips_endpoint: bool = False, pad_ms: int = 0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.clips_endpoint = clips_endpoint
        self.pad_ms = pad_ms
        self.client = httpx.Client(timeout=3600)

    def post(self, path: str, name: str, data: bytes, form: dict) -> tuple[dict, float]:
        t0 = time.perf_counter()
        r = self.client.post(f"{self.url}{path}", files={"file": (name, data, "audio/wav")}, data=form)
        r.raise_for_status()
        return r.json(), time.perf_counter() - t0

    def diarize(self, name: str, data: bytes) -> tuple[list[Seg], float]:
        body, dt = self.post("/v1/audio/diarization", name, data, {"model": DIAR_MODEL})
        return [Seg(int(s["start"] * 1000), int(s["end"] * 1000), s["speaker"]) for s in body["segments"]], dt

    def embed(self, data: bytes) -> tuple[list[float], float]:
        body, dt = self.post("/v1/audio/speech/embedding", "audio.wav", data, {"model": EMB_MODEL})
        return [float(x) for x in body["data"][0]["embedding"]], dt

    def transcribe(self, name: str, data: bytes, timing: Timing) -> tuple[list[Seg], float]:
        form = {
            "model": self.model,
            "language": LANGUAGE,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        body, dt = self.post("/v1/audio/transcriptions", name, data, form)
        return self.keep(body.get("segments", []), timing), dt

    def transcribe_clips(self, name: str, data: bytes, chunks: list[Seg], timing: Timing) -> tuple[list[Seg], float]:
        """One request per track: the whole file plus the chunk list (new contract)."""
        clips = [{"start": c.start_ms / 1000, "end": c.end_ms / 1000, "speaker": c.speaker} for c in chunks]
        form = {"model": self.model, "language": LANGUAGE, "clips": json.dumps(clips), "pad_ms": str(self.pad_ms)}
        body, dt = self.post("/v1/audio/transcriptions/clips", name, data, form)
        if body.get("failed_clips"):
            print(f"      failed clips: {body['failed_clips'][:5]}")
        return self.keep(body.get("segments", []), timing), dt

    @staticmethod
    def keep(segments: list[dict], timing: Timing) -> list[Seg]:
        """MeetScribe's hallucination filter."""
        out: list[Seg] = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            if (
                seg.get("no_speech_prob", 0.0) >= NO_SPEECH_PROB_THRESHOLD
                and seg.get("avg_logprob", 0.0) <= AVG_LOGPROB_THRESHOLD
            ):
                timing.filtered_segments += 1
                continue
            timing.kept_segments += 1
            out.append(Seg(int(seg["start"] * 1000), int(seg["end"] * 1000), None, text))
        return out


# --- pipeline replay ---


def map_clusters(backend: Backend, wav: Wav, segments: list[Seg], voiceprints: dict, timing: Timing) -> dict[str, str]:
    clusters: dict[str, list[Seg]] = {}
    for s in segments:
        if s.speaker is not None:
            clusters.setdefault(s.speaker, []).append(s)
    mapping: dict[str, str] = {}
    unknown = 0
    for label, segs in clusters.items():
        candidates = [s for s in segs if 3000 <= s.duration_ms <= 12000] or [
            s for s in segs if s.duration_ms >= EMB_MIN_DURATION_MS
        ]
        if not candidates:
            unknown += 1
            mapping[label] = f"Unknown-{unknown}"
            continue
        rep = max(candidates, key=lambda s: s.duration_ms)
        emb, dt = backend.embed(wav.slice(rep.start_ms, rep.end_ms))
        timing.emb_calls += 1
        timing.emb_wall_s += dt
        name, _ = identify(emb, voiceprints)
        if name is None:
            unknown += 1
            name = f"Unknown-{unknown}"
        mapping[label] = name
    return mapping


def replay_track(
    backend: Backend, path: Path, speaker_name: str | None, open_space: bool, voiceprints: dict, timing: Timing
) -> list[Seg]:
    wav = Wav(path)
    if speaker_name and not open_space:
        segs, dt = backend.transcribe(path.name, path.read_bytes(), timing)
        timing.stt_file_audio_s += wav.duration_ms / 1000
        timing.stt_file_wall_s += dt
        for s in segs:
            s.speaker = speaker_name
        return segs

    segments, dt = backend.diarize(path.name, path.read_bytes())
    timing.diar_audio_s += wav.duration_ms / 1000
    timing.diar_wall_s += dt
    if not segments:
        return []
    mapping = map_clusters(backend, wav, segments, voiceprints, timing)
    for s in segments:
        s.speaker = mapping.get(s.speaker or "", s.speaker)
    if open_space and speaker_name:
        segments = [s for s in segments if s.speaker == speaker_name]
        if not segments:
            return []

    merged = merge_close_segments(segments, MAX_GAP_MS, MAX_CHUNK_MS)
    timing.chunks += len(merged)
    if backend.clips_endpoint:
        t0 = time.perf_counter()
        segs, dt = backend.transcribe_clips(path.name, path.read_bytes(), merged, timing)
        timing.stt_chunk_elapsed_s += time.perf_counter() - t0
        timing.stt_chunk_wall_s += dt
        timing.stt_chunk_audio_s += sum(c.duration_ms for c in merged) / 1000
        for s in segs:
            s.speaker = find_speaker(s.start_ms, s.end_ms, segments)
        return segs
    results: list[list[Seg]] = [[] for _ in merged]

    def work(i: int, chunk: Seg) -> None:
        segs, dt = backend.transcribe("chunk.wav", wav.slice(chunk.start_ms, chunk.end_ms), timing)
        timing.stt_chunk_audio_s += chunk.duration_ms / 1000
        timing.stt_chunk_wall_s += dt
        for s in segs:
            s.start_ms += chunk.start_ms
            s.end_ms += chunk.start_ms
            s.speaker = find_speaker(s.start_ms, s.end_ms, segments)
        results[i] = segs

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as ex:
        list(ex.map(lambda ic: work(*ic), enumerate(merged)))
    timing.stt_chunk_elapsed_s += time.perf_counter() - t0
    return [s for segs in results for s in segs]


# --- comparison ---

_norm = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"ё": "е"}),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def wer(reference: str, hypothesis: str) -> float | None:
    if not reference.strip():
        return None
    if not hypothesis.strip():
        return 1.0
    return jiwer.wer(reference, hypothesis, reference_transform=_norm, hypothesis_transform=_norm)


def canonical_speaker(name: str | None) -> str:
    if not name:
        return "Unknown"
    return "Unknown" if name.startswith("Unknown") else name


def speaker_agreement(stored: list[Seg], new: list[Seg]) -> tuple[float, int]:
    """Share of stored speech time whose best-overlapping new segment carries the same speaker."""
    agree = total = 0
    for s in stored:
        best, best_ov = None, 0
        for n in new:
            ov = max(0, min(s.end_ms, n.end_ms) - max(s.start_ms, n.start_ms))
            if ov > best_ov:
                best, best_ov = n, ov
        total += s.duration_ms
        if best is not None and canonical_speaker(best.speaker) == canonical_speaker(s.speaker):
            agree += s.duration_ms
    return (agree / total if total else 1.0), total


def vram_mb() -> int | None:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | Where-Object { $_.InstanceName -like '*pid_*' } | ForEach-Object { $p=[regex]::Match($_.InstanceName,'pid_(\\d+)').Groups[1].Value; if ((Get-Process -Id $p -ErrorAction SilentlyContinue).ProcessName -eq 'vmwp') { [math]::Round($_.CookedValue/1MB) } }",  # noqa: E501
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return max(vals) if vals else None
    except Exception:
        return None


@dataclass
class TrackResult:
    track_num: int
    mode: str
    audio_s: float
    stored_segments: int
    new_segments: int
    wer: float | None
    speaker_agree: float | None
    diffs: list[dict] = field(default_factory=list)


def compare_track(stored: list[Seg], new: list[Seg]) -> tuple[float | None, float | None, list[dict]]:
    ref_text = " ".join(s.text for s in sorted(stored, key=lambda s: s.start_ms))
    hyp_text = " ".join(s.text for s in sorted(new, key=lambda s: s.start_ms))
    w = wer(ref_text, hyp_text)
    agree, total = speaker_agreement(stored, new)
    return w, (agree if total else None), word_diffs(ref_text, hyp_text)


def word_diffs(ref_text: str, hyp_text: str, context: int = 4) -> list[dict]:
    """Differing word runs (from jiwer's alignment) with a few words of context on each side."""
    if not ref_text.strip() or not hyp_text.strip():
        return []
    out = jiwer.process_words(ref_text, hyp_text, reference_transform=_norm, hypothesis_transform=_norm)
    ref, hyp = out.references[0], out.hypotheses[0]
    diffs: list[dict] = []
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            continue
        diffs.append(
            {
                "type": chunk.type,
                "stored": " ".join(ref[chunk.ref_start_idx : chunk.ref_end_idx]),
                "new": " ".join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx]),
                "context": " ".join(ref[max(0, chunk.ref_start_idx - context) : chunk.ref_start_idx])
                + " [...] "
                + " ".join(ref[chunk.ref_end_idx : chunk.ref_end_idx + context]),
            }
        )
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--db", default="E:/meetscribe/data/meetscribe.db")
    ap.add_argument("--data", default="E:/meetscribe/data")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--model", default=STT_MODEL, help="Whisper model id sent with every transcription request")
    ap.add_argument("--clips", action="store_true", help="use /v1/audio/transcriptions/clips (one request per track)")
    ap.add_argument("--pad-ms", type=int, default=0, help="clip padding for --clips")
    ap.add_argument("sessions", nargs="+")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    backend = Backend(args.server, args.model, args.clips, args.pad_ms)
    data = Path(args.data)
    report: dict = {
        "server": args.server,
        "model": args.model,
        "clips_endpoint": args.clips,
        "pad_ms": args.pad_ms,
        "sessions": [],
        "vram_start_mb": vram_mb(),
    }
    grand = Timing()
    all_ref_words = all_err_words = 0.0
    agree_time = total_time = 0.0

    for sid in args.sessions:
        row = con.execute("select team_id, created_at from sessions where id=?", (sid,)).fetchone()
        if row is None:
            print(f"!! session {sid} not found")
            continue
        team_id, created = row
        voiceprints = {
            n: json.loads(e)
            for n, e in con.execute("select name, embedding from voiceprints where team_id=?", (team_id,))
        }
        tracks = con.execute(
            "select track_num, filename, speaker_name, diarize, open_space from session_tracks where session_id=? order by track_num",  # noqa: E501
            (sid,),
        ).fetchall()
        stored_all = [
            (tn, Seg(a, b, sp, tx))
            for tn, a, b, sp, tx in con.execute(
                "select track_num, start_ms, end_ms, speaker, text from session_segments where session_id=?", (sid,)
            )
        ]
        print(f"\n== session {sid} ({created}) tracks={len(tracks)} stored_segments={len(stored_all)}")
        sess: dict = {"id": sid, "created_at": created, "tracks": []}
        for track_num, filename, speaker_name, _diarize, open_space in tracks:
            path = data / "sessions" / sid / "tracks" / filename
            if not path.exists():  # the DB keeps the upload name; on disk tracks are stored as track_N.wav
                path = path.with_name(f"track_{track_num}.wav")
            if not path.exists():
                print(f"   track {track_num}: file missing ({path.name}), skipped")
                continue
            timing = Timing()
            new = replay_track(backend, path, speaker_name, bool(open_space), voiceprints, timing)
            stored = [s for tn, s in stored_all if tn == track_num]
            w, agree, diffs = compare_track(stored, new)
            mode = "named" if speaker_name and not open_space else ("open-space" if open_space else "diarized")
            audio_s = Wav(path).duration_ms / 1000
            tr = TrackResult(track_num, mode, audio_s, len(stored), len(new), w, agree, diffs)
            sess["tracks"].append(
                {
                    **tr.__dict__,
                    "timing": timing.__dict__,
                    "new_segments": [s.__dict__ for s in sorted(new, key=lambda s: s.start_ms)],
                    "stored_segments_data": [s.__dict__ for s in sorted(stored, key=lambda s: s.start_ms)],
                }
            )
            ref_words = sum(len(s.text.split()) for s in stored)
            if w is not None:
                all_ref_words += ref_words
                all_err_words += w * ref_words
            if agree is not None:
                st = sum(s.duration_ms for s in stored)
                agree_time += agree * st
                total_time += st
            for k, v in timing.__dict__.items():
                setattr(grand, k, getattr(grand, k) + v)
            print(
                f"   track {track_num} [{mode:9}] {audio_s:7.0f}s  segs stored/new {len(stored):4}/{len(new):4}"
                f"  WER {w if w is None else round(w, 3)!s:>6}  speaker-agree {agree if agree is None else round(agree, 3)!s:>6}"  # noqa: E501
                f"  diar {timing.diar_wall_s:6.1f}s  stt-chunks {timing.stt_chunk_elapsed_s:6.1f}s ({timing.chunks} chunks)"  # noqa: E501
                f"  stt-file {timing.stt_file_wall_s:5.1f}s  filtered {timing.filtered_segments}  word-diffs {len(diffs)}"  # noqa: E501
            )
        report["sessions"].append(sess)

    report["vram_end_mb"] = vram_mb()
    summary = {
        "wer_overall": (all_err_words / all_ref_words) if all_ref_words else None,
        "speaker_agreement_overall": (agree_time / total_time) if total_time else None,
        "rtf_diarization": grand.diar_wall_s / grand.diar_audio_s if grand.diar_audio_s else None,
        "rtf_stt_chunks_throughput": grand.stt_chunk_elapsed_s / grand.stt_chunk_audio_s
        if grand.stt_chunk_audio_s
        else None,
        "rtf_stt_chunks_latency": grand.stt_chunk_wall_s / grand.stt_chunk_audio_s if grand.stt_chunk_audio_s else None,
        "rtf_stt_whole_file": grand.stt_file_wall_s / grand.stt_file_audio_s if grand.stt_file_audio_s else None,
        "embedding_avg_latency_s": grand.emb_wall_s / grand.emb_calls if grand.emb_calls else None,
        "chunks": grand.chunks,
        "segments_kept": grand.kept_segments,
        "segments_filtered_as_hallucination": grand.filtered_segments,
        "audio_total_s": grand.diar_audio_s + grand.stt_file_audio_s,
    }
    report["summary"] = summary
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"replay_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n== summary")
    for k, v in summary.items():
        print(f"   {k:36} {v if not isinstance(v, float) else round(v, 4)}")
    print(f"   vram start/end MB                    {report['vram_start_mb']}/{report['vram_end_mb']}")
    print(f"   saved {out}")


if __name__ == "__main__":
    main()
