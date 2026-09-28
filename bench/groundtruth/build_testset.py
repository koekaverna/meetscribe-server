"""Build the ground-truth test set: utterance subsets, concatenated 20-30 s chunks, and non-speech clips."""
import io, json, random, tarfile, csv, collections
from pathlib import Path
import numpy as np, soundfile as sf, librosa, pyarrow.parquet as pq

SR = 16000
OUT = Path("testset"); (OUT / "wav").mkdir(parents=True, exist_ok=True)
rng = random.Random(1234)
manifest = []

def load(b):
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR: x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x

def save(name, x, subset, ref, kind, extra=None):
    p = OUT / "wav" / f"{name}.wav"
    sf.write(p, np.clip(x, -1, 1), SR, subtype="PCM_16")
    manifest.append(dict(id=name, subset=subset, kind=kind, path=str(p).replace("\\", "/"), ref=ref,
                         duration=round(len(x) / SR, 3), **(extra or {})))

def take_until(items, minutes):
    out, tot = [], 0.0
    while items and tot < minutes * 60:
        it = items.pop(); out.append(it); tot += len(it[0]) / SR
    return out

def concat_sets(items, prefix, subset, n_target_min):
    """Join consecutive utterances with 0.3-1.5 s silences into 20-30 s clips."""
    clips, tot = 0, 0.0
    while items and tot < n_target_min * 60:
        target = rng.uniform(20, 30); parts, refs, dur = [], [], 0.0
        while items:
            x, ref = items[-1]
            gap = rng.uniform(0.3, 1.5)
            if parts and dur + gap + len(x) / SR > 30: break
            items.pop()
            if parts: parts.append(np.zeros(int(gap * SR), np.float32)); dur += gap
            parts.append(x); refs.append(ref); dur += len(x) / SR
            if dur >= target: break
        if dur < 15 and not items: break
        save(f"{prefix}_{clips:03d}", np.concatenate(parts), subset, " ".join(refs), "concat", {"n_utts": len(refs)})
        clips += 1; tot += dur

def parquet_items(f, text_col="transcription"):
    t = pq.read_table(f).to_pylist()
    return [(load(r["audio"]["bytes"]), r[text_col]) for r in t]

# Golos farfield / crowd: shuffle, disjoint utt and concat subsets
for name, f in [("golos_farfield", "raw/bond005__sberdevices_golos_100h_farfield/data/test-00000-of-00001-7fb58d4cda8ff9fe.parquet"),
                ("golos_crowd", "raw/bond005__sberdevices_golos_10h_crowd/data/test-00000-of-00003-cad0cfaddbc8fa71.parquet")]:
    items = [it for it in parquet_items(f) if (it[1] or "").strip()]
    print(name, "available", len(items))
    rng.shuffle(items)
    for i, (x, ref) in enumerate(take_until(items, 8)):
        save(f"{name}_u{i:04d}", x, name + "_utt", ref, "utt")
    concat_sets(items, name + "_c", name + "_concat", 10)

# FLEURS ru_ru test: one recording per sentence id
rows = list(csv.reader(open("raw/google__fleurs/data/ru_ru/test.tsv", encoding="utf8"), delimiter="\t", quoting=csv.QUOTE_NONE))
by_file = {r[1]: r for r in rows}
seen, fl = set(), []
with tarfile.open("raw/google__fleurs/data/ru_ru/audio/test.tar.gz") as tar:
    for m in tar:
        if not m.isfile(): continue
        fn = Path(m.name).name
        r = by_file.get(fn)
        if r is None or r[0] in seen: continue
        seen.add(r[0]); fl.append((load(tar.extractfile(m).read()), r[2]))
print("fleurs unique sentences", len(fl))
rng.shuffle(fl)
for i, (x, ref) in enumerate(take_until(fl, 8)):
    save(f"fleurs_u{i:04d}", x, "fleurs_utt", ref, "utt")
concat_sets(fl, "fleurs_c", "fleurs_concat", 10)

# Podlodka (conversational IT podcast), test split: 20 utterances; concat in original order (re-uses the same audio)
pod = parquet_items("raw/bond005__podlodka_speech/data/test-00000-of-00001.parquet")
for i, (x, ref) in enumerate(pod):
    save(f"podlodka_u{i:04d}", x, "podlodka_utt", ref, "utt")
concat_sets(list(reversed(pod)), "podlodka_c", "podlodka_concat", 99)

# Non-speech: 4 silence, 8 ESC-50 room sounds (4x5 s joined), 8 instrumental music (20 s)
for i, (dur, lvl) in enumerate([(10, 0), (30, 0), (20, 1e-4), (25, 3e-4)]):
    x = np.random.default_rng(i).normal(0, lvl, int(dur * SR)).astype(np.float32) if lvl else np.zeros(int(dur * SR), np.float32)
    save(f"nonspeech_silence_{i}", x, "nonspeech", "", "silence", {"desc": f"{dur}s, noise std {lvl}"})
esc = pq.read_table("raw/ashraq__esc50/data/train-00000-of-00002-2f1ab7b824ec751f.parquet").to_pylist()
cats = collections.defaultdict(list)
for r in esc: cats[r["category"]].append(r)
for c in ["keyboard_typing", "mouse_click", "clock_tick", "door_wood_knock", "footsteps", "vacuum_cleaner", "rain", "breathing"]:
    x = np.concatenate([load(r["audio"]["bytes"]) for r in cats[c][:4]])
    save(f"nonspeech_room_{c}", x, "nonspeech", "", "room", {"desc": "ESC-50 " + c + " x4"})
mus = [r for r in pq.read_table("raw/lewtun__music_genres_small/data/train-00000-of-00001-63d68663287b1638.parquet").to_pylist() if r["genre"] == "Instrumental"][:8]
for r in mus:
    x = load(r["audio"]["bytes"])[5 * SR:25 * SR]
    save(f"nonspeech_music_{r['song_id']}", x, "nonspeech", "", "music", {"desc": "music_genres_small Instrumental song " + str(r["song_id"])})

with open(OUT / "manifest.jsonl", "w", encoding="utf8") as f:
    for m in manifest: f.write(json.dumps(m, ensure_ascii=False) + "\n")
summ = collections.defaultdict(lambda: [0, 0.0])
for m in manifest: summ[m["subset"]][0] += 1; summ[m["subset"]][1] += m["duration"]
for k, (n, d) in summ.items(): print(f"{k:22s} n={n:4d} {d/60:6.1f} min")
print("total", sum(d for _, d in summ.values()) / 60, "min")
