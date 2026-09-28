"""Send every test clip to the experimental server (3 in flight), score, and write a result JSON."""
import argparse, json, subprocess, threading, time, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from score import normalize, normalize_num, wer_cer

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--server", default="http://127.0.0.1:8001")
ap.add_argument("--model"); ap.add_argument("--prompt"); ap.add_argument("--hotwords")
ap.add_argument("--inflight", type=int, default=3); ap.add_argument("--subsets", default="")
ap.add_argument("--note", default=""); ap.add_argument("--manifest", default="testset/manifest.jsonl"); ap.add_argument("--extra", default="", help="extra form fields k=v,k=v")
a = ap.parse_args()

items = [json.loads(l) for l in open(a.manifest, encoding="utf8")]
if a.subsets: items = [m for m in items if any(m["subset"].startswith(s) for s in a.subsets.split(","))]

def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])

peak = [0]; stop = threading.Event()
def sampler():
    while not stop.is_set():
        try: peak[0] = max(peak[0], smi())
        except Exception: pass
        time.sleep(0.5)

client = httpx.Client(timeout=900)
def one(m):
    data = {"language": "ru", "response_format": "verbose_json"}
    if a.model: data["model"] = a.model
    if a.prompt: data["prompt"] = a.prompt
    if a.hotwords: data["hotwords"] = a.hotwords
    for kv in filter(None, a.extra.split(",")):
        k, v = kv.split("=", 1); data[k] = v
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            r = client.post(a.server + "/v1/audio/transcriptions", data=data,
                            files={"file": (Path(m["path"]).name, open(m["path"], "rb"), "audio/wav")})
            r.raise_for_status(); break
        except Exception as e:
            err = str(e); time.sleep(2)
    else:
        return dict(id=m["id"], error=err)
    lat = time.perf_counter() - t0
    j = r.json(); segs = j.get("segments") or []
    kept = [s for s in segs if not (s["no_speech_prob"] >= 0.5 and s["avg_logprob"] <= -0.25)]
    return dict(id=m["id"], subset=m["subset"], kind=m["kind"], duration=m["duration"], ref=m["ref"], latency=round(lat, 3),
                raw_text=j.get("text", ""), text="".join(s["text"] for s in kept).strip(), n_segs=len(segs), n_kept=len(kept),
                segs=[dict(t=s["text"], st=round(s["start"], 2), en=round(s["end"], 2), nsp=round(s["no_speech_prob"], 3), lp=round(s["avg_logprob"], 3), cr=round(s["compression_ratio"], 2)) for s in segs])

# warm-up (loads the model if a different one is requested) - not timed
one(items[0])
idle = smi()
th = threading.Thread(target=sampler, daemon=True); th.start()
t0 = time.perf_counter()
with ThreadPoolExecutor(a.inflight) as ex: res = list(ex.map(one, items))
wall = time.perf_counter() - t0
stop.set(); th.join()

errors = [r for r in res if "error" in r]; ok = [r for r in res if "error" not in r]
by = collections.defaultdict(list)
for r in ok: by[r["subset"]].append(r)
speech = [r for r in ok if r["subset"] != "nonspeech"]
summary = {}
for name, rs in list(by.items()) + [("ALL_SPEECH", speech)]:
    if name == "nonspeech": continue
    w, c, nw = wer_cer([r["ref"] for r in rs], [r["text"] for r in rs])
    wn, cn, _ = wer_cer([r["ref"] for r in rs], [r["text"] for r in rs], normalize_num)
    wr, _, _ = wer_cer([r["ref"] for r in rs], [r["raw_text"] for r in rs])
    summary[name] = dict(n=len(rs), ref_words=nw, audio_min=round(sum(r["duration"] for r in rs) / 60, 2),
                         wer=round(w, 4), cer=round(c, 4), wer_num=round(wn, 4), cer_num=round(cn, 4), wer_unfiltered=round(wr, 4))
ns = by.get("nonspeech", [])
summary["nonspeech"] = dict(n=len(ns), with_text=sum(1 for r in ns if normalize(r["raw_text"])),
                            pass_client_filter=sum(1 for r in ns if normalize(r["text"])),
                            examples=[(r["id"], r["text"][:80]) for r in ns if normalize(r["text"])])
audio = sum(r["duration"] for r in ok)
out = dict(name=a.name, note=a.note, args=vars(a), n_items=len(items), n_errors=len(errors), errors=errors[:5],
           wall_s=round(wall, 1), audio_s=round(audio, 1), rtf=round(wall / audio, 4),
           rtf_per_request=round(sum(r["latency"] for r in ok) / audio, 4),
           vram_host_idle_mib=idle, vram_host_peak_mib=peak[0], summary=summary, items=res)
Path("results").mkdir(exist_ok=True)
json.dump(out, open(f"results/{a.name}.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(f"{a.name}: rtf={out['rtf']} wall={out['wall_s']}s errors={len(errors)} vram idle/peak={idle}/{peak[0]}")
for k, v in summary.items():
    if k == "nonspeech": print(f"  nonspeech: {v['with_text']}/{v['n']} text, {v['pass_client_filter']} pass filter")
    else: print(f"  {k:22s} n={v['n']:4d} WER={v['wer']:.4f} CER={v['cer']:.4f} WERnum={v['wer_num']:.4f} WERraw={v['wer_unfiltered']:.4f}")
