"""Summarise result JSONs: per-subset WER/CER, paired bootstrap vs baseline, truncations, non-speech."""
import json, sys, glob, random
import jiwer
from score import normalize, normalize_num

SUBS = ["golos_farfield_utt", "golos_farfield_concat", "golos_crowd_utt", "golos_crowd_concat", "fleurs_utt", "fleurs_concat", "podlodka_utt"]
def load(n): return json.load(open(f"results/{n}.json", encoding="utf8"))

def counts(items, fn=normalize):
    """per-item (errors, ref_words) for bootstrap."""
    out = {}
    for r in items:
        ref = fn(r["ref"])
        if not ref: continue
        o = jiwer.process_words(ref, fn(r["text"]) or "")
        out[r["id"]] = (o.substitutions + o.deletions + o.insertions, len(ref.split()), o.deletions, o.insertions)
    return out

def boot(a, b, iters=2000, seed=0):
    ids = sorted(set(a) & set(b)); rnd = random.Random(seed); diffs = []
    for _ in range(iters):
        s = [ids[rnd.randrange(len(ids))] for _ in ids]
        wa = sum(a[i][0] for i in s) / sum(a[i][1] for i in s); wb = sum(b[i][0] for i in s) / sum(b[i][1] for i in s)
        diffs.append(wb - wa)
    diffs.sort(); return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]

names = sys.argv[1:] or sorted(p[8:-5] for p in glob.glob("results/*.json"))
base = load("baseline_r1"); bi = [r for r in base["items"] if r.get("subset", "nonspeech") != "nonspeech" and "error" not in r]
bc = counts(bi); bcn = counts(bi, normalize_num)
rows = []
for n in names:
    try: d = load(n)
    except Exception: continue
    s = d.get("summary", {})
    items = [r for r in d["items"] if "error" not in r and r.get("subset") != "nonspeech"]
    if not items or "ALL_SPEECH" not in s: continue
    c = counts(items); cn = counts(items, normalize_num)
    lo, hi = boot(bc, c); lon, hin = boot(bcn, cn)
    trunc = sum(1 for r in items if len(normalize(r["text"]).split()) < 0.5 * len(normalize(r["ref"]).split()))
    dels = sum(v[2] for v in c.values()); ins = sum(v[3] for v in c.values()); nw = sum(v[1] for v in c.values())
    ns = s.get("nonspeech", {})
    rows.append((n, s, d, lo, hi, lon, hin, trunc, dels / nw, ins / nw, ns))
print("| config | " + " | ".join(SUBS) + " | ALL WER | ALL CER | ALL WER-num | dWER vs base 95% CI | dWER-num CI | del% | ins% | trunc clips | RTF | VRAM peak host MiB | non-speech text / pass filter |")
print("|" + "---|" * (len(SUBS) + 13))
for n, s, d, lo, hi, lon, hin, trunc, dl, ins, ns in rows:
    cells = [f"{s[k]['wer']*100:.1f} / {s[k]['cer']*100:.1f}" if k in s else "-" for k in SUBS]
    a = s["ALL_SPEECH"]
    print(f"| {n} | " + " | ".join(cells) + f" | {a['wer']*100:.2f} | {a['cer']*100:.2f} | {a['wer_num']*100:.2f} | {lo*100:+.2f}..{hi*100:+.2f} | {lon*100:+.2f}..{hin*100:+.2f} | {dl*100:.1f} | {ins*100:.1f} | {trunc} | {d.get('rtf','-')} | {d.get('vram_host_peak_mib','-')} | {ns.get('with_text','-')}/{ns.get('n','-')} , {ns.get('pass_client_filter','-')} |")
