"""Evaluate hallucination-filter rules on segment-level outputs of a main run + an extended non-speech run."""

import json
import sys

from score import normalize

main, nsx = sys.argv[1], sys.argv[2]
BLOCK = [
    "Редактор субтитров",
    "Корректор",
    "Субтитры",
    "Продолжение следует",
    "С вами был Игорь Негода",
    "Спасибо за просмотр",
    "КОНЕЦ",
    "Фондю любит тебя",
    "Увидимся в следующем видео",
    "СПОКОЙНАЯ МУЗЫКА",
    "СМЕХ",
]
BLOCK = [normalize(b) for b in BLOCK]
segs = []  # (label, seg, clip_id)
for run in (main, nsx):
    for r in json.load(open(f"results/{run}.json", encoding="utf8"))["items"]:
        if "error" in r:
            continue
        refw = set(normalize(r["ref"]).split())
        for s in r["segs"]:
            w = normalize(s["t"]).split()
            if not w:
                continue
            if r["subset"] == "nonspeech":
                lab = "halluc"
            else:
                lab = "speech" if sum(x in refw for x in w) >= 0.5 * len(w) else "unmatched"
            segs.append((lab, s, r["id"], len(w)))


def blocked(s, n):
    """blocklist phrase present AND it makes up most of the segment (segment <= phrase words + 8)."""
    t = normalize(s["t"])
    return any(b and b in t and n <= len(b.split()) + 8 for b in BLOCK)


def wps(s, n):
    d = max(s["en"] - s["st"], 0.01)
    return n / d


RULES = {
    "client (nsp>=0.5 & lp<=-0.25)": lambda s, n: s["nsp"] >= 0.5 and s["lp"] <= -0.25,
    "client, only if <=5 words": lambda s, n: s["nsp"] >= 0.5 and s["lp"] <= -0.25 and n <= 5,
    "blocklist phrase (segment <= phrase+8 words)": blocked,
    "compression_ratio > 2.4": lambda s, n: s["cr"] > 2.4,
    "words/s > 5": lambda s, n: wps(s, n) > 5,
    "words/s < 0.3 (seg >= 3 s)": lambda s, n: (s["en"] - s["st"]) >= 3 and wps(s, n) < 0.3,
    "nsp>=0.6 only": lambda s, n: s["nsp"] >= 0.6,
    "lp <= -1.0": lambda s, n: s["lp"] <= -1.0,
    "client<=5w | blocklist | cr>2.4": lambda s, n: (
        (s["nsp"] >= 0.5 and s["lp"] <= -0.25 and n <= 5) or blocked(s, n) or s["cr"] > 2.4
    ),
    "client<=5w | blocklist | cr>2.4 | lp<=-1.0": lambda s, n: (
        (s["nsp"] >= 0.5 and s["lp"] <= -0.25 and n <= 5) or blocked(s, n) or s["cr"] > 2.4 or s["lp"] <= -1.0
    ),
}
H = [x for x in segs if x[0] == "halluc"]
S = [x for x in segs if x[0] == "speech"]
U = [x for x in segs if x[0] == "unmatched"]
nsclips = len({x[2] for x in H})
print(
    f"{main}+{nsx}: non-empty segments: halluc(non-speech clips)={len(H)} from {nsclips} clips; speech-matched={len(S)}; speech-unmatched={len(U)}"
)
print(
    "| rule | halluc caught (recall) | speech segs dropped / words | precision (halluc vs matched speech) | unmatched-in-speech dropped |"
)
print("|---|---|---|---|---|")
for name, f in RULES.items():
    h = sum(1 for _, s, _, n in H if f(s, n))
    sp = [(s, n) for _, s, _, n in S if f(s, n)]
    u = sum(1 for _, s, _, n in U if f(s, n))
    prec = h / (h + len(sp)) if h + len(sp) else float("nan")
    print(
        f"| {name} | {h}/{len(H)} ({h / len(H) * 100 if H else 0:.0f}%) | {len(sp)}/{len(S)} segs, {sum(n for _, n in sp)} words | {prec * 100:.0f}% | {u}/{len(U)} |"
    )
print("hallucinated texts (non-speech):", sorted({normalize(s["t"])[:40] for _, s, _, _ in H})[:40])
# clip-level effect of each rule on speech clips of the main run (WER with rule applied vs. no filter)
from score import wer_cer

items = [
    r
    for r in json.load(open(f"results/{main}.json", encoding="utf8"))["items"]
    if "error" not in r and r["subset"] != "nonspeech"
]
w0, _, nw = wer_cer([r["ref"] for r in items], [r["raw_text"] for r in items])
print(f"clip level, {len(items)} speech clips, {nw} ref words; WER with no filter = {w0 * 100:.2f}")
print("| rule | speech clips with >=1 seg dropped | clips emptied | WER after rule |")
print("|---|---|---|---|")
for name, f in RULES.items():
    hyps = []
    aff = emp = 0
    for r in items:
        kept = [s for s in r["segs"] if not f(s, len(normalize(s["t"]).split()))]
        hyps.append("".join(s["t"] for s in kept))
        if len(kept) < len(r["segs"]):
            aff += 1
        if normalize(r["raw_text"]) and not normalize(hyps[-1]):
            emp += 1
    w, _, _ = wer_cer([r["ref"] for r in items], hyps)
    print(f"| {name} | {aff} | {emp} | {w * 100:.2f} |")
