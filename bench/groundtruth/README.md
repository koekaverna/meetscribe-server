# Ground-truth benchmark

Measures word error rate of a running server against public Russian speech with human references.
Results of the 2026-09-28 run are in [docs/benchmarks.md](../../docs/benchmarks.md).

## Test set

424 speech clips (62.8 min, 5750 reference words) plus non-speech clips:

| Subset | Source on Hugging Face |
|---|---|
| Golos far-field, single and joined 20-30 s | `bond005/sberdevices_golos_100h_farfield`, test |
| Golos crowd, single and joined | `bond005/sberdevices_golos_10h_crowd`, test |
| FLEURS, single and joined | `google/fleurs`, `ru_ru`, test |
| Podlodka podcast | `bond005/podlodka_speech`, test |
| Non-speech | silence, `ashraq/esc50`, `lewtun/music_genres_small` |

`testset/` (129 MB) and `raw/` are git-ignored. To rebuild: download the files named at the top of
`build_testset.py` into `raw/<owner>__<dataset>/` keeping the repository paths, then run it.
The seed is fixed, so the set is reproducible.

## Run

Run from this folder, against a server that is not serving users:

```
uv run --with httpx --with jiwer python run_bench.py baseline --server http://127.0.0.1:8001
uv run --with httpx --with jiwer python run_bench.py turbo --server http://127.0.0.1:8001 --model deepdml/faster-whisper-large-v3-turbo-ct2
uv run --with jiwer --with numpy python analyze.py baseline turbo
```

- `run_bench.py NAME` writes `results/NAME.json` with every hypothesis and segment.
- `analyze.py` prints WER and CER per subset and a paired bootstrap interval against the first run.
  A difference whose interval contains zero is noise.
- `filter_analysis.py` evaluates hallucination filter rules on the saved outputs.
- `score.py` holds the normalisation: lowercase, ё to е, punctuation removed. `WER-num` collapses
  numbers, because Golos spells them as words and Whisper writes digits.

Decoding is deterministic: repeated runs of one configuration give identical output.
