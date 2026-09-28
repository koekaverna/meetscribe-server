# Benchmarks

Measurements behind the defaults of this server. Hardware: RTX 4080 16 GB, Docker on WSL2.
Date: 2026-09-28. Versions: faster-whisper 1.2.1, CTranslate2 4.8.2, ONNX Runtime 1.26.

## Accuracy on public Russian speech

Test set: 424 clips, 62.8 minutes, 5750 reference words, built by
[`bench/groundtruth`](../bench/groundtruth) from Golos far-field, Golos crowd, FLEURS and the
Podlodka podcast test split, single utterances and joined clips of 20 to 30 s. Plus 120 clips of
silence, room noise and instrumental music.

Decoding is deterministic: three runs of the baseline gave identical output. Noise is therefore
estimated by a paired bootstrap over clips. "Interval" is the 95% interval of the WER difference
to the baseline; an interval that contains zero means no measurable difference.

### Models

| Model | WER | CER | Interval vs medium | Clips with over half the words lost | Real-time factor | GPU memory vs medium |
|---|---|---|---|---|---|---|
| `Systran/faster-whisper-medium` | 17.5% | 10.9% | baseline | 49 | 0.027 | baseline |
| `Systran/faster-whisper-large-v2` | 14.9% | 8.9% | -4.0 .. -1.3 | 24 | 0.038 | +1.9 GB |
| `Systran/faster-whisper-large-v3` | 13.4% | 8.0% | -5.1 .. -3.0 | 22 | 0.038 | +2.2 GB |
| `deepdml/faster-whisper-large-v3-turbo-ct2` | 14.1% | 8.2% | -4.6 .. -2.3 | 12 | 0.016 | same |
| `bond005/whisper-podlodka-turbo`, converted to CT2 | 13.2% | 8.4% | -6.0 .. -2.6 | 16 | 0.015 | same |
| `bzikst/faster-whisper-large-v3-russian` | 12.8% | 7.2% | -6.4 .. -3.2 | 16 | 0.038 | +2.1 GB |

- Every large model beats medium. The differences between the large models are within noise.
- The turbo models are faster than medium at the same memory, which makes them the practical
  choice.
- Medium drops the tail of long clips in batched decoding: 49 clips lost more than half of their
  words.
- Russian fine-tunes write English terms in Cyrillic. On 54 Latin-script reference words: medium
  35 exact, large-v3 32, large-v3-turbo 28, podlodka-turbo 28. On a real meeting the gap was
  larger: 44 Latin-script words with large-v3-turbo against 7 with podlodka-turbo.

Models outside the Whisper family, run through `onnx-asr`:

| Model | WER | Notes |
|---|---|---|
| NVIDIA Parakeet TDT 0.6b v3 | 7.1% | Lead comes from Golos, possibly training overlap, not verified. Not better than large-v3 on FLEURS and Podlodka. About 7 GB of GPU memory. No confidence scores. Writes English on music |
| GigaAM v3 RNN-T with punctuation | 13.1% | 12 of 54 English terms |
| GigaAM v3 RNN-T | 15.9% | Loses text on 16 of 79 clips of 20 to 30 s |

### Decoding settings

All on the medium model unless stated.

| Change | WER | Interval vs baseline | Verdict |
|---|---|---|---|
| Beam size 1 | 17.7% | -0.8 .. +1.4 | no difference |
| Beam size 10 | 17.4% | -0.5 .. +0.2 | no difference, 30% slower |
| `int8_float16` | 17.4% | -0.6 .. +0.4 | no difference, 0.7 GB less memory |
| VAD off | 18.2% | -1.0 .. +3.3 | text on 20 of 20 non-speech clips, against 1 of 20 |
| `without_timestamps=False` | 20.1% | -0.7 .. +8.4 | worse |
| Sequential decoding with temperature fallback | 19.7% | +0.7 .. +4.6 | worse |
| Sequential, large model, `condition_on_previous_text=True` | 20.9% | | repetition loops |
| `initial_prompt` with rare terms | 22.4% | +2.3 .. +7.9 | worse |
| `hotwords` with rare terms | 46.0% | +21 .. +36 | repetition loops |

The prompt raised exact hits on the prompted terms from 35 to 48 of 54 and still lost overall.
A glossary prompt applied to every request is not worth it.

## Hallucinations

Whisper emits subtitle credits and sign-offs on silence and noise. On one real two-track meeting
transcribed in 421 diarization-based chunks:

| Model | Segments | Known hallucinated phrases | Their `no_speech_prob` | Their median `avg_logprob` |
|---|---|---|---|---|
| large-v3-turbo | 393 | 48 | 0.0 for every segment | -0.28 |

Real speech in the same run has a median `avg_logprob` of -0.18, and a tenth of it is below -0.64.
On large-v3-turbo no confidence threshold works. `avg_logprob <= -1.0` caught 0 of the 48 and
dropped 12 short real replies. Matching the text caught 48 of 48 and nothing else, which is why
the server filters by phrase.

A confidence rule has a second cost: it removes a whole segment at once. With medium, the rule
`no_speech_prob >= 0.5 and avg_logprob <= -0.25` dropped 78 of 456 segments of that meeting, and
about 14 of them were real utterances.

## Speed of the clips endpoint

Two real sessions, 677 clips, 84 minutes of speech, medium model.

| Path | Real-time factor | GPU memory peak, whole card |
|---|---|---|
| One upload per clip, 3 in flight | 0.031 | not measured |
| Clips endpoint, batch 8 | 0.020 | not measured |
| Clips endpoint, batch 16 | 0.013 | 9.9 GB |
| Clips endpoint, batch 32 | 0.012 | 12.3 GB |

The card held about 4.4 GB before the runs. Text is the same on both paths: 616 segments kept on
each, 12454 against 12456 words. Batch 8 is the default so that two transcriptions and a
diarization fit on a 16 GB card together.

Two variants were tried and rejected:

- **VAD over the whole track instead of per clip.** Different clip boundaries, 1.2% fewer words,
  and the output no longer matches the per-clip path.
- **Padding of 200 ms around clips.** The text changes by 3 to 9% with more hallucinations and no
  visible gain. Padding stays available as `pad_ms`, default 0.

## Other timings

| Task | Real-time factor |
|---|---|
| Diarization, whole file | 0.013, one hour in about 47 s |
| Whole-file transcription | 0.003 to 0.005 |
| Speaker embedding | about 0.08 s per request |

GPU memory with all three models loaded: about 2.3 GB, the same before and after a load test.
