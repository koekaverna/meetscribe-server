import io
import wave

import numpy as np
import pytest


def make_wav_bytes(duration_s: float = 3.0, sample_rate: int = 16000, seed: int = 0) -> bytes:
    """Synthetic 16 kHz mono PCM16 WAV: a low tone with noise, enough to exercise decoders and models."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    signal = tone + 0.02 * rng.standard_normal(t.size)
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()
