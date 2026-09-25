from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (and an optional .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Transcription (faster-whisper / CTranslate2) ---
    stt_model: str = "Systran/faster-whisper-medium"
    """Default Whisper model. A request may name another CTranslate2 Whisper model; it is
    loaded on demand and replaces the resident one (only one Whisper model stays in memory)."""
    stt_device: Literal["auto", "cpu", "cuda"] = "auto"
    stt_device_index: int = 0
    stt_compute_type: str = "default"
    """CTranslate2 compute type: default, int8, int8_float16, float16, float32, ..."""
    stt_cpu_threads: int = 0
    stt_num_workers: int = 1
    stt_batch_size: int = 8
    stt_beam_size: int = 5
    stt_vad_filter: bool = True
    """Run Silero VAD inside faster-whisper before decoding (same behaviour as speaches)."""
    stt_vad_min_silence_ms: int = 160
    stt_vad_max_speech_s: float = 30.0

    # --- Diarization / speaker embeddings (onnx-diarization, ONNX Runtime) ---
    seg_model: str = "fedirz/segmentation_community_1"
    seg_model_file: str = "model.onnx"
    emb_model: str = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
    emb_model_file: str = "voxceleb_resnet34_LM.onnx"
    diarization_embedding_batch_size: int = 32

    # --- Concurrency: how many requests of each kind may run on the GPU at once ---
    max_concurrent_stt: int = 2
    max_concurrent_diarization: int = 1
    max_concurrent_embedding: int = 2

    # --- Startup ---
    require_gpu: bool = True
    """Fail at startup unless both CTranslate2 and ONNX Runtime can see a CUDA device."""
    preload: bool = True
    """Load all models during startup instead of on first request."""
    log_level: str = "info"
