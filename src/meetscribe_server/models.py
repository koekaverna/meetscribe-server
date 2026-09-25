"""Model loading, residency and GPU concurrency limits."""

import gc
import logging
import threading
import time
from typing import Any

import huggingface_hub
from faster_whisper import WhisperModel
from onnx_diarization.clustering.plda import PLDATransform, load_xvex_and_plda_data
from onnx_diarization.embedding import WeSpeakerEmbeddingModel
from onnx_diarization.fbank import FbankExtractor
from onnx_diarization.pipeline import PyannnoteSegmentation, SpeakerDiarizationPipeline

from meetscribe_server.config import Settings

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class GpuUnavailableError(RuntimeError):
    pass


def ort_providers(require_gpu: bool) -> list[Any]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    logger.info("ONNX Runtime %s, available providers: %s", ort.__version__, available)
    if CUDA_PROVIDER in available:
        return [(CUDA_PROVIDER, {"device_id": 0}), CPU_PROVIDER]
    if require_gpu:
        raise GpuUnavailableError(
            f"ONNX Runtime has no {CUDA_PROVIDER} (available: {available}). "
            "Check the CUDA libraries in the image or set REQUIRE_GPU=false."
        )
    return [CPU_PROVIDER]


def ctranslate2_cuda_devices() -> int:
    import ctranslate2

    count = ctranslate2.get_cuda_device_count()
    logger.info("CTranslate2 %s, CUDA devices: %d", ctranslate2.__version__, count)
    return count


class ModelStore:
    """Holds the resident models and the semaphores that bound GPU concurrency.

    * The Whisper model is loaded lazily and swapped when a request names a different model.
    * The segmentation and speaker-embedding ONNX sessions are loaded once and never unloaded.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.stt_semaphore = threading.Semaphore(settings.max_concurrent_stt)
        self.diarization_semaphore = threading.Semaphore(settings.max_concurrent_diarization)
        self.embedding_semaphore = threading.Semaphore(settings.max_concurrent_embedding)

        self._whisper_lock = threading.Lock()
        self._whisper_id: str | None = None
        self._whisper: WhisperModel | None = None

        self._onnx_lock = threading.Lock()
        self._seg_session: Any = None
        self._emb_session: Any = None

        self.fbank_extractor = FbankExtractor(sample_rate=16000)
        self.plda = PLDATransform(*load_xvex_and_plda_data())

    # --- startup ---

    def load_all(self) -> None:
        """Load every model. Called from the app lifespan when PRELOAD=true."""
        self.check_gpu()
        self.segmentation_session()
        self.embedding_session()
        self.whisper(self.settings.stt_model)

    def check_gpu(self) -> None:
        if not self.settings.require_gpu:
            return
        if self.settings.stt_device != "cpu" and ctranslate2_cuda_devices() == 0:
            raise GpuUnavailableError(
                "CTranslate2 sees no CUDA device. Check the NVIDIA runtime / driver or set REQUIRE_GPU=false."
            )
        ort_providers(require_gpu=True)

    # --- whisper ---

    def whisper(self, model_id: str) -> WhisperModel:
        """Return the resident Whisper model, loading/replacing it if a different id is requested."""
        with self._whisper_lock:
            if self._whisper is not None and self._whisper_id == model_id:
                return self._whisper
            if self._whisper is not None:
                logger.info("Unloading Whisper model %s", self._whisper_id)
                self._whisper = None
                self._whisper_id = None
                gc.collect()
            s = self.settings
            t0 = time.perf_counter()
            logger.info(
                "Loading Whisper model %s (device=%s, compute_type=%s)", model_id, s.stt_device, s.stt_compute_type
            )
            model = WhisperModel(
                model_id,
                device=s.stt_device,
                device_index=s.stt_device_index,
                compute_type=s.stt_compute_type,
                cpu_threads=s.stt_cpu_threads,
                num_workers=s.stt_num_workers,
            )
            logger.info("Loaded Whisper model %s in %.1fs", model_id, time.perf_counter() - t0)
            self._whisper = model
            self._whisper_id = model_id
            return model

    @property
    def whisper_id(self) -> str | None:
        return self._whisper_id

    # --- ONNX sessions ---

    def _load_onnx_session(self, repo_id: str, filename: str) -> Any:
        import onnxruntime as ort

        path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
        providers = ort_providers(self.settings.require_gpu)
        t0 = time.perf_counter()
        session = ort.InferenceSession(path, providers=providers)
        active = session.get_providers()
        logger.info("Loaded %s/%s in %.1fs, providers: %s", repo_id, filename, time.perf_counter() - t0, active)
        if self.settings.require_gpu and active[0] != CUDA_PROVIDER:
            raise GpuUnavailableError(
                f"{repo_id} session fell back to {active}; the CUDA provider failed to initialise."
            )
        return session

    def segmentation_session(self) -> Any:
        with self._onnx_lock:
            if self._seg_session is None:
                self._seg_session = self._load_onnx_session(self.settings.seg_model, self.settings.seg_model_file)
            return self._seg_session

    def embedding_session(self) -> Any:
        with self._onnx_lock:
            if self._emb_session is None:
                self._emb_session = self._load_onnx_session(self.settings.emb_model, self.settings.emb_model_file)
            return self._emb_session

    # --- pipelines (cheap wrappers around the resident sessions) ---

    def embedding_model(self) -> WeSpeakerEmbeddingModel:
        return WeSpeakerEmbeddingModel(session=self.embedding_session(), fbank_extractor=self.fbank_extractor)

    def diarization_pipeline(self) -> SpeakerDiarizationPipeline:
        return SpeakerDiarizationPipeline(
            segmentation=PyannnoteSegmentation(session=self.segmentation_session()),
            embedding=self.embedding_model(),
            plda=self.plda,
            embedding_batch_size=self.settings.diarization_embedding_batch_size,
        )

    # --- introspection ---

    def loaded(self) -> dict[str, bool]:
        return {
            self.settings.stt_model: self._whisper_id == self.settings.stt_model,
            self.settings.seg_model: self._seg_session is not None,
            self.settings.emb_model: self._emb_session is not None,
        }
