from typing import Literal

from pydantic import BaseModel, Field

# --- /v1/audio/transcriptions (OpenAI-compatible) ---


class TranscriptionWord(BaseModel):
    start: float
    end: float
    word: str


class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class Transcription(BaseModel):
    text: str


class TranscriptionVerbose(BaseModel):
    task: Literal["transcribe"] = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[TranscriptionSegment]
    words: list[TranscriptionWord] | None = None


# --- /v1/audio/diarization ---


class DiarizationSegment(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizationResponse(BaseModel):
    duration: float
    segments: list[DiarizationSegment]


# --- /v1/audio/speech/embedding (OpenAI embeddings shape) ---


class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int = 0


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class CreateEmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingObject] = Field(..., min_length=1, max_length=1)
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


# --- /v1/models ---


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str
    task: str
    loaded: bool


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
