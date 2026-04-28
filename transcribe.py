import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from typing import NamedTuple, Self, TextIO
# from typing import Callable
from mlx_audio.stt import load
from mlx_audio.stt.models.whisper.whisper import Model as WhisperModel

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    TypeAdapter,
    computed_field,
    model_validator,
)

# from subtitles import SubtitleProcessor

class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float
    probability: float

    @classmethod
    def merge_words(cls, word_1: Self, word_2: Self) -> Self:
        return cls(
            word=word_1.word + word_2.word,
            start=min(word_1.start, word_2.start),
            end=max(word_1.end, word_2.end),
            probability=(word_1.probability + word_2.probability) / 2
        )

    def __add__(self, other: Self):
        return self.merge_words(self, other)

class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Sequence[TranscriptionWord]

class TranscriptionResult(BaseModel):
    text: str
    segments: Sequence[TranscriptionSegment]
    language: str

def transcribe_audio(model: WhisperModel, media_path: Path) -> TranscriptionResult:
    return TranscriptionResult.model_validate(
        model.generate(str(media_path), word_timestamps=True),
        from_attributes=True
    )
