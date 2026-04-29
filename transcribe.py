import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from typing import ClassVar, NamedTuple, Self, TextIO
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

class TranscriptionWordType(BaseModel):
    word: str
    start: float
    end: float

class TranscriptionBlank(TranscriptionWordType):
    word: ClassVar[str] = ""

class TranscriptionWord(TranscriptionWordType):
    probability: float

    @classmethod
    def merge_words(cls, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> Self:
        check1, check2 = isinstance(word_1, cls), isinstance(word_2, cls)
        if check1 and check2:
            return cls(
                word=word_1.word + word_2.word,
                start=min(word_1.start, word_2.start),
                end=max(word_1.end, word_2.end),
                probability=(word_1.probability + word_2.probability) / 2
            )
        elif check1 or check2:
            this, other = (word_1, word_2) if check1 else (word_2, word_1)
            if isinstance(other, TranscriptionBlank):
                return cls(
                    word=this.word,
                    start=min(this.start, other.start),
                    end=max(this.end, other.end),
                    probability=this.probability
                )
        raise TypeError(f"Incompatible merge types ({type(word_1)}, {type(word_2)})")

    def __add__(self, other: TranscriptionWordType):
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
    words: Sequence[TranscriptionWord|TranscriptionBlank]

class TranscriptionResult(BaseModel):
    text: str
    segments: Sequence[TranscriptionSegment]
    language: str

def transcribe_audio(model: WhisperModel, media_path: Path) -> TranscriptionResult:
    return TranscriptionResult.model_validate(
        model.generate(str(media_path), word_timestamps=True),
        from_attributes=True
    )
