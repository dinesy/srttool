from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import re

from collections.abc import Callable, Generator, Iterable, Iterator, MutableSequence, Sequence
from typing import cast, Generic, NamedTuple, Self, TextIO, TypeVar, ParamSpec, Protocol


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

# from subtitlewriter import Timecode, SubtitleEntry, AssEntry, SubtitleFile, AssFile
from transcribe import TranscriptionResult, TranscriptionSegment, TranscriptionWordType, TranscriptionWord, TranscriptionBlank

# P = ParamSpec("P")
# R = TypeVar("R", bound=bool)
# S = TypeVar("S", bound=Iterable[TranscriptionWord])
# class SubtitleProcessor(ABC, Generic[P, R, S]):
# type T = TranscriptionWord
class SubtitleProcessor[**P](ABC):
# class SubtitleProcessor[*Ts, **P = [tuple[*Ts]]](ABC):
    @classmethod
    @abstractmethod
    def test(cls, *args: P.args, **kwargs: P.kwargs) -> bool: ...

    @classmethod
    @abstractmethod
    def action(cls, *args: P.args, **kwargs: P.kwargs) -> Sequence[TranscriptionWordType]: ...

    def transform_words(cls, words: Iterable[TranscriptionWordType]) -> Generator[TranscriptionWordType]:
    @classmethod
        # if len(words) < 2:
        #     return words
        argcount = cls.test.__func__.__code__.co_argcount-1
        if argcount != cls.action.__func__.__code__.co_argcount-1:
            raise ValueError(f"test and action methods must have the same number of arguments: {argcount-1} != {cls.action.__func__.__code__.co_argcount-1}")
        iwords = iter(words)
        current_words = deque()
        while True:
            try:
                while len(current_words) < argcount:
                    current_words.append(next(iwords))
                if cls.test(*current_words):
                    result = cls.action(*current_words)
                    current_words.clear()
                    current_words.extend(result)
                    if len(current_words) < argcount:
                        continue
                yield current_words.popleft()
            except StopIteration:
                yield from current_words
                break

    def transform_words_inplace(cls, words: MutableSequence[TranscriptionWordType]):

    @classmethod
        # if len(words) < 2:
        #     return words
        argcount = cls.test.__func__.__code__.co_argcount-1
        if argcount != cls.action.__func__.__code__.co_argcount-1:
            raise ValueError(f"test and action methods must have the same number of arguments: {argcount-1} != {cls.action.__func__.__code__.co_argcount-1}")
        i, j = 0, argcount-1
        last_word = None
        while j < len(words):
            if last_word is not words[j]:
                # protect against infinite loops (hopefully)
                if cls.test(*words[i:j+1]):
                    result = cls.action(*words[i:j+1])
                    words[i:j+1] = result
                    if len(result) < argcount:
                        continue
            last_word = words[j]
            i, j = i+1, j+1
            # yield last_word

class StripWords[TranscriptionWordType](SubtitleProcessor):
    @classmethod
    def test(cls, word_1: TranscriptionWordType) -> bool:
        return True
    @classmethod
    def action(cls, word_1: TranscriptionWordType) -> Sequence[TranscriptionWordType]:
        return [word_1.model_copy(update={"word": word_1.word.strip()})]

class FixCommaNumbers(SubtitleProcessor):
    rx1a = re.compile(r"\d+,$")
    rx1b = re.compile(r"^\d+")
    rx2a = re.compile(r"\d+$")
    rx2b = re.compile(r"^,\d+")

    @classmethod
    def test(cls, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> bool:
        return bool(
            (cls.rx1a.search(word_1.word) and cls.rx1b.search(word_2.word)) or \
            (cls.rx2a.search(word_1.word) and cls.rx2b.search(word_2.word))
        )

    @classmethod
    def action(cls, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> Sequence[TranscriptionWordType]:
        return [word_1 + word_2]
    # def fix_comma_numbers(cls, ):
        # yield from cls.process_words(words, test, lambda w1, w2: [w1 + w2])


class SubtitleChunkBase(BaseModel):
    id: int = 0
    start: float
    end: float
class SubtitleChunk(SubtitleChunkBase):
    words: Sequence[TranscriptionWordType]
    @classmethod
    def with_words(cls, words: Sequence[TranscriptionWordType]):
        starts, ends = zip(*[(word.start, word.end) for word in words])
        return cls(start=min(starts), end=max(ends), words=words)
class MultilineSubtitleChunk(SubtitleChunkBase):
    lines: Sequence[SubtitleChunk]
    @classmethod
    def with_chunks(cls, chunks: Sequence[SubtitleChunk]):
        starts, ends = zip(*[(chunk.start, chunk.end) for chunk in chunks])
        return cls(start=min(starts), end=max(ends), lines=chunks)

type TranscriptionPart = TranscriptionResult|TranscriptionSegment|Iterable[TransciptionResult|TranscriptionSegment|TranscriptionWordType]
@dataclass
class SubtitleChunker:
    transcription: TranscriptionPart
    max_line_length: int = 30
    line_limit_is_chars: bool = False
    # max_chars_per_line: int|None = None
    max_lines_per_chunk: int = 1
    strict_line_length: bool = True
    subtitle_processors: Sequence[SubtitleProcessor] = field(default_factory=[])
    def chunks(self) -> Generator[SubtitleChunk]|Generator[MultilineSubtitleChunk]:
        def iter_words(top_part: TranscriptionPart):
            if isinstance(top_part, TranscriptionResult):
                top_part = top_part.segments
            if isinstance(top_part, TranscriptionSegment):
                top_part = [top_part]
            if isinstance(top_part, Iterable):
                for part in top_part:
                    if isinstance(part, TranscriptionSegment):
                        words = part.words
                        if self.subtitle_processors:
                            for proc in self.subtitle_processors:
                                words = proc.transform_words(words)
                        yield from words
                    elif isinstance(part, TranscriptionWordType):
                        yield part
        lines = []
        current_line = []
        # current_line_words = 0
        current_line_chars = 0
        words = iter_words(self.transcription)
        word = None
        while True:
            try:
                if word is None:
                    word = next(words)
                if (
                    self.line_limit_is_chars and current_line_chars + len(word.word) >= self.max_line_length
                ) or len(current_line) >= self.max_line_length:
                    if (
                        not self.line_limit_is_chars
                    ) or (
                        current_line_chars >= self.max_line_length
                    ) or self.strict_line_length:
                        lines.append(SubtitleChunk.with_words(current_line))
                        current_line = []
                        current_line_chars = 0
                        if len(lines) == self.max_lines_per_chunk:
                            yield lines[0] if len(lines) == 1 else MultilineSubtitleChunk.with_chunks(lines)
                            lines = []
                current_line.append(word)
                word = next(words)
            except StopIteration:
                if current_line:
                    lines.append(SubtitleChunk.with_words(current_line))
                if lines:
                    yield lines[0] if len(lines) == 1 else MultilineSubtitleChunk.with_chunks(lines)
                break

# def process_llm_results(result: TranscriptionResult, subtitle_class=SubtitleEntry):
#     index = 1
#     for segment in result.segments:
#         # text = result.text
#         last_word_idx = 0
#         fix_comma_numbers = FixCommaNumbers.transform_words(segment.words)
#         srt_entries = []
#         for seg_word in segment.words:
#             word = seg_word.word.strip()
#             word_idx = last_word_idx = segment.text.find(word, last_word_idx+1)
#             start = Timecode.from_seconds(seg_word.start)
#             end   = Timecode.from_seconds(seg_word.end)
#             srt_entries.append(
#                 subtitle_class(
#                     number=index,
#                     time_range=(start, end),
#                     text=segment.text,
#                     highlight_word=word,
#                     highlight_idx=word_idx
#                 )
#             )
#         segment.words = srt_entries
