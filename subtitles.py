from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from inspect import signature

import re

from collections.abc import Callable, Generator, Iterable, Iterator, MutableSequence, Sequence
from typing import cast, overload, Generic, NamedTuple, Self, TextIO, TypeVar, ParamSpec, Protocol


from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializeAsAny,
    TypeAdapter,
    computed_field,
    model_validator,
)

from transcribe import TranscriptionResult, TranscriptionSegment, TranscriptionWordType, TranscriptionWord, TranscriptionBlank

class SubtitleProcessor[**P, T](ABC):
    @abstractmethod
    def test(self, *args: P.args, **kwargs: P.kwargs) -> bool: ...

    @abstractmethod
    def action(self, *args: P.args, **kwargs: P.kwargs) -> Sequence[T]: ...

    def transform_words(self, words: Iterable[T]) -> Generator[T]:
        argcount = len(signature(self.test).parameters)
        if argcount != len(signature(self.action).parameters):
            raise ValueError(f"test and action methods must have the same number of arguments: {argcount-1} != {self.action.__func__.__code__.co_argcount-1}")
        iwords = iter(words)
        current_words = deque()
        while True:
            try:
                while len(current_words) < argcount:
                    current_words.append(next(iwords))
                if self.test(*current_words): #pyright: ignore[reportCallIssue]
                    result = self.action(*current_words) #pyright: ignore[reportCallIssue]
                    current_words.clear()
                    current_words.extend(result)
                    if len(current_words) < argcount:
                        continue
                yield current_words.popleft()
            except StopIteration:
                yield from current_words
                break

    def transform_words_inplace(self, words: MutableSequence[T]):
        argcount = self.test.__func__.__code__.co_argcount-1
        if argcount != self.action.__func__.__code__.co_argcount-1:
            raise ValueError(f"test and action methods must have the same number of arguments: {argcount-1} != {self.action.__func__.__code__.co_argcount-1}")
        i, j = 0, argcount-1
        last_word = None
        while j < len(words):
            if last_word is not words[j]:
                # protect against infinite loops (hopefully)
                if self.test(*words[i:j+1]): #pyright: ignore[reportCallIssue]
                    result = self.action(*words[i:j+1]) #pyright: ignore[reportCallIssue]
                    words[i:j+1] = result
                    if len(result) < argcount:
                        continue
            last_word = words[j]
            i, j = i+1, j+1

class StripWords(SubtitleProcessor[[TranscriptionWordType], TranscriptionWordType]):
    def test(self, word_1: TranscriptionWordType) -> bool:
        return True
    def action(self, word_1: TranscriptionWordType) -> Sequence[TranscriptionWordType]:
        return [word_1.model_copy(update={"word": word_1.word.strip()})]

class FixCommaNumbers(SubtitleProcessor[[TranscriptionWordType, TranscriptionWordType], TranscriptionWordType]):
    rx1a: re.Pattern = re.compile(r"\d+,$")
    rx1b: re.Pattern = re.compile(r"^\d+")
    rx2a: re.Pattern = re.compile(r"\d+$")
    rx2b: re.Pattern = re.compile(r"^,\d+")

    def test(self, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> bool:
        return bool(
            (self.rx1a.search(word_1.word) and self.rx1b.search(word_2.word)) or \
            (self.rx2a.search(word_1.word) and self.rx2b.search(word_2.word))
        )

    def action(self, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> Sequence[TranscriptionWordType]:
        return [word_1 + word_2]

@dataclass
class RemoveWordGaps(SubtitleProcessor[[TranscriptionWordType, TranscriptionWordType], TranscriptionWordType]):
    max_gap_threshold: float = 0.0
    extend_by: float|None = None
    extend_highlights: bool = True

    def test(self, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> bool:
        return word_2.start - word_1.end >= self.max_gap_threshold

    def action(self, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> Sequence[TranscriptionWordType]:
        if self.extend_by is None:
            new_end = word_2.start
        else:
            new_end = word_1.end + self.extend_by
        if self.extend_highlights:
            return [word_1.model_copy(update={"end": new_end}), word_2]
        else:
            return [word_1, TranscriptionBlank(start=word_1.end, end=new_end), word_2]

type ChunkType = SubtitleChunk|MultilineSubtitleChunk
@dataclass
class RemoveChunkGaps(SubtitleProcessor[[ChunkType, ChunkType], ChunkType]):
    max_gap_threshold: float = 0.0
    extend_by: float|None = None
    extend_highlights: bool = True

    def test(self, chunk_1: ChunkType, chunk_2: ChunkType) -> bool:
        return chunk_2[0].start - chunk_1[-1].end >= self.max_gap_threshold

    def action(self, chunk_1: ChunkType, chunk_2: ChunkType) -> Sequence[ChunkType]:
        if self.extend_by is None:
            new_end = chunk_2[0].start
        else:
            new_end = chunk_1[-1].end + self.extend_by
        if self.extend_highlights:
            chunk_1[-1] = chunk_1[-1].model_copy(update={"end": new_end})
        else:
            blank = TranscriptionBlank(start=chunk_1.end, end=new_end)
            chunk_1[-1:] = [chunk_1[-1], blank]
        chunk_1.end = max(chunk_1.end, chunk_1[-1].end)
        return [chunk_1, chunk_2]

class ExtendTrailingWords(SubtitleProcessor[[TranscriptionWordType, TranscriptionWordType], TranscriptionWordType]):
    def __init__(self, min_gap_threshold: float, extend_by: float):
        self._min_gap_threshold: float = min_gap_threshold
        self._extend_by: float = extend_by

    def test(self, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> bool:
        return word_2.start - word_1.end >= self._min_gap_threshold

    def action(self, word_1: TranscriptionWordType, word_2: TranscriptionWordType) -> Sequence[TranscriptionWordType]:
        return [word_1.model_copy(update={"end": word_1.end+self._extend_by}), word_2]

class SubtitleChunkBase(BaseModel):
    id: int = 0
    start: float
    end: float
    @property
    def duration(self):
        return self.end - self.start

class SubtitleChunk(SubtitleChunkBase):
    items: MutableSequence[TranscriptionWordType]
    @classmethod
    def with_items(cls, items: MutableSequence[TranscriptionWordType]):
        starts, ends = zip(*[(item.start, item.end) for item in items])
        return cls(start=min(starts), end=max(ends), items=items)
    @property
    def words(self) -> MutableSequence[TranscriptionWord]:
        return [word for word in self.items if isinstance(word, TranscriptionWord)]
    @overload
    def __getitem__(self, key: int) -> TranscriptionWordType: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[TranscriptionWordType]: ...
    def __getitem__(self, key: int|slice) -> TranscriptionWordType|Sequence[TranscriptionWordType]:
        return self.items[key]
    def __setitem__(self, key, value):
        self.items[key] = value
        return
    def __delitem__(self, key: int|slice):
        del self.items[key]
    def __len__(self):
        return len(self.items)

class MultilineSubtitleChunk(SubtitleChunkBase):
    @dataclass
    class IndexMap:
        one2two: list[tuple[int, int]]
        two2one: dict[tuple[int, int], int]
        len: int
        @classmethod
        def from_lines(cls, lines: Sequence[SubtitleChunk]):
            one2two = [(i, j) for i, line in enumerate(lines) for j in range(len(line.words))]
            two2one = {item: i for i, item in enumerate(one2two)}
            return cls(one2two=one2two, two2one=two2one, len=len(one2two))
    lines: MutableSequence[SubtitleChunk]
    _item_index_map: IndexMap = PrivateAttr()
    @classmethod
    def with_chunks(cls, chunks: MutableSequence[SubtitleChunk]):
        starts, ends = zip(*[(chunk.start, chunk.end) for chunk in chunks])
        return cls(start=min(starts), end=max(ends), lines=chunks)
    def _populate_index_map(self):
        self._item_index_map = self.IndexMap.from_lines(self.lines)
    @overload
    def __getitem__(self, key: int) -> TranscriptionWordType: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> TranscriptionWordType: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[Sequence[TranscriptionWordType]]: ...
    def __getitem__(self, key: int|tuple[int, int]|slice) -> TranscriptionWordType|Sequence[Sequence[TranscriptionWordType]]:
        if isinstance(key, int):
            i, j = self._item_index_map.one2two[key]
            return self.lines[i][j]
        if isinstance(key, tuple):
            return self.lines[key[0]][key[1]]
        if isinstance(key, slice):
            coords = self._item_index_map.one2two[key]
            groups = {}
            for i, j in coords:
                groups.setdefault(i, []).append(j)
            return [[self.lines[i][j] for j in sorted(group)] for i, group in sorted(groups.items())]
    def __setitem__(self,
        key: int|tuple[int, int]|slice,
        value: TranscriptionWordType|Sequence[TranscriptionWordType],
    ):
        if isinstance(key, (int, tuple)):
            if isinstance(value, TranscriptionWordType):
                if isinstance(key, int):
                    i, j = self._item_index_map.one2two[key]
                    self.lines[i][j] = value
                if isinstance(key, tuple):
                    self.lines[key[0]][key[1]] = value
            else:
                raise TypeError("value must be a TranscriptionWordType when key is an int or tuple")
        if isinstance(key, slice):
            coords = self._item_index_map.one2two[key]
            if isinstance(value, Sequence):
                coord_len = len(coords)
                val_len = len(value)
                if coord_len <= val_len:
                    for (i, j), item in zip(coords, value):
                        self.lines[i][j] = item
                    if coord_len < val_len:
                        i, j = coords[-1]
                        self.lines[i][j:j] = value[coord_len:]
                else:
                    for (i, j), item in zip(coords[:val_len], value):
                        self.lines[i][j] = item
                    for (i, j) in reversed(coords[val_len:]):
                        del self.lines[i][j]
                        if not self.lines[i].items:
                            del self.lines[i]
            else:
                for i, j in coords:
                    self.lines[i][j] = value

        self._populate_index_map()
        self.start = self.lines[0].items[0].start
        self.end = self.lines[-1].items[-1].end

    def __delitem__(self, key: int|slice):
        if isinstance(key, int):
            coords = [self._item_index_map.one2two[key]]
        elif isinstance(key, tuple):
            coords = [key]
        elif isinstance(key, slice):
            coords = self._item_index_map.one2two[key]
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

        for (i, j) in reversed(coords):
            del self.lines[i][j]
            if not self.lines[i].items:
                del self.lines[i]

        self._populate_index_map()
        self.start = self.lines[0].items[0].start
        self.end = self.lines[-1].items[-1].end

    def __len__(self):
        return sum([len(chunk) for chunk in self.lines])



type TranscriptionPart = TranscriptionResult|TranscriptionSegment|Iterable[TranscriptionResult|TranscriptionSegment|TranscriptionWordType]
@dataclass
class SubtitleChunker:
    transcription: TranscriptionPart
    max_line_length: int = 30
    line_limit_is_chars: bool = False
    # max_chars_per_line: int|None = None
    max_lines_per_chunk: int = 1
    strict_line_length: bool = True
    subtitle_preprocessors: Sequence[SubtitleProcessor] = field(default_factory=list)
    chunk_preprocessors: Sequence[SubtitleProcessor] = field(default_factory=list)
    def chunks(self) -> Generator[SubtitleChunk|MultilineSubtitleChunk]:
        if isinstance(self.transcription, TranscriptionResult):
            segments = self.transcription.segments
        elif isinstance(self.transcription, TranscriptionSegment):
            segments = [self.transcription]
        elif isinstance(self.transcription, Iterable):
            def breakout(item):
                if isinstance(item, TranscriptionResult):
                    return item.segments
                elif isinstance(item, TranscriptionSegment):
                    return [item]
                else:
                    return []
            segments = [i for item in self.transcription for i in breakout(item)]
        else:
            raise ValueError(f"Unexpected transcription type: {type(self.transcription)}")

        def stream_chunks():
            for i, segment in enumerate(segments):
                lines = []
                current_line = []
                current_line_chars = 0
                next_segment = segments[i+1] if i < len(segments)-1 else None
                words = segment.words
                if self.subtitle_preprocessors:
                    for proc in self.subtitle_preprocessors:
                        words = proc.transform_words(words)
                for word in words:
                    if (
                        self.line_limit_is_chars and current_line_chars + len(word.word) >= self.max_line_length
                    ) or len(current_line) >= self.max_line_length:
                        if (
                            not self.line_limit_is_chars
                        ) or (
                            current_line_chars >= self.max_line_length
                        ) or self.strict_line_length:
                            lines.append(SubtitleChunk.with_items(current_line))
                            current_line = []
                            current_line_chars = 0
                            if len(lines) == self.max_lines_per_chunk:
                                yield lines[0] if len(lines) == 1 else MultilineSubtitleChunk.with_chunks(lines)
                                lines = []
                    current_line.append(word)
                if current_line:
                    lines.append(SubtitleChunk.with_items(current_line))
                if lines:
                    yield lines[0] if len(lines) == 1 else MultilineSubtitleChunk.with_chunks(lines)

        output = stream_chunks()
        if self.chunk_preprocessors:
            for proc in self.chunk_preprocessors:
                output = proc.transform_words(output)

        yield from output

    def chunks_ignore_segments(self) -> Generator[SubtitleChunk]|Generator[MultilineSubtitleChunk]:
        def iter_words(top_part: TranscriptionPart):
            if isinstance(top_part, TranscriptionResult):
                top_part = top_part.segments
            if isinstance(top_part, TranscriptionSegment):
                top_part = [top_part]
            if isinstance(top_part, Iterable):
                for part in top_part:
                    if isinstance(part, TranscriptionSegment):
                        words = part.words
                        if self.subtitle_preprocessors:
                            for proc in self.subtitle_preprocessors:
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
                        lines.append(SubtitleChunk.with_items(current_line))
                        current_line = []
                        current_line_chars = 0
                        if len(lines) == self.max_lines_per_chunk:
                            yield lines[0] if len(lines) == 1 else MultilineSubtitleChunk.with_chunks(lines)
                            lines = []
                current_line.append(word)
                word = next(words)
            except StopIteration:
                if current_line:
                    lines.append(SubtitleChunk.with_items(current_line))
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
