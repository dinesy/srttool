#!/usr/bin/env python3

import re
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from itertools import pairwise
from typing import NamedTuple, Self, TextIO


class Timecode(NamedTuple):
    _rx = re.compile(r"(\d+):(\d+):(\d+),(\d+)")
    _rx2 = re.compile(rf"^{_rx.pattern} --> {_rx.pattern}$")
    hours: int
    minutes: int
    seconds: int
    milliseconds: int

    @classmethod
    def from_string(cls, string: str) -> Self:
        srch = cls._rx.search(string)
        if srch:
            return cls(*map(int, srch.groups()))
        raise ValueError(string)

    @classmethod
    def range_from_string(cls, string: str) -> tuple[Self, Self]:
        srch = cls._rx2.search(string)
        if srch:
            return cls(*map(int, srch.group(1, 2, 3, 4))), cls(
                *map(int, srch.group(5, 6, 7, 8))
            )
        raise ValueError(string)

    @classmethod
    def from_time(cls, td: time) -> Self:
        return cls(
            hours=td.hour,
            minutes=td.minute,
            seconds=td.second,
            milliseconds=td.microsecond // 1000,
        )

    @classmethod
    def from_timedelta(cls, td: timedelta) -> Self:
        newtime = (datetime.combine(date.today(), time()) + td).time()
        return cls.from_time(newtime)

    def as_time(self) -> time:
        return time(
            hour=self.hours,
            minute=self.minutes,
            second=self.seconds,
            microsecond=self.milliseconds * 1000,
        )

    def as_timedelta(self) -> timedelta:
        return timedelta(
            hours=self.hours,
            minutes=self.minutes,
            seconds=self.seconds,
            milliseconds=self.milliseconds,
        )

    def as_timedelta_slow(self) -> timedelta:
        return timedelta(**self._asdict())

    def as_timedelta_slower(self) -> timedelta:
        today = date.today()
        return datetime.combine(today, self.as_time()) - datetime.combine(today, time())

    def __cmp__(self, other: Self):
        return self.as_time().__cmp__(other.as_time())

    # def __math_op(
    #     self, op: callable, other: Self | time | timedelta
    # ) -> Self | timedelta:
    #     othertype = type(other)
    #     me = datetime.combine(date.today(), self.as_time())
    #     if isinstance(other, type(self)):
    #         other = other.as_timedelta()
    #     elif isinstance(other, time):
    #         other = self.from_time(other).as_timedelta()
    #     if isinstance(other, timedelta):
    #         result = self.from_time(op(me, other).time())
    #     else:
    #         raise TypeError(othertype)

    #     if othertype is timedelta:
    #         return result
    #     else:
    #         return result.as_timedelta()

    def __add__(self, other: Self | time | timedelta) -> Self | timedelta:
        me = datetime.combine(date.today(), self.as_time())
        if isinstance(other, time):
            other = type(self).from_time(other)
        if isinstance(other, type(self)):
            other = other.as_timedelta()
        if isinstance(other, timedelta):
            return type(self).from_time((me + other).time())
        raise TypeError(type(other))
        # return self.__math_op(operator.add, other)

    def __sub__(self, other: Self | time | timedelta) -> Self | timedelta:
        me = datetime.combine(date.today(), self.as_time())
        if isinstance(other, type(self)):
            other = other.as_time()
        if isinstance(other, time):
            return me - datetime.combine(me.date(), other)
        if isinstance(other, timedelta):
            return type(self).from_time((me - other).time())
        raise TypeError(type(other))
        # return self.__math_op(operator.sub, other)

    def __repr__(self):
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d},{self.milliseconds:03d}"


@dataclass
class SRTEntry:
    number: int
    time_range: tuple[Timecode, Timecode]
    text: str

    @property
    def start(self):
        return self.time_range[0]

    @property
    def end(self):
        return self.time_range[1]


def parse_srt(buffer: TextIO):
    # entries = []
    i = 0
    error = False
    num, timecodes, text = (None, None, [])
    for line in buffer:
        line = line.strip()
        if not line:
            if not error:
                # entries.append(SRTEntry(num, timecodes, "\n".join(text)))
                yield SRTEntry(num, timecodes, "\n".join(text))
            i = 0
            error = False
            num, timecodes, text = (None, None, [])
        elif error:
            i += 1
        elif i == 0:
            if line.isdigit():
                num = int(line)
            else:
                error = True
            i += 1
        elif i == 1:
            try:
                timecodes = Timecode.range_from_string(line)
            except ValueError:
                error = True
            finally:
                i += 1
        elif i > 1:
            text.append(line)
            i += 1
        else:
            raise Exception("Why am I here?")
    if num:
        print("last")
        yield SRTEntry(num, timecodes, "\n".join(text))
    # return entries


def dump_srt(entries: Iterable[SRTEntry], buffer: TextIO):
    for entry in entries:
        print(str(entry.number), file=buffer)
        print(f"{entry.start} --> {entry.end}", file=buffer)
        print(entry.text, file=buffer)
        print("", file=buffer)


def extend_trailing_titles(
    entries: Iterable[SRTEntry],
    blank_threshold: timedelta = timedelta(seconds=1),
    extend_by: timedelta = timedelta(seconds=1),
) -> Iterable[SRTEntry]:
    for entry, next in pairwise(entries):
        if next.start - entry.end > blank_threshold:
            new_end = min(entry.end + extend_by, next.start)
            print(
                f"{next.start} - {entry.end} = {next.start - entry.end}  -->  {new_end}"
            )
            time_range = (entry.start, new_end)
        else:
            time_range = entry.time_range
        yield SRTEntry(entry.number, time_range, entry.text)
    yield next


def transform_entry_text(
    entries: Iterable[SRTEntry], transforms: Iterator[Callable[[str], str]]
) -> Iterable[SRTEntry]:
    for entry in entries:
        text = entry.text
        for transform in transforms:
            text = transform(text)
        yield SRTEntry(entry.number, entry.time_range, text)


def make_replacer(old, new):
    def replace(text):
        return text.replace(old, new)

    return replace


def renumber_entries(entries: Iterable[SRTEntry]) -> Iterable[SRTEntry]:
    for i, entry in enumerate(entries, 1):
        yield (SRTEntry(number=i, time_range=entry.time_range, text=entry.text))


def retime_entries(
    entries: Iterable[SRTEntry], scale_factor: float
) -> Iterable[SRTEntry]:
    last = entry = next(entries)
    new_end = Timecode.from_timedelta(entry.end.as_timedelta() * scale_factor)
    yield SRTEntry(
        number=entry.number, time_range=(entry.start, new_end), text=entry.text
    )
    cum_shift = new_end - entry.end

    for entry in entries:
        gap_length = entry.start - last.end
        new_gap = gap_length * scale_factor
        cum_shift += new_gap - gap_length

        new_start = entry.start + cum_shift
        entry_length = entry.end - entry.start
        new_length = entry_length * scale_factor
        new_end = entry.start + cum_shift + new_length

        cum_shift += new_length - entry_length
        last = entry

        yield (
            SRTEntry(
                number=entry.number, time_range=(new_start, new_end), text=entry.text
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_in",
        type=argparse.FileType(),
        help="SRT file to read. '-' or leave blank for stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType(mode="w"),
        help="File to write out. '-' or omit for stdout.",
    )
    parser.add_argument(
        "-x",
        "--extend-trailing",
        nargs="*",
        type=lambda n: timedelta(seconds=n),
        metavar=("blank_threshold", "extend_by"),
        help="If the amount of time between the end of one title and the start of the next is greater than <blank_threshold> seconds (default: 1), "
        "then extend that title for <extend_by> seconds (or to the start of the next title, whichever is less, default: 1, "
        "any more than 2 args are ignored)",
    )
    parser.add_argument(
        "-r",
        "--replace",
        action="append",
        nargs=2,
        default=[],
        metavar=("old", "new"),
        help="Pairs of strings to find and replace in the text of each title (can be given multiple times)",
    )
    parser.add_argument(
        "-n",
        "--renumber",
        action="store_true",
        default=False,
        help="Renumber the titles consecutively (if the source has been modified by hand)",
    )
    parser.add_argument(
        "-t", "--retime", type=float, help="Scale clip timing by factor"
    )

    args = parser.parse_args(sys.argv[1:])
    transforms = [make_replacer(*a) for a in args.replace]

    entries = parse_srt(args.file_in)
    if args.extend_trailing is not None:
        blank_threshold = (
            args.extend_trailing[0]
            if len(args.extend_trailing) > 0
            else timedelta(seconds=1)
        )
        extend_by = (
            args.extend_trailing[1]
            if len(args.extend_trailing) > 1
            else timedelta(seconds=1)
        )
        entries = extend_trailing_titles(entries, blank_threshold, extend_by)

    if transforms:
        entries = transform_entry_text(entries, transforms)

    if args.renumber:
        entries = renumber_entries(entries)

    if args.retime:
        entries = retime_entries(entries, args.retime)

    # e = list(entries)
    # dump_srt(e[-3:], sys.stdout)
    dump_srt(entries, args.output)
