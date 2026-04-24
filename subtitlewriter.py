import re
import sys
from collections.abc import Callable, Iterable, Iterator
import configparser
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from itertools import pairwise
from typing import ClassVar, NamedTuple, Self, TextIO
from functools import partial

@dataclass(frozen=True, slots=True)
class Timecode:
    _rx: ClassVar[re.Pattern] = re.compile(r"(\d+):(\d+):(\d+)[,\.](\d+)")
    _rx2: ClassVar[re.Pattern] = re.compile(rf"^{_rx.pattern} --> {_rx.pattern}$")
    _milli_sep: ClassVar[str] = ","
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    milliseconds: int = 0

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

    @classmethod
    def from_seconds(cls, seconds: int) -> Self:
        return cls.from_timedelta(timedelta(seconds=seconds))

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
        return timedelta(**asdict(self))

    def as_timedelta_slower(self) -> timedelta:
        today = date.today()
        return datetime.combine(today, self.as_time()) - datetime.combine(today, time())

    def __lt__(self, other: Self):
        return self.as_time().__lt__(other.as_time())

    def __eq__(self, other: Self):
        return self.as_time().__eq__(other.as_time())

    def __hash__(self):
        return hash(str(self))

    def __add__(self, other: Self | time | timedelta | str) -> Self | timedelta:
        me = datetime.combine(date.today(), self.as_time())
        if isinstance(other, time):
            other = type(self).from_time(other)
        if isinstance(other, type(self)):
            other = other.as_timedelta()
        if isinstance(other, timedelta):
            return type(self).from_time((me + other).time())
        if isinstance(other, str):
            return str(self) + other
        raise TypeError(type(other))
        # return self.__math_op(operator.add, other)

    def __radd__(self, other: Self | time | timedelta | str) -> Self | timedelta:
        if isinstance(other, str):
            return other + str(self)
        return self + other

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
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}{self._milli_sep}{self.milliseconds:03d}"


@dataclass
class SubtitleEntry:
    number: int
    time_range: tuple[Timecode, Timecode]
    text: str
    highlight_word: str|None = None
    highlight_idx: int|None = None

    @property
    def start(self):
        return self.time_range[0]

    @property
    def end(self):
        return self.time_range[1]

    @property
    def text_parts(self):
        if None not in (self.highlight_word, self.highlight_idx):
            return (
                self.text[:self.highlight_idx],
                self.highlight_word,
                self.text[self.highlight_idx+len(self.highlight_word):]
            )

@dataclass
class AssEntry(SubtitleEntry):
    effect: str|None = None
    layer: int = 0
    margin_l: int = 0
    margin_r: int = 0
    margin_v: int = 0
    name: str|None = None
    style: str = "Default"

class HighlightTag(NamedTuple):
    open: str
    close: str

@dataclass
class SubtitleFile:
    entries: Iterable[SubtitleEntry]

    @classmethod
    def from_text(cls, buffer: TextIO, highlight_tag: HighlightTag = None):
        return cls(list(cls.parse_srt(buffer, highlight_tag)))

    @staticmethod
    def parse_srt(buffer: TextIO, highlight_tag: HighlightTag = None):
        # entries = []
        def new_entry(num, timecodes, text):
            if highlight_tag:
                highlight_start = text.find(highlight_tag.open)
                if highlight_start is not None:
                    highlight_end = text.find(highlight_tag.close)
                    if highlight_end is not None:
                        word = text[highlight_start+len(highlight_tag.open):highlight_end]
                        new_text = text[:highlight_start] + word + text[highlight_end+len(highlight_tag.close):]
                        return SubtitleEntry(num, timecodes, new_text, word, highlight_start)
            return SubtitleEntry(num, timecodes, text)
        i = 0
        error = False
        num, timecodes, text = (None, None, [])
        for line in buffer:
            line = line.strip()
            if not line:
                if not error:
                    # entries.append(SubtitleEntry(num, timecodes, "\n".join(text)))
                    yield new_entry(num, timecodes, "\n".join(text))
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
            yield new_entry(num, timecodes, "\n".join(text))
        # return entries


    def dump_srt(self, buffer: TextIO, highlight_tag: HighlightTag = None):
        for entry in self.entries:
            print(str(entry.number), file=buffer)
            print(f"{entry.start} --> {entry.end}", file=buffer)
            if None not in (highlight_tag, entry.highlight_word, entry.highlight_idx):
                before, word, after = entry.text_parts
                print(before + highlight_tag.open + word + highlight_tag.close + after, file=buffer)
            else:
                print(entry.text, file=buffer)
            print("", file=buffer)

    class AppendDict(dict):
        def __setitem__(self, key, val):
            if key in self:
                if isinstance(self[key], list):
                    if isinstance(val, list):
                        self[key].extend(val)
                    # else:
                    #     self[key].append(val)
                else:
                    super().__setitem__(key, [self[key], val])
            else:
                super().__setitem__(key, val)

@dataclass
class AssFile:
    property_name_map: ClassVar[tuple[tuple[str, str], ...]] = (
        ("effect", "Effect"),
        ("layer", "Layer"),
        ("margin_l", "MarginL"),
        ("margin_r", "MarginR"),
        ("margin_v", "MarginV"),
        ("name", "Name"),
        ("style", "Style"),
        # ("text", "Text")
    )
    default_config: ClassVar[dict[str, dict[str, list[str]]]] = {
        "Script Info": {
            'ScriptType': ['v4.00+'],
            'PlayResX': ['384'],
            'PlayResY': ['288'],
            'ScaledBorderAndShadow': ['yes'],
            'YCbCr Matrix': ['None', '']
        },
        "V4+ Styles": {
            'Format': ['Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding'],
            'Style': ['Default,Arial,16,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1',]
        },
        "Events": {
            "Format": ['Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
        }
    }
    config: dict|configparser.ConfigParser|None = None
    entries: Iterable[AssEntry]|None = field(default_factory=list)

    @classmethod
    def from_text(cls, buffer: TextIO, highlight_tag: HighlightTag|re.Pattern = None):
        parser = configparser.ConfigParser(strict=False, dict_type=SubtitleFile.AppendDict, interpolation=None, delimiters=(":",))
        parser.optionxform = lambda o: o
        parser.read_file(buffer)
        return cls(parser, list(cls.parse_ass_events(parser["Events"], highlight_tag=highlight_tag)))

    @staticmethod
    def parse_ass_events(events: dict, highlight_tag: HighlightTag|re.Pattern = None):
        # entries = []
        def new_entry(**params):
            properties = {
                "number": params["number"],
                "time_range": (Timecode.from_string(params["Start"]), Timecode.from_string(params["End"])),
            }
            for key1, key2 in AssFile.property_name_map:
                properties[key1] = params[key2]
                # "layer":    params["Layer"],
                # "margin_l": params["MarginL"],
                # "margin_r": params["MarginR"],
                # "margin_v": params["MarginV"],
                # "name":     params["Name"],
                # "style":    params["Style"]
            text = params["Text"]
            if highlight_tag:
                if isinstance(highlight_tag, HighlightTag):
                    highlight_start = text.find(highlight_tag.open)
                    if highlight_start is not None:
                        highlight_end = text.find(highlight_tag.close)
                        if highlight_end is not None:
                            word = text[highlight_start+len(highlight_tag.open):highlight_end]
                            new_text = text[:highlight_start] + word + text[highlight_end+len(highlight_tag.close):]
                            return AssEntry(**properties, text=new_text, highlight_word=word, highlight_idx=highlight_start)
                elif isinstance(highlight_tag, re.Pattern):
                    if {"open", "close", "word"} - highlight_tag.groupindex.keys():
                        raise ValueError("RegEx must have groups for 'open', 'close', and 'word'")
                    srch = highlight_tag.search(text)
                    if srch:
                        new_text = text[:srch.start()] + srch["word"] + text[srch.end():]
                        return AssEntry(**properties, text=new_text, highlight_word=srch["word"], highlight_idx=srch.start())
                    # else:
                    #     num = properties["number"]
                    #     tm = properties["time_range"]
                    #     print(f"NO MATCH: ({num} - {tm}) {text}")
            return AssEntry(**properties, text=text)

        start_times = set()
        dialog_keys = events["Format"][0].split(", ")
        for idx, dialog in enumerate(events["Dialogue"], 1):
            params = dict(zip(dialog_keys, dialog.split(",", len(dialog_keys)-1)))
            entry = new_entry(**params, number=idx)
            if entry.time_range[0] not in start_times:
                start_times.add(entry.time_range[0])
                yield entry

    def dump_ass(self, buffer: TextIO, highlight_tag: HighlightTag = None):
        print("[Script Info]", file=buffer)
        print("; comments go here", file=buffer)
        config = self.config or self.default_config
        if "Script Info" in config:
            for key, val in config["Script Info"].items():
                print(f"{key}: {val[0]}", file=buffer)
        print("", file=buffer)
        if "V4+ Styles" in config:
            print("[V4+ Styles]", file=buffer)
            for key, val in config["V4+ Styles"].items():
                print(f"{key}: {val[0]}", file=buffer)
            print("", file=buffer)
        if "Events" in config and "Format" in config["Events"]:
            print("[Events]", file=buffer)
            print(f"Format: {config['Events']['Format'][0]}", file=buffer)
        fmt_keys = config["Events"]["Format"][0].split(", ")
        key_map = {key2: key1 for key1, key2 in self.property_name_map}
        for entry in self.entries:
            vals = {key: getattr(entry, key_map[key]) for key in fmt_keys if key in key_map}
            vals["Start"] = str(entry.time_range[0])
            vals["End"] = str(entry.time_range[1])
            if None not in (highlight_tag, entry.highlight_word, entry.highlight_idx):
                before, word, after = entry.text_parts
                vals["Text"] = f"{before}{highlight_tag.open}{word}{highlight_tag.close}{after}"
            else:
                vals["Text"] = entry.text
            vals = [vals[key] for key in fmt_keys]
            print(f"Dialogue: {','.join(vals)}", file=buffer)
