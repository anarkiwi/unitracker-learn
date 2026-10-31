"""Tunes by path, HVSC name or tune list, and their songs' names."""

import csv
import hashlib
import re
from pathlib import Path

from pygoattracker import decompile_sid
from pysidtracker import find_format, parse_sid_header, resolve_tune
from pysidtracker.errors import SidError


def tune(source, hvsc=None) -> Path:
    """The ``.sid`` file ``source``: a local path, else a path under ``hvsc``, else an HVSC name
    resolved through ``$HVSC``, pysidtracker's tune cache or the mirror."""
    path = Path(source)
    if path.is_file():
        return path
    if hvsc is not None and (Path(hvsc) / source).is_file():
        return Path(hvsc) / source
    found = resolve_tune(str(source))
    if found is None:
        raise FileNotFoundError(f"{source}: no such tune")
    return Path(found)


def multi_sid(raw: bytes) -> bool:
    """Whether the tune ``raw`` plays more than one SID: its header's 2nd SID address (PSID v3
    on) or 3rd (v4 on), or a GoatTracker 2 build with a channel past the first SID's three
    voices."""
    try:
        head = parse_sid_header(raw)
        if head.version >= 3 and head.second_sid or head.version >= 4 and head.third_sid:
            return True
        if find_format(raw).name != "goattracker2":
            return False
        voices = decompile_sid(raw).info.playback.voices or ()
    except SidError:
        return False
    return len(voices) > 3 or any(off is not None and off >= 0x20 for off in voices)


def name(row: dict) -> str:
    """A song's output name: its tune's file stem, after its md5's first 8 digits when it has
    one, then ``-s<k>`` for a 0-based ``subtune`` k past the first."""
    stem = Path(row["path"]).stem
    k = int(row.get("subtune") or 0)
    return (f"{row['md5'][:8]}-{stem}" if row.get("md5") else stem) + f"-s{k}" * bool(k)


def songs(tunes, subtune: int | str = 0, min_seconds: float = 0) -> list:
    """Tune rows' songs, each its row with its 0-based ``subtune`` (``"all"``: each of the
    tune's ``songs``), but those its ``lengths`` (seconds per song) put under ``min_seconds``."""
    out = []
    for row in tunes:
        lengths = (row.get("lengths") or "").split()
        ks = range(int(row.get("songs") or 1)) if subtune == "all" else [int(subtune)]
        out += [
            row | {"subtune": k}
            for k in ks
            if k >= len(lengths) or float(lengths[k]) >= min_seconds
        ]
    return out


def rows(sources, player: str = "", subtunes: int | None = None) -> list:
    """Tune rows (``path`` and any other columns) of tune paths and of ``.csv`` tune lists (a
    survey ``find`` list or an earlier result table) and of directories' ``.sid`` files (with
    their md5), the lists' rows filtered by their SIDId ``player`` (a regex) and ``songs``."""
    out = []
    for source in sources:
        if Path(source).is_dir():
            out += [
                {"path": str(p), "md5": hashlib.md5(p.read_bytes()).hexdigest()}
                for p in sorted(Path(source).rglob("*.sid"))
            ]
            continue
        if not str(source).endswith(".csv"):
            out.append({"path": str(source)})
            continue
        with open(source, newline="", encoding="utf-8") as fh:
            out += [
                r
                for r in csv.DictReader(fh)
                if re.search(player, r.get("player", ""), re.I)
                and (subtunes is None or int(r.get("songs") or 1) <= subtunes)
            ]
    return out
