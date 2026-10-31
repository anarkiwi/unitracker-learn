"""Each tune's document forms (lifted and optimised) under a corpus directory, cached and
verified against the reference player; the forms the learnability and tokenization studies read.

A tune is a path, a name under ``--hvsc`` or an HVSC name; a ``.csv`` is a survey ``find`` list
or an earlier result table. Usage: ``unitracker-corpus TUNES... [--out DIR] [--forms ...]``.
"""

import argparse
import csv
import functools
import inspect
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path

from pysidtracker.jobs import deadline, imap
from tqdm import tqdm
from unitracker import cli
from unitracker.optimiser import PASSES, Guarantee, optimise
from unitracker.optimiser.inline import MODES, inline
from unitracker_grammar.document import printed, read
from unitracker_player.player import Player, load

from unitracker_learn import catalog

LIBRARIES = ("unitracker", "pysidtracker", "unitracker-grammar", "unitracker-player")
ROOT = Path(__file__).resolve().parents[2]


def failure(exc: Exception) -> str:
    """An exception's one line: its type and its message's first line."""
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"


def write_failure(path: Path, line: str) -> None:
    """``path``: an exception's ``line``, then the traceback being handled."""
    path.write_text(f"{line}\n{traceback.format_exc()}", encoding="utf-8")


@functools.cache
def versions() -> dict:
    """The unitracker-learn commit (``UNITRACKER_LEARN_COMMIT``, else git's, else its package
    version, else None) and its libraries' installed versions."""
    git = ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    commit = os.environ.get("UNITRACKER_LEARN_COMMIT")
    try:
        commit = commit or subprocess.run(git, capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        try:
            commit = metadata.version("unitracker-learn")
        except metadata.PackageNotFoundError:
            commit = None
    out = {"unitracker-learn": commit and commit.strip()}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    return out


def verified(raw: bytes, subtune: int, frames: int) -> dict:
    """Whether a song's document plays the reference grid over ``frames`` ticks and its cells that
    differ, or the exception verifying it raised."""
    try:
        bad = len(cli.verify(raw, subtune, frames))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {"verified": False, "verify_error": failure(exc)}
    return {"verified": not bad, "bad_cells": bad}


def forms(ablations: bool = True) -> dict:
    """``{form: (base form, optimise keywords)}`` of the forms this optimiser writes, each after
    its base: the lifted document, the named form, each other inline mode of it, and where the
    optimiser has a lossy mode, the lossy form at its default tolerances and its inline modes,
    and with ``ablations`` the same without phrase templates (``lossy-nophrases``, its passes
    but ``phrases``)."""
    modes = [m for m in MODES if m != "named"]
    out = {"lifted": (None, None), "named": ("lifted", {})}
    out |= {m: ("named", {"inline": m}) for m in modes}
    if "lossy" in inspect.signature(optimise).parameters:
        from unitracker.optimiser.fit import TOLERANCES  # pylint: disable=C0415

        out["lossy"] = ("lifted", {"lossy": dict(TOLERANCES)})
        out |= {f"lossy-{m}": ("lossy", {"inline": m}) for m in modes}
        if not ablations:
            return out
        order = tuple(p for p in PASSES if p[0] != "phrases")
        out["lossy-nophrases"] = ("lifted", {"lossy": dict(TOLERANCES), "order": order})
        out |= {f"lossy-nophrases-{m}": ("lossy-nophrases", {"inline": m}) for m in modes}
    return out


def with_bases(names) -> list:
    """The forms ``names`` and the forms they derive from, in :func:`forms` order."""
    table, need = forms(), set(names)
    for name in reversed(list(table)):
        if name in need and table[name][0]:
            need.add(table[name][0])
    return [n for n in table if n in need]


def song_ticks(text: str, limit: int) -> int:
    """The ticks a document plays to its end or until a row starts at or before the row that
    started last, its loop (at most ``limit``)."""
    player, starts = Player(load(text)), []
    seq = player._seq  # pylint: disable=protected-access
    start_row = seq._start_row  # pylint: disable=protected-access

    def counted(t, ending):
        start_row(t, ending)
        starts.append((seq.row, t))

    seq._start_row = counted  # pylint: disable=protected-access
    while player.tick < limit and player.step() is not None:
        if len(starts) > 1 and starts[-1][0] <= starts[-2][0]:
            return starts[-1][1]
    return player.tick


def _form(name: str, text: dict, opts: dict) -> str:
    """Form ``name``'s text from its base's: an inline mode of it, else it optimised."""
    base, spec = forms()[name]
    if set(spec) == {"inline"}:
        holds = Guarantee(text[base], opts["ticks"]).holds
        return printed(inline(read(text[base]), spec["inline"], holds)[0])
    return optimise(text[base], opts["ticks"], **spec).text


def forms_one(row: dict, opts: dict) -> dict | None:
    """One song's forms under ``out/<form>/<doc>.yaml``, each kept from an earlier run, a failed
    one's exception and traceback (a timeout's from the start) in ``<doc>.err``, and
    ``out/meta/<doc>.json``: its format, versions, song length in ticks and verification; None
    for a multi-SID tune it lifts unless ``opts["multi"]``."""
    out, doc, subtune = Path(opts["out"]), catalog.name(row), int(row.get("subtune") or 0)
    meta, text = out / "meta" / f"{doc}.json", {}
    res = {"doc": doc} | dict.fromkeys(opts["forms"], "")
    for name in opts["forms"]:
        path = out / name / f"{doc}.yaml"
        err = path.with_suffix(".err")
        if path.exists():
            text[name], res[name] = path.read_text(encoding="utf-8"), "ok"
            continue
        if err.exists() and not opts["retry"]:
            res[name] = err.read_text(encoding="utf-8").split("\n", 1)[0]
            continue
        base = forms()[name][0]
        if base and base not in text:
            res[name] = f"no {base} form"
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(f"TimeoutError: exceeded {opts['timeout']}s\n", encoding="utf-8")
        try:
            with deadline(opts["timeout"]):
                if name == "lifted":
                    raw = catalog.tune(row["path"], opts["hvsc"]).read_bytes()
                    if not opts.get("multi") and catalog.multi_sid(raw):
                        err.unlink()
                        return None
                    meta.parent.mkdir(parents=True, exist_ok=True)
                    rec = {"player": cli.lifter(raw)[0], "versions": versions()}
                    meta.write_text(json.dumps(rec), "utf-8")
                    text[name] = cli.lift(raw, subtune)
                    rec["ticks"] = song_ticks(text[name], opts["limit"])
                    meta.write_text(json.dumps(rec | verified(raw, subtune, rec["ticks"])))
                else:
                    text[name] = _form(name, text, opts)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            res[name] = failure(exc)
            write_failure(err, res[name])
            continue
        path.write_text(text[name], encoding="utf-8")
        err.unlink()
        res[name] = "ok"
    return res | (json.loads(meta.read_text()) if meta.exists() else {})


def write_forms(args) -> int:
    """The ``forms`` command: each song's forms and ``forms.csv``, its forms' status."""
    opts = {"out": str(args.out), "hvsc": args.hvsc, "timeout": args.timeout}
    opts.update(ticks=args.ticks, limit=args.limit, retry=args.retry)
    opts.update(forms=with_bases(args.forms or forms()), multi=args.multi_sid)
    tunes = catalog.rows(args.tunes, args.player, args.subtunes)
    tunes = catalog.songs(tunes, args.subtune, args.min_seconds)
    rows = []
    for i, value, error in tqdm(
        imap(forms_one, [(r, opts) for r in tunes], args.jobs), total=len(tunes), unit="sid"
    ):
        if value or error is not None:
            rows.append(value or {"doc": catalog.name(tunes[i]), "lifted": repr(error)})
    print(f"{len(tunes) - len(rows)} multi-SID tunes excluded")
    print(f"{sum(r.get('verified') is True for r in rows)}/{len(rows)} verified")
    rows.sort(key=lambda r: r["doc"])
    fields = ["doc", "player", "ticks", "verified", *opts["forms"]]
    with (args.out / "forms.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for name in opts["forms"]:
        done = Counter(r.get("player", "") for r in rows if r.get(name) == "ok")
        print(f"{name}: {sum(done.values())}/{len(rows)} {dict(sorted(done.items()))}")
    return 0


def _subtune(value: str) -> int | str:
    """A ``--subtune``: 0-based, or ``all``."""
    return value if value == "all" else int(value)


def main(argv=None) -> int:
    """Parse the command line and write the forms."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0], fromfile_prefix_chars="@")
    ap.add_argument("tunes", nargs="+", metavar="TUNE.sid|LIST.csv|DIR")
    ap.add_argument("--hvsc", type=Path, default=os.environ.get("HVSC"))
    ap.add_argument("--out", type=Path, default=Path("out/tokenize"))
    ap.add_argument("--subtune", type=_subtune, default=0, help="0-based, or all")
    ap.add_argument("--min-seconds", type=float, default=0, help="shortest song length kept")
    ap.add_argument("--forms", nargs="*", help=f"of {', '.join(forms())} (default all)")
    ap.add_argument("--ticks", type=int, default=3000, help="the optimiser's guarantee horizon")
    ap.add_argument("--limit", type=int, default=100000, help="most ticks of a song's length")
    ap.add_argument("--player", default="", help="a list's SIDId players (regex)")
    ap.add_argument("--subtunes", type=int, help="a list's tunes with at most this many")
    ap.add_argument("--timeout", type=float, default=600, help="seconds per form (0: none)")
    ap.add_argument("--retry", action="store_true", help="retry forms that failed before")
    ap.add_argument("--jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    ap.add_argument("--multi-sid", action="store_true", help="include multi-SID tunes")
    return write_forms(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
