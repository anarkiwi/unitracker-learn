"""The tokenization study of the document forms ``tools/corpus.py forms`` writes: tokens per
document, tick, event and character by form and encoder, each lexical category's tokens, and
each printer transform's saving (docs/tokenization.md).
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pysidtracker.jobs import imap
from tqdm import tqdm

import corpus as lift
from unitracker.optimiser import tokenizer as tk

ROOT = Path(__file__).resolve().parents[1]
WRAPS = ("_SEP", "_LB", "_LC", "ADD", "MOD", "FLOORDIV", "CMP")
_WRAP = re.compile(r"\n +")
_INDENT = re.compile(r"(?<=\n) +")
_LONG = re.compile(r"\d{3,}")
_LONG_ARG = re.compile(r"(?<=[(,+-])\d{3,}")
_NOTE = re.compile(r"([A-G]#?)(-?\d+)")
_KEY = re.compile(r"([\s-]*)([a-z_][a-z_.]*): $")


def _edits(lexemes, pred, repl) -> list:
    return [(t.start_pos, t.end_pos, repl(t)) for t, ctx in lexemes if pred(t, ctx)]


def comma_space(_text, _tree, lexemes) -> list:
    """``, `` between flow items written ``,``."""
    return _edits(lexemes, lambda t, _: t.type == "_SEP" and t == ", ", lambda _: ",")


def colon_space(_text, _tree, lexemes) -> list:
    """A key's ``: `` written ``:``."""
    return _edits(lexemes, lambda t, c: t.endswith(": ") and c != "text", lambda t: t[:-1])


def no_wrap(_text, _tree, lexemes) -> list:
    """Flow collections and expressions on one line: a line wrap written as a space (none after
    an opening bracket)."""
    return _edits(
        lexemes,
        lambda t, _: t.type in WRAPS and "\n" in t,
        lambda t: _WRAP.sub("" if t.type in ("_LB", "_LC") else " ", t),
    )


def indent_one(text, _tree, _lexemes) -> list:
    """One space of indentation per level instead of two."""
    return [(m.start(), m.end(), " " * (len(m.group()) // 2)) for m in _INDENT.finditer(text)]


def hexadecimal(_text, _tree, lexemes) -> list:
    """Integers of three or more decimal digits, a call's arguments too, written ``$`` and
    hexadecimal."""

    def repl(t):
        pattern = _LONG_ARG if t.type in tk.ARGUED else _LONG
        return pattern.sub(lambda m: f"${int(m.group()):X}", t)

    return _edits(
        lexemes,
        lambda t, _: (t.type in tk.NUMBERS and _LONG.search(t))
        or (t.type in tk.ARGUED and _LONG_ARG.search(t)),
        repl,
    )


def delta_tables(_text, tree, _lexemes) -> list:
    """Each ``tables:`` list written as its first value and then each value's difference from
    the one before."""
    out = []
    for section in tree.find_data("tables"):
        for ints in section.find_data("intlist"):
            values = [t for t in ints.children if getattr(t, "type", "") == "INT"]
            out += [
                (b.start_pos, b.end_pos, f"{int(b) - int(a):+d}")
                for a, b in zip(values, values[1:])
            ]
    return out


def short_keys(_text, _tree, lexemes) -> list:
    """Each key written as the shortest prefix of it no other key of the document has."""
    found = [(t, _KEY.match(t)) for t, c in lexemes if c != "text"]
    found = [(t, m) for t, m in found if m]
    keys = sorted({m.group(2) for _, m in found})
    short = {}
    for key in keys:
        n = 1
        while n < len(key) and any(o != key and o.startswith(key[:n]) for o in keys):
            n += 1
        short[key] = key[:n]
    return [(t.start_pos, t.end_pos, f"{m.group(1)}{short[m.group(2)]}: ") for t, m in found]


def sharp_lower(_text, _tree, lexemes) -> list:
    """A sharp note's letter and ``#`` written as the letter in lower case (``A#2``: ``a2``)."""
    return _edits(
        lexemes,
        lambda t, _: t.type in tk.NOTES and "#" in t,
        lambda t: t[0].lower() + t[2:],
    )


def octave_elide(_text, tree, _lexemes) -> list:
    """A phrase's note written without its octave where it is the previous note's."""
    out = []
    for phrase in tree.find_data("phrase"):
        last = None
        for event in phrase.find_data("event"):
            m = _NOTE.fullmatch(event.children[0])
            if m is None:
                continue
            if m.group(2) == last:
                out.append((event.children[0].start_pos, event.children[0].end_pos, m.group(1)))
            last = m.group(2)
    return out


def command_space(_text, _tree, lexemes) -> list:
    """An event's commands introduced by a space instead of ``!`` (``C4 legato tick.freq:1``)."""
    return _edits(
        lexemes,
        lambda t, c: t.startswith("!") and c != "text",
        lambda t: " " + t[1:],
    )


def code_tab(_text, _tree, lexemes) -> list:
    """A code block's lines indented by a tab instead of four spaces."""
    return _edits(lexemes, lambda t, c: c == "code" and t == "    ", lambda _: "\t")


def notes(text, tree, lexemes) -> list:
    """:func:`sharp_lower` and :func:`octave_elide` together."""
    elided = {e[0]: e for e in octave_elide(text, tree, lexemes)}
    out = [
        (start, end, repl if start not in elided else elided[start][2][0].lower())
        for start, end, repl in sharp_lower(text, tree, lexemes)
    ]
    sharp = {e[0] for e in out}
    return out + [e for start, e in elided.items() if start not in sharp]


TRANSFORMS = {
    "comma-space": comma_space,
    "colon-space": colon_space,
    "no-wrap": no_wrap,
    "indent-1": indent_one,
    "hex": hexadecimal,
    "delta-tables": delta_tables,
    "short-keys": short_keys,
    "sharp-lower": sharp_lower,
    "octave-elide": octave_elide,
    "notes": notes,
    "command-space": command_space,
    "code-tab": code_tab,
}
TOP = 20
COMBINED = ("comma-space", "colon-space", "no-wrap", "notes", "command-space", "code-tab")


def applied(text: str, edits: list) -> str:
    """``text`` with the ``(start, end, replacement)`` edits applied, an edit that overlaps an
    earlier one (in position order) skipped."""
    out, at = [], 0
    for start, end, repl in sorted(edits, key=lambda e: e[:2]):
        if start >= at:
            out += [text[at:start], repl]
            at = end
    return "".join(out) + text[at:]


def _top_tokens(text: str, attributed, starts, ends) -> list:
    top = [Counter() for _ in tk.CATEGORIES]
    for cat, start, end in zip(attributed, starts, ends):
        top[cat][text[start:end]] += 1
    return [dict(c.most_common(TOP)) for c in top]


def measure(text: str, encoders: list) -> dict:
    """A document's characters, events and characters per category, and per encoder its
    tokens, the tokens each category takes and its tokens after each transform; for the first
    encoder, each category's most frequent token strings (:data:`TOP` of them)."""
    tree, lexemes = tk.lex(text)
    cats = tk.categories(text, lexemes)
    edits = {name: fn(text, tree, lexemes) for name, fn in TRANSFORMS.items()}
    edits["combined"] = [e for name in COMBINED for e in edits[name]]
    variants = {name: applied(text, e) for name, e in edits.items()}
    res = {"chars": len(text), "events": tk.events(tree)}
    res["cat_chars"] = np.bincount(cats, minlength=len(tk.CATEGORIES)).tolist()
    for name in encoders:
        enc = tk.encoder(name)
        starts, ends = enc.spans(text)
        attributed = tk.attribute(cats, starts, ends)
        res[name] = {
            "tokens": len(starts),
            "cats": np.bincount(attributed, minlength=len(tk.CATEGORIES)).tolist(),
            "transforms": {k: enc.count(v) for k, v in variants.items()},
        }
        if name == encoders[0]:
            res["top"] = _top_tokens(text, attributed, starts, ends)
    return res


def study_one(doc: str, paths: dict, encoders: list) -> dict:
    """One tune's :func:`measure` per form."""
    forms = {f: measure(Path(p).read_text(encoding="utf-8"), encoders) for f, p in paths.items()}
    return {"doc": doc, "forms": forms}


def corpus(root: Path, forms: list) -> list:
    """``(doc, {form: path}, meta)`` of every tune with a lifted document under ``root``."""
    out = []
    for lifted in sorted((root / "lifted").glob("*.yaml")):
        doc = lifted.stem
        paths = {f: root / f / f"{doc}.yaml" for f in forms}
        meta = root / "meta" / f"{doc}.json"
        out.append(
            (
                doc,
                {f: str(p) for f, p in paths.items() if p.exists()},
                json.loads(meta.read_text()) if meta.exists() else {},
            )
        )
    return out


def _pct(a: float, b: float) -> str:
    return f"{100 * a / b:.1f}%" if b else "-"


def _study(common: list, meta: dict, forms: list, encoders: list) -> str:
    ticks = sum(meta[r["doc"]].get("ticks", 0) for r in common)
    events = sum(r["forms"]["lifted"]["events"] for r in common)
    lines = [
        "| form | tokenizer | tokens/doc mean | median | tokens/tick | tokens/event "
        "| chars/token |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for form in forms:
        chars = sum(r["forms"][form]["chars"] for r in common)
        for name in encoders:
            n = np.array([r["forms"][form][name]["tokens"] for r in common], float)
            lines.append(
                f"| {form} | {Path(name).name} | {n.mean():.0f} | {np.median(n):.0f} "
                f"| {n.sum() / ticks:.3f} | {n.sum() / events:.2f} | {chars / n.sum():.2f} |"
            )
    return "\n".join(lines)


def _categories(common: list, form: str, encoders: list) -> str:
    cat_chars = np.sum([r["forms"][form]["cat_chars"] for r in common], axis=0)
    cols = [np.sum([r["forms"][form][n]["cats"] for r in common], axis=0) for n in encoders]
    head = " | ".join(f"{Path(n).name} share | tokens/char" for n in encoders)
    lines = [f"| category | chars share | {head} |", "|---|---:|" + "---:|---:|" * len(encoders)]
    for i, cat in enumerate(tk.CATEGORIES):
        cells = " | ".join(
            f"{_pct(c[i], c.sum())} | {c[i] / max(1, cat_chars[i]):.2f}" for c in cols
        )
        lines.append(f"| {cat} | {_pct(cat_chars[i], cat_chars.sum())} | {cells} |")
    return "\n".join(lines)


def _transforms(common: list, forms: list, encoders: list) -> str:
    pairs = [(f, n) for f in forms for n in encoders]
    lines = [
        "| transform | " + " | ".join(f"{f} {Path(n).name}" for f, n in pairs) + " |",
        "|---|" + "---:|" * len(pairs),
    ]
    for name in [*TRANSFORMS, "combined"]:
        cells = []
        for form, enc in pairs:
            base = sum(r["forms"][form][enc]["tokens"] for r in common)
            after = sum(r["forms"][form][enc]["transforms"][name] for r in common)
            cells.append(_pct(base - after, base))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _top(common: list, form: str, encoder: str) -> str:
    lines = [f"| category | {Path(encoder).name} tokens (share of the category's) |", "|---|---|"]
    for i, cat in enumerate(tk.CATEGORIES):
        top = sum((Counter(r["forms"][form]["top"][i]) for r in common), Counter())
        total = sum(r["forms"][form][encoder]["cats"][i] for r in common)
        cells = ", ".join(
            f"`{json.dumps(k)[1:-1]}` {_pct(v, total)}" for k, v in top.most_common(8)
        )
        lines.append(f"| {cat} | {cells.replace('|', chr(92) + '|')} |")
    return "\n".join(lines)


def summary(results: list, meta: dict, forms: list, encoders: list) -> dict:
    """The study's markdown tables over the tunes every form has: tokens by form and encoder,
    each form's categories, the first form's top tokens, and each transform's saving."""
    common = [r for r in results if all(f in r["forms"] for f in forms)]
    tables = {"study": _study(common, meta, forms, encoders)}
    tables |= {f"categories {f}": _categories(common, f, encoders) for f in forms}
    tables[f"top tokens {forms[0]}"] = _top(common, forms[0], encoders[0])
    tables["transforms"] = _transforms(common, forms, encoders)
    tables["docs"] = f"{len(common)} tunes with every form of {', '.join(forms)}"
    return tables


def main(argv=None) -> int:
    """Measure the corpus and print and write the study's tables."""
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("root", type=Path, nargs="?", default=Path("out/tokenize"))
    ap.add_argument("--forms", nargs="*", default=list(lift.forms()))
    ap.add_argument("--encoders", nargs="*", default=list(tk.REFERENCES))
    ap.add_argument("--jobs", type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    args = ap.parse_args(argv)
    for name in args.encoders:
        tk.encoder(name)
    tunes = corpus(args.root, args.forms)
    meta = {doc: m for doc, _, m in tunes}
    results, failed = [], []
    tasks = [(doc, paths, args.encoders) for doc, paths, _ in tunes]
    for i, value, error in tqdm(imap(study_one, tasks, args.jobs), total=len(tasks), unit="tune"):
        if error is None:
            results.append(value)
        else:
            failed.append(f"{tunes[i][0]}: {type(error).__name__}: {error}")
    tables = summary(results, meta, args.forms, args.encoders)
    tables["failed"] = "\n".join(failed)
    (args.root / "study.json").write_text(json.dumps(results), encoding="utf-8")
    text = "\n\n".join(f"## {k}\n\n{v}" for k, v in tables.items())
    (args.root / "study.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
