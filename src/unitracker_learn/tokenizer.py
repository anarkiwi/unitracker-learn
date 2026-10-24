"""Token accounting of canonical documents (docs/tokenization.md): characters' grammar categories,
tokens attributed by them, and a byte-level BPE trained on documents keeping lexemes whole.

``python -m unitracker.optimiser.tokenizer train CORPUS... --vocab N --out DIR`` and
``... eval DIR CORPUS...``.
"""

import argparse
import functools
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from lark import Lark, Token, Tree
from tqdm import tqdm
from unitracker_grammar.canonical import GRAMMAR

CATEGORIES = (
    "key",
    "name",
    "note",
    "number",
    "long number",
    "table",
    "code",
    "expr",
    "text",
    "unicode",
    "punct",
    "space",
    "indent",
)
CAT = {c: i for i, c in enumerate(CATEGORIES)}
NAMES = frozenset(
    "INS CTRL PW FILTER RESTART PHRASE XTABLE SIGNATURE PROGRAM_SIG CALL PARAM REL_PARAM "
    "TABLE_BYTE NAME STATE_NAME DEF_INS DEF_RESTART DEF_CTRL DEF_PW DEF_FILTER DEF_PHRASE DEF_SIG "
    "ARGS".split()
)
NUMBERS = frozenset("INT SIGNED OFFSET CTRLMASK STATE_SIZE".split())
NOTES = frozenset(("NOTE", "NOTE_ATOM"))
TEXTS = frozenset(("TEXT", "SQUOTED", "DQUOTED"))
CONTEXTS = {"intlist": "table", "code": "code", "asm": "code", "expr": "expr", "text": "text"}
LONG = 3
WHOLE = ("table", "code")
ARGUED = frozenset(("CALL", "ARGS"))
_ARG = re.compile(r"(?<=[(,])[+-]?\d+")
_INDENT = re.compile(r"\n[ \t]*")
# lexemes: newline+indentation, note, key or name after its space with its ":", number,
# punctuation run, spaces
LEXEME = (
    r"\n[ \t]*|[A-Ga-g]-?\d+\b| ?[A-Za-z_][A-Za-z0-9_]*(?:\.[a-z]+)*:?|[+-]?\d+"
    r"|[^\sA-Za-z0-9]+ ?| +|\S"
)
REFERENCES = {"cl100k_base": None, "o200k_base": None, "qwen2.5": "Qwen/Qwen2.5-0.5B"}
HELD_OUT = 10


@functools.cache
def _parser() -> Lark:
    return Lark(
        GRAMMAR.read_text(encoding="utf-8"),
        parser="lalr",
        lexer="contextual",
        keep_all_tokens=True,
    )


def lex(text: str) -> tuple:
    """``(tree, lexemes)``: canonical ``text``'s parse keeping every terminal, and each terminal
    as ``(token, context)`` in text order, its context its innermost table, code, expression or
    meta text rule (``""`` outside them)."""
    tree = _parser().parse(text)
    out, stack = [], [(tree, "")]
    while stack:
        node, ctx = stack.pop()
        if isinstance(node, Token):
            out.append((node, ctx))
            continue
        ctx = CONTEXTS.get(node.data, ctx) if ctx not in ("table", "code") else ctx
        stack.extend((child, ctx) for child in reversed(node.children))
    return tree, out


def events(tree: Tree) -> int:
    """The events a document's tree writes."""
    return sum(1 for _ in tree.find_data("event"))


def _category(token: Token, ctx: str) -> int | None:
    """The category of a terminal's non-space characters, or None for its characters' own."""
    if ctx in ("table", "code", "text"):
        return CAT[ctx]
    if token.type in NUMBERS:
        return CAT["long number" if sum(c.isdigit() for c in token) >= LONG else "number"]
    if token.type in NOTES:
        return CAT["note"]
    if token.type in TEXTS:
        return CAT["text"]
    if token.type in NAMES:
        return CAT["name"]
    return CAT["expr"] if ctx == "expr" else None


_WORD = np.array([chr(c).isalnum() or c == 95 for c in range(128)])
_SPACE = np.array([chr(c) in " \n\t" for c in range(128)])


def categories(text: str, lexemes: list | None = None) -> np.ndarray:
    """Each character's index into :data:`CATEGORIES` (uint8): its terminal's category (a table
    or code block whole), else key (a letter), punct, space or indent (a newline and the spaces
    or tab after it) by the character; any non-ASCII character is unicode."""
    if lexemes is None:
        lexemes = lex(text)[1]
    code = np.frombuffer(text.encode("utf-32-le"), np.uint32)
    ascii_ = code < 128
    low = np.where(ascii_, code, 0)
    space, word = _SPACE[low] & ascii_, _WORD[low] & ascii_
    cats = np.where(space, CAT["space"], np.where(word, CAT["key"], CAT["punct"])).astype(np.uint8)
    for m in _INDENT.finditer(text):
        cats[m.start() : m.end()] = CAT["indent"]
    for token, ctx in lexemes:
        cat = _category(token, ctx)
        if cat is None:
            continue
        span = slice(token.start_pos, token.end_pos)
        if ctx in WHOLE:
            cats[span] = cat
            continue
        keep = ~word[span] if cat == CAT["name"] else space[span]
        cats[span] = np.where(keep, cats[span], cat)
        if token.type in ARGUED:
            for m in _ARG.finditer(token):
                n = sum(c.isdigit() for c in m.group())
                cats[token.start_pos + m.start() : token.start_pos + m.end()] = CAT[
                    "long number" if n >= LONG else "number"
                ]
    cats[~ascii_] = CAT["unicode"]
    return cats


def attribute(cats: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Each token's category: the one most characters of its ``[start, end)`` have (a token
    within one character, that character's)."""
    ends = np.maximum(ends, np.minimum(starts + 1, len(cats)))
    cum = np.zeros((len(cats) + 1, len(CATEGORIES)), np.int32)
    np.cumsum(np.eye(len(CATEGORIES), dtype=np.int32)[cats], axis=0, out=cum[1:])
    return np.argmax(cum[ends] - cum[starts], axis=1)


class Encoder:
    """A tokenizer: a tiktoken encoding or a Hugging Face ``tokenizer.json``."""

    def __init__(self, name: str, tiktoken_encoding=None, hf=None):
        self.name, self._tt, self._hf = name, tiktoken_encoding, hf

    def count(self, text: str) -> int:
        """``text``'s token count."""
        if self._tt is not None:
            return len(self._tt.encode(text, disallowed_special=()))
        return len(self._hf.encode(text, add_special_tokens=False).ids)

    def spans(self, text: str) -> tuple:
        """``(starts, ends)``: each token's character span of ``text``."""
        if self._tt is not None:
            ids = self._tt.encode(text, disallowed_special=())
            starts = np.asarray(self._tt.decode_with_offsets(ids)[1], np.int64)
            return starts, np.append(starts[1:], len(text))
        offsets = np.asarray(self._hf.encode(text, add_special_tokens=False).offsets, np.int64)
        return offsets.reshape(-1, 2).T


@functools.cache
def encoder(name: str) -> Encoder:
    """The encoder ``name``: a tiktoken encoding, a :data:`REFERENCES` Hugging Face repository
    (fetched into ``$HF_HOME``), or a directory or file holding a ``tokenizer.json``."""
    if name in REFERENCES and REFERENCES[name] is None:
        import tiktoken  # pylint: disable=import-outside-toplevel

        return Encoder(name, tiktoken_encoding=tiktoken.get_encoding(name))
    from tokenizers import Tokenizer  # pylint: disable=import-outside-toplevel

    if name in REFERENCES:
        from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

        path = Path(hf_hub_download(REFERENCES[name], "tokenizer.json"))
    else:
        path = Path(name)
        path = path / "tokenizer.json" if path.is_dir() else path
    return Encoder(name, hf=Tokenizer.from_file(str(path)))


def held_out(path: Path) -> bool:
    """Whether a document is held out: by its file stem's hash, so a tune's forms share a split."""
    return int(hashlib.sha1(path.stem.encode()).hexdigest()[:8], 16) % HELD_OUT == 0


def documents(*corpus: Path) -> tuple:
    """``(train, held)``: the ``.yaml`` documents under the ``corpus`` directories, sorted, by
    split."""
    paths = sorted(p for c in corpus for p in c.rglob("*.yaml"))
    return [p for p in paths if not held_out(p)], [p for p in paths if held_out(p)]


def bpe():
    """An untrained byte-level BPE tokenizer whose pre-tokenizer splits :data:`LEXEME`s."""
    # pylint: disable=import-outside-toplevel
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(LEXEME), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def train(paths: list, vocab: int, out: Path, progress: bool = True):
    """A BPE of ``vocab`` tokens trained on the documents ``paths``, saved under ``out``."""
    from tokenizers import pre_tokenizers, trainers  # pylint: disable=import-outside-toplevel

    tok = bpe()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    texts = (p.read_text(encoding="utf-8") for p in tqdm(paths, unit="doc", disable=not progress))
    tok.train_from_iterator(texts, trainer, length=len(paths))
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    return tok


def evaluate(names: list, paths: list, progress: bool = True) -> dict:
    """Per encoder of ``names``: the mean and median tokens per document of ``paths`` and
    their characters per token."""
    texts = [p.read_text(encoding="utf-8") for p in paths]
    chars = sum(map(len, texts))
    out = {}
    for name in names:
        enc = encoder(name)
        counts = np.array([enc.count(t) for t in tqdm(texts, desc=name, disable=not progress)])
        out[name] = {
            "docs": len(texts),
            "mean": float(counts.mean()) if len(counts) else 0.0,
            "median": float(np.median(counts)) if len(counts) else 0.0,
            "chars_per_token": chars / max(1, int(counts.sum())),
        }
    return out


def table(results: dict) -> str:
    """A markdown table of :func:`evaluate`'s results."""
    lines = [
        "| tokenizer | docs | tokens/doc mean | median | chars/token |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['docs']} | {r['mean']:.0f} | {r['median']:.0f} "
            f"| {r['chars_per_token']:.2f} |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Train or evaluate a document tokenizer."""
    ap = argparse.ArgumentParser(
        prog="python -m unitracker.optimiser.tokenizer", description=__doc__
    )
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="train a BPE on the corpus's training split")
    p.add_argument("corpus", type=Path, nargs="+")
    p.add_argument("--vocab", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("eval", help="tokens per held-out document against the references")
    p.add_argument("tokenizer", type=Path)
    p.add_argument("corpus", type=Path, nargs="+")
    p.add_argument("--references", nargs="*", default=list(REFERENCES))
    args = ap.parse_args(argv)
    train_paths, held = documents(*args.corpus)
    if args.command == "train":
        train(train_paths, args.vocab, args.out)
        print(f"{args.out / 'tokenizer.json'}: {len(train_paths)} documents, vocab {args.vocab}")
        return 0
    results = evaluate([str(args.tokenizer), *args.references], held)
    (args.tokenizer / "eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
