"""Grammar-constrained decoding of canonical documents: llguidance token masks over a tokenizer's
native ids admitting the continuations the document grammar allows, EOS where it may end.

``unitracker-constrain check|sample|bench|hazards TOKENIZER [DOCS...] [--eos ID]`` (``-h``).
"""

import argparse
import functools
import json
import re
import sys
import time
from pathlib import Path

import llguidance
import numpy as np

from tqdm import tqdm
from unitracker_grammar.canonical import GRAMMAR, parse, parser
from unitracker_grammar.document import read

from unitracker_learn import lm

_LARK = re.compile(
    r'"(?:[^"\\]|\\.)*"|//[^\n]*|/((?:[^/\\\n]|\\.)+)/|^(\w+)\.\d+:|\s*->\s*\w+$', re.M
)
_TAIL = re.compile(r"(?:\(\?[!=](?:[^()\\]|\\.|\((?:[^()\\]|\\.)*\))*\)|\\b)+$")
_LOOK = re.compile(r"\(\?<?[!=]|\\[bB]")
SAFE = {("NEG", "SIGNED")}


def _close(pat: str, start: int) -> int:
    """The index just past the group or class opening at ``pat[start]``."""
    depth, i, cls = 0, start, -1
    while i < len(pat):
        ch = pat[i]
        if ch == "\\":
            i += 1
        elif cls >= 0:
            if ch == "]" and i > cls + 1 + (pat[cls + 1] == "^"):
                depth, cls = depth - 1, -1
        elif ch == "[":
            depth, cls = depth + 1, i
        elif ch in "()":
            depth += 1 if ch == "(" else -1
        i += 1
        if not depth:
            return i
    raise ValueError(f"/{pat}/: unbalanced at {start}")


def _alternatives(pat: str) -> list:
    """``pat`` split at its top-level ``|``."""
    out, last, i = [], 0, 0
    while i < len(pat):
        if pat[i] in "([":
            i = _close(pat, i)
            continue
        if pat[i] == "|":
            out.append(pat[last:i])
            last = i + 1
        i += 2 if pat[i] == "\\" else 1
    return out + [pat[last:]]


def regex(body: str) -> str:
    """A Lark regex's llguidance form: a leading ``(?!X)`` as ``& ~`` X's alternatives (each
    whole where it ends ``\\b``, else a prefix), trailing lookarounds and ``\\b`` dropped (the
    lexer's longest match and the grammar end a lexeme there); ValueError for any other."""
    exclude, end = [], _close(body, 0) if body.startswith("(?!") else 0
    if 0 < end < len(body):
        exclude = [
            a[:-2] if a.endswith(r"\b") else a + "(?s:.*)" for a in _alternatives(body[3 : end - 1])
        ]
        body = body[end:]
    body = _TAIL.sub("", body)
    if _LOOK.search(body):
        raise ValueError(f"/{body}/: a lookaround llguidance cannot express")
    out = f"/{_classes(body)}/" if body else ""
    return f"({out} & ~/{_classes('|'.join(exclude))}/)" if exclude else out


@functools.cache
def _ranges(escape: str) -> str:
    """The characters Python's ``re`` matches by ``escape`` (``\\d``, ``\\s``, ``\\w``) as
    Rust class ranges: the two engines' Unicode tables differ."""
    test = re.compile(escape).match
    hits = np.flatnonzero([bool(test(chr(c))) for c in range(sys.maxunicode + 1)])
    cuts = np.flatnonzero(np.diff(hits) > 1)
    lows, highs = hits[np.r_[0, cuts + 1]], hits[np.r_[cuts, len(hits) - 1]]
    return "".join(f"\\x{{{lo:x}}}-\\x{{{hi:x}}}" for lo, hi in zip(lows, highs))


def _classes(pat: str) -> str:
    """``pat`` with its ``\\d``, ``\\s`` and ``\\w`` spelled out as Python reads them."""
    out, i, cls = [], 0, False
    while i < len(pat):
        ch = pat[i : i + 2] if pat[i] == "\\" else pat[i]
        if ch in (r"\d", r"\s", r"\w"):
            ch = _ranges(ch) if cls else f"[{_ranges(ch)}]"
        elif ch in "[]":
            cls = ch == "[" or (cls and pat[i - 1] in "[^")
        out.append(ch)
        i += len(pat[i : i + 2]) if pat[i] == "\\" else 1
    return "".join(out)


def lark(text: str) -> str:
    """Lark grammar ``text`` as llguidance reads it: comments, terminal priorities and aliases
    dropped, its regexes through :func:`regex`."""

    def sub(m):
        if m[0].startswith('"'):
            return m[0]
        if m[1] is not None:
            return regex(m[1])
        return f"{m[2]}:" if m[2] else ""

    return _LARK.sub(sub, text)


@functools.cache
def grammar(forcing: bool = False) -> str:
    """The canonical document grammar compiled for llguidance: any tokenization of bytes the
    grammar forces allowed, or with ``forcing`` only their canonical one."""
    text = lark(GRAMMAR.read_text(encoding="utf-8"))
    if not forcing:
        text = '%llguidance {"no_forcing": true}\n' + text
    return llguidance.LLMatcher.grammar_from_lark(text)


def tokenizer(tok):
    """A ``tokenizers.Tokenizer``: ``tok`` itself, else the tokenizer.json path, its directory
    or the reference name that :func:`unitracker_learn.tokenizer.encoder` resolves."""
    if hasattr(tok, "to_str"):
        return tok
    from unitracker_learn import tokenizer as tk  # pylint: disable=import-outside-toplevel

    return tk.encoder(str(tok))._hf  # pylint: disable=protected-access


class Matcher:
    """A document's grammar state (or ``compiled``'s): :meth:`feed` its tokens, then its next
    token's mask."""

    def __init__(self, constraint: "Constraint", compiled: str | None = None):
        self.constraint = constraint
        compiled = compiled or grammar(constraint.forcing)
        self.m = llguidance.LLMatcher(constraint.lltok, compiled)

    def valid(self, tokens) -> int:
        """How many of ``tokens`` (native ids, EOS never) the grammar takes from here, none
        consumed."""
        return self.m.validate_tokens([int(t) for t in tokens])

    def feed(self, tokens) -> bool:
        """Consume ``tokens``, an EOS only last (more after the end are ignored): False where the
        grammar refuses one, none consumed unless it is the EOS."""
        tokens, eos = [int(t) for t in tokens], self.constraint.eos
        if self.done:
            return set(tokens) <= {eos}
        body = tokens[: -1 if tokens and tokens[-1] == eos else None]
        if self.valid(body) < len(body) or body and not self.m.consume_tokens(body):
            return False
        if len(body) < len(tokens):
            if not self.accepting:
                return False
            self.m.consume_token(eos)
        return True

    @property
    def accepting(self) -> bool:
        """Whether the document may end here."""
        return self.m.is_accepting()

    @property
    def done(self) -> bool:
        """Whether the document has ended: its EOS consumed, or the only token left."""
        return self.m.is_stopped()

    def mask(self) -> np.ndarray:
        """The next token's mask, a bool per native id."""
        return self.constraint.masks([self])[0]


class Constraint:
    """The document grammar over tokenizer ``tok``'s native ids, ``eos`` the end of document (an
    added special token where the tokenizer has no such id); masks are at least ``n_vocab``
    wide."""

    def __init__(self, tok, eos: int | None = None, n_vocab: int = 0, forcing: bool = False):
        tok = tokenizer(tok)
        size = tok.get_vocab_size(with_added_tokens=True)
        eos = size if eos is None else eos
        if eos >= size:
            tok = type(tok).from_str(tok.to_str())
            tok.add_special_tokens([f"<|eod{i}|>" for i in range(size, eos + 1)])
        self.tok, self.eos, self.forcing = tok, eos, forcing
        self.lltok = llguidance.LLTokenizer(tok.to_str(), eos_token=eos)
        self.n_vocab = max(n_vocab, self.lltok.vocab_size)
        self.executor = llguidance.LLExecutor()

    def encode(self, text: str) -> list:
        """``text``'s native ids."""
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        """The text of native ``ids``."""
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def matcher(self, prompt=()) -> Matcher:
        """A new document's :class:`Matcher` with ``prompt`` (native ids) fed; ValueError where
        the grammar refuses it."""
        m = Matcher(self)
        if not m.feed(prompt):
            raise ValueError(f"the grammar refuses the prompt's token {m.valid(prompt)}")
        return m

    def bitmask(self, matchers: list) -> np.ndarray:
        """Each matcher's next-token mask as llguidance's int32 bit words (bit ``t % 32`` of word
        ``t // 32``), computed in parallel; ``llguidance.torch`` applies it to logits."""
        out = np.zeros((len(matchers), (self.lltok.vocab_size + 31) // 32), np.int32)
        if matchers:
            jobs = [(m.m, i) for i, m in enumerate(matchers)]
            self.executor.unsafe_compute_mask_ptr(jobs, out.ctypes.data, out.shape[1] * 4, len(out))
        return np.pad(out, ((0, 0), (0, (self.n_vocab + 31) // 32 - out.shape[1])))

    def masks(self, matchers: list, tensor: bool = False):
        """Each matcher's next-token mask, ``(len(matchers), n_vocab)`` bools: a numpy array, or
        with ``tensor`` a CPU torch tensor sharing its memory."""
        bits = np.unpackbits(self.bitmask(matchers).view(np.uint8), axis=1, bitorder="little")
        out = bits[:, : self.n_vocab].view(bool)
        if tensor:
            import torch  # pylint: disable=import-outside-toplevel,import-error

            return torch.from_numpy(out)
        return out

    def check(self, docs: list, progress: bool = False) -> list:
        """Per document (a text, or its native ids), the index of the first of its tokens (or
        of its EOS, last) outside the mask, -1 where none is: documents fed in lockstep, every
        step's mask computed."""
        seqs = [(self.encode(d) if isinstance(d, str) else list(d)) + [self.eos] for d in docs]
        ms, out = [self.matcher() for _ in seqs], [-1] * len(seqs)
        shown = tqdm(total=sum(map(len, seqs)), unit="tok", disable=not progress)
        for step in range(max(map(len, seqs), default=0)):
            live = [i for i, s in enumerate(seqs) if step < len(s) and out[i] < 0]
            words = self.bitmask([ms[i] for i in live])
            for row, i in enumerate(live):
                t = seqs[i][step]
                if not (words[row, t >> 5] >> (t & 31)) & 1 or not ms[i].feed([t]):
                    out[i] = step
            shown.update(len(live))
        shown.close()
        return out

    def sample(self, model, matchers: list, limit: int, seed: int = 0) -> list:
        """Each matcher's document continued under its mask, up to ``limit`` tokens or its EOS
        (not among them), sampled (``seed``) from ``model(step, last)``: logits per native id
        given each live document's last token (EOS at first)."""
        rng, out = np.random.default_rng(seed), [[] for _ in matchers]
        last = np.full(len(matchers), self.eos)
        for step in range(limit):
            live = [i for i, m in enumerate(matchers) if not m.done]
            if not live:
                break
            logits = model(step, last[live])
            for i, lg, mask in zip(live, logits, self.masks([matchers[i] for i in live])):
                ok = np.flatnonzero(mask)
                last[i] = ok[np.argmax(lg[ok] + rng.gumbel(size=len(ok)))]
                matchers[i].feed([last[i]])
                out[i] += [int(last[i])] * int(last[i] != self.eos)
        return out


class Masked:  # pylint: disable=too-few-public-methods
    """A ``model.generate`` logits processor that keeps each row to the document grammar: each
    row's matcher fed its new tokens (``cols`` maps model ids to native ids, None the same; a
    window's first call's last token is new unless ``fresh``) and its logits masked; a row whose
    mask leaves no model id is ended, its index in ``stuck``."""

    def __init__(self, constraint, matchers: list, cols, eos: int, fresh: bool):
        self.constraint, self.matchers, self.cols, self.eos = constraint, matchers, cols, eos
        self.pending, self.stuck = not fresh, set()

    def __call__(self, input_ids, scores):
        import torch  # pylint: disable=import-outside-toplevel,import-error

        if self.pending:
            for m, t in zip(self.matchers, input_ids[:, -1].tolist()):
                m.feed([t if self.cols is None else int(self.cols[t])])
        self.pending = True
        mask = self.constraint.masks(self.matchers, tensor=True)
        mask = mask[:, : scores.shape[1]] if self.cols is None else mask[:, self.cols]
        mask = mask.to(scores.device)
        empty = ~mask.any(1)
        self.stuck.update(empty.nonzero().flatten().tolist())
        mask[empty, self.eos] = True
        return scores.masked_fill(~mask, -torch.inf)


def documents(paths: list, n: int = 0, seed: int = 0) -> list:
    """The ``.yaml`` documents of ``paths`` (files, or directories searched), a seeded sample of
    ``n`` where ``n``, their meta lines replaced as the learnability experiment's models read
    them."""
    files = sorted(f for p in map(Path, paths) for f in (p.rglob("*.yaml") if p.is_dir() else [p]))
    if 0 < n < len(files):
        files = [files[i] for i in sorted(np.random.default_rng(seed).choice(len(files), n, False))]
    return [lm.META_LINE.sub(lm.META, f.read_text("utf-8"), count=1) for f in files]


def _kinds(con: Constraint) -> tuple:
    """``(close, end, item)`` per native id: a closing policy's logits (tokens that only close
    lines or brackets, other punctuation opening none, EOS), whether a token is ``end``, and
    whether it ends with an item's ``-``."""
    out = np.zeros((4, con.n_vocab))
    for i in range(con.lltok.vocab_size):
        w = con.decode([i])
        shut = bool(w) and set(w) <= set("\n]})'")
        punct = bool(w) and not shut and not any(c.isalnum() or c in "_([{" for c in w)
        out[:, i] = shut, punct, w.strip() == "end", w.strip().endswith("-")
    out[0, con.eos] = 2
    return 50 * (out[0] + out[1] / 2), out[2], out[3]


def standin(con: Constraint, texts: list, wander: int, ramp: int = 200, seed: int = 0):
    """A stand-in model ``model(step, last)``: Gaussian logits (``seed``) for ``wander`` steps,
    then ``texts``' token bigrams over their unigrams (log add-one, each text between EOSs) blended
    over ``ramp`` steps into a policy closing brackets, lines and the song, so documents end."""
    seqs = [np.array([con.eos] + con.encode(t) + [con.eos], np.int64) for t in texts]
    v, rng, (close, end, item) = con.n_vocab, np.random.default_rng(seed), _kinds(con)
    uni = np.log1p(np.bincount(np.concatenate(seqs), minlength=v))
    keys, counts = np.unique(np.concatenate([s[:-1] * v + s[1:] for s in seqs]), return_counts=True)

    def model(step: int, last: np.ndarray) -> np.ndarray:
        if step < wander:
            return rng.normal(size=(len(last), v))
        out = np.tile(uni, (len(last), 1))
        for row, t in zip(out, last):
            lo, hi = np.searchsorted(keys, [t * v, (t + 1) * v])
            row[keys[lo:hi] - t * v] += 4 * np.log1p(counts[lo:hi])
        mix = min(1.0, (step - wander) / ramp)
        return (1 - mix) * out + mix * (close + 50 * np.outer(item[last], end))

    return model


def judge(doc: str) -> dict:
    """Per check of a document (``syntax``, the grammar's parser; ``parse``,
    ``canonical.parse``; ``read``, ``document.read``), its refusal, or None."""
    out = {}
    for key, fn in (("syntax", parser().parse), ("parse", parse), ("read", read)):
        try:
            fn(doc)
            out[key] = None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            out[key] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"
    return out


def agreement(con: Constraint, texts: list, n: int, limit: int, wander: int) -> tuple:
    """``(tally, refusals)``: ``n`` documents sampled under the mask by :func:`standin` of
    ``texts`` after the first one's header and meta lines; how many ended, and of those how many
    each :func:`judge` check takes."""
    head = texts[0][: texts[0].index("\n", texts[0].index("meta:")) + 1]
    ms = [con.matcher(con.encode(head)) for _ in range(n)]
    ids = con.sample(standin(con, texts, wander), ms, limit)
    verdicts = [judge(head + con.decode(s)) for s, m in zip(ids, ms) if m.done]
    tally = {"samples": n, "ended": len(verdicts)}
    tally |= {k: sum(v[k] is None for v in verdicts) for k in ("syntax", "parse", "read")}
    return tally, [f"{k}: {e}" for v in verdicts for k, e in v.items() if e]


@functools.cache
def _terminals() -> tuple:
    """``(defs, expr)``: the llguidance grammar's terminal definitions, and per Lark terminal its
    expression there (its name, or its literal)."""
    defs = "\n".join(re.findall(r"(?m)^_?[A-Z]\w*:.*$", lark(GRAMMAR.read_text("utf-8"))))
    named = set(re.findall(r"(?m)^(\w+):", defs))
    return defs, {
        t.name: t.name if t.name in named else json.dumps(t.pattern.value)
        for t in parser().terminals
    }


def _follow(con: Constraint, allowed) -> tuple:
    """``(accept, after)``: each LALR state's acceptable terminals, and per terminal the ASCII
    characters that may start a terminal shifted after it."""
    table, expr = parser().parser.parser.parser.parse_table.states, _terminals()[1]
    first = {}
    for n, e in expr.items():
        heads = {con.decode([t])[:1] for t in np.flatnonzero(allowed(e)[: con.eos])}
        first[n] = {h for h in heads if h.isascii()}
    accept = {s: [n for n in acts if n in expr] for s, acts in table.items()}
    after = {n: set() for n in expr}
    for s, acts in table.items():
        for n, (act, nxt) in acts.items():
            if n in expr and act.name == "Shift":
                after[n].update(c for t in accept[nxt] for c in first[t])
    return accept, after


def hazards(con: Constraint, progress: bool = False) -> list:
    """Terminal pairs ``(a, b)`` where llguidance's greedy lexer may read text otherwise than
    Lark's, which tries terminals in turn: both acceptable in a state of Lark's LALR table, and
    ``b`` extending an ``a`` match by an ASCII character that may start a terminal after ``a``
    (:data:`SAFE`'s: ``-`` and an integer are the same factor whichever way it is read)."""
    defs, expr = _terminals()

    def allowed(rx: str) -> np.ndarray:
        g = llguidance.LLMatcher.grammar_from_lark(f"start: HZ\n{defs}\nHZ: {rx}")
        return Matcher(con, g).mask()

    accept, after = _follow(con, allowed)
    pairs = sorted({(a, b) for ns in accept.values() for a in ns for b in ns if a != b})
    out = []
    for a, b in tqdm(pairs, unit="pair", disable=not progress):
        nxt = " | ".join(map(json.dumps, sorted(after[a])))
        if nxt and allowed(f"({expr[b]}) & (({expr[a]}) ({nxt}) /(?s:.*)/)")[: con.eos].any():
            out.append((a, b))
    return out


def bench(con: Constraint, text: str, batch: int) -> tuple:
    """``(mask, feed)``: the mean seconds per step of ``batch`` documents feeding ``text`` in
    lockstep to compute their masks, and to consume a token."""
    ms, spent = [con.matcher() for _ in range(batch)], np.zeros(2)
    ids = con.encode(text)
    for t in tqdm(ids, unit="tok"):
        start = time.perf_counter()
        con.bitmask(ms)
        mid = time.perf_counter()
        for m in ms:
            m.feed([t])
        spent += mid - start, time.perf_counter() - mid
    return tuple(spent / len(ids))


def main(argv=None) -> int:
    """``check``: whether every document's tokens and EOS are in their masks; ``sample``:
    documents sampled under the mask, parsed; ``bench``: mask time per step of a batch;
    ``hazards``: :func:`hazards` beyond :data:`SAFE`."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("mode", choices=("check", "sample", "bench", "hazards"))
    ap.add_argument("tokenizer", help="tokenizer.json, its directory, or qwen2.5")
    ap.add_argument("docs", nargs="*", help=".yaml documents or directories of them")
    ap.add_argument("--eos", type=int, help="end of document id (default: one appended)")
    ap.add_argument("-n", type=int, default=0, help="documents: a seeded sample (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16, help="documents at once")
    ap.add_argument("--limit", type=int, default=4096, help="tokens per sample")
    ap.add_argument("--wander", type=int, default=30, help="random tokens before the bigrams")
    args = ap.parse_intermixed_args(argv)
    con, texts = Constraint(args.tokenizer, args.eos), documents(args.docs, args.n, args.seed)
    if args.mode == "hazards":
        found = set(hazards(con, progress=True))
        print(sorted(found))
        return int(bool(found - SAFE))
    if args.mode == "sample":
        tally, refusals = agreement(con, texts, args.batch, args.limit, args.wander)
        print("\n".join(refusals + [str(tally)]))
        return int(tally["syntax"] < tally["ended"])
    if args.mode == "bench":
        mask, feed = bench(con, texts[0][: args.limit * 8], args.batch)
        print(f"per step of {args.batch}: mask {mask * 1e3:.3f} ms, feed {feed * 1e3:.3f} ms")
        return 0
    bad = []
    for k in tqdm(range(0, len(texts), args.batch), unit="batch"):
        bad += [(k + i, at) for i, at in enumerate(con.check(texts[k : k + args.batch])) if at >= 0]
    for i, at in bad:
        print(f"doc {i}: token {at} refused after {con.decode(con.encode(texts[i])[:at])[-60:]!r}")
    print(f"{len(texts) - len(bad)}/{len(texts)} documents accepted")
    return int(bool(bad))


if __name__ == "__main__":
    sys.exit(main())
