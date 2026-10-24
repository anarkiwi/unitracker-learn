"""unitracker.optimiser.tokenizer: grammar categories, token attribution, encoders, and a BPE
trained, split and evaluated on small documents."""

import re

import numpy as np
import pytest

import optimiser_docs as D

from unitracker.optimiser import tokenizer as tk

DOC = (
    "unitracker: 2.0\n"
    "meta: {title: T, author: A}\n"
    "voices: 1\n"
    "tables:\n"
    "  PL: [0, 132, 32, 255]\n"
    "mod:\n"
    "  vib(d): v + d*(φ % 4)\n"
    "  up(c): |\n"
    "    lda PL.lo,x\n"
    "    adc #c\n"
    "    sta v+1\n"
    "    rts\n"
    "ins:\n"
    "  i1: {ad: 9, sr: 240, ctrl: [{ctrl: 65, note: +0}, {wait: 1, note: G#5}, end],"
    " pw: up(300), freq: vib(2)}\n"
    "song:\n"
    "  - def:\n"
    "      phrase:\n"
    "        p1: [C4.i1, -, A#2, -*3, C4, off, C#4, D5!freq:1234]\n"
    "    play: [p1]\n"
    "  - end\n"
)


def _cat(text, cats, sub, nth=0):
    at = [m.start() for m in re.finditer(re.escape(sub), text)][nth]
    return {tk.CATEGORIES[c] for c in cats[at : at + len(sub)]}


def test_lexemes_cover_the_text_in_order_with_their_contexts():
    tree, lexemes = tk.lex(DOC)
    assert "".join(t for t, _ in lexemes) == DOC
    assert tk.events(tree) == 8
    ctx = {t.start_pos: c for t, c in lexemes}
    assert ctx[DOC.index("lda")] == "code" and ctx[DOC.index("132")] == "table"
    assert ctx[DOC.index("φ")] == "expr" and ctx[DOC.index("T,")] == "text"
    assert ctx[DOC.index("C4.i1")] == ""


@pytest.mark.parametrize(
    "sub,nth,cat",
    [
        ("C4", 0, "note"),
        ("A#2", 0, "note"),
        ("G#5", 0, "note"),
        ("off", 0, "note"),
        ("sr", 0, "key"),
        ("wait", 0, "key"),
        ("i1", 1, "name"),
        ("p1", 1, "name"),
        ("9", 0, "number"),
        ("240", 0, "long number"),
        ("1234", 0, "long number"),
        ("300", 0, "long number"),
        ("[0, 132, 32, 255]", 0, "table"),
        ("    lda PL.lo,x\n", 0, "code"),
        ("φ", 0, "unicode"),
        ("%", 0, "expr"),
        ("T", 0, "text"),
        ("\n      ", 0, "indent"),
        ("{", 1, "punct"),
    ],
)
def test_characters_take_their_grammar_category(sub, nth, cat):
    cats = tk.categories(DOC)
    assert cats.dtype == np.uint8 and len(cats) == len(DOC)
    assert _cat(DOC, cats, sub, nth) == {cat}


def test_a_call_names_its_template_and_numbers_its_arguments():
    cats = tk.categories(DOC)
    at = DOC.index("vib(2)")
    assert [tk.CATEGORIES[c] for c in cats[at : at + 6]] == ["name"] * 3 + ["punct", "number"] + [
        "punct"
    ]
    assert _cat(DOC, cats, ": ", 0) == {"punct", "space"}


def test_a_token_takes_the_category_most_of_its_characters_have():
    cats = np.array([tk.CAT["note"]] * 2 + [tk.CAT["punct"]] + [tk.CAT["space"]] * 3, np.uint8)
    starts, ends = np.array([0, 1, 3, 5]), np.array([3, 3, 6, 5])
    got = [tk.CATEGORIES[c] for c in tk.attribute(cats, starts, ends)]
    assert got == ["note", "note", "space", "space"]


def test_tiktoken_spans_tile_the_text():
    enc = tk.encoder("cl100k_base")
    starts, ends = enc.spans(DOC)
    assert starts[0] == 0 and ends[-1] == len(DOC) and np.all(starts[1:] == ends[:-1])
    assert len(starts) == enc.count(DOC)


def _corpus(tmp_path):
    docs = {
        "defaults": D.DEFAULTS,
        "merge": D.MERGE,
        "hoist": D.HOIST,
        "template": D.TEMPLATE,
        "looping": D.LOOPING_CODE,
        "doc": DOC,
    }
    for form in ("lifted", "named"):
        (tmp_path / form).mkdir(parents=True)
        for k, text in docs.items():
            (tmp_path / form / f"{k}.yaml").write_text(text, encoding="utf-8")
    return docs


def test_the_split_is_by_stem_so_a_tunes_forms_share_it(tmp_path):
    docs = _corpus(tmp_path)
    train, held = tk.documents(tmp_path)
    assert len(train) + len(held) == 2 * len(docs)
    for path in held:
        assert tk.held_out(path) and tk.held_out(tmp_path / "x" / path.name)
    assert {p.stem for p in train}.isdisjoint({p.stem for p in held})
    tk.HELD_OUT, old = 2, tk.HELD_OUT
    try:
        assert len(tk.documents(tmp_path)[1]) not in (0, 2 * len(docs))
    finally:
        tk.HELD_OUT = old


def test_a_trained_bpe_keeps_lexemes_whole_and_round_trips(tmp_path):
    _corpus(tmp_path)
    paths = sorted(tmp_path.rglob("*.yaml"))
    tok = tk.train(paths, 300, tmp_path / "bpe", progress=False)
    assert tok.get_vocab_size() <= 300 and (tmp_path / "bpe" / "tokenizer.json").is_file()
    enc = tk.encoder(str(tmp_path / "bpe"))
    bounds = {m.end() for m in re.finditer(tk.LEXEME, DOC)} | {0}
    starts, ends = enc.spans(DOC)
    assert enc.count(DOC) == len(starts) < len(DOC)
    for start, end in zip(starts, ends):
        inside = [b for b in bounds if start < b < end]
        assert not inside, DOC[start:end]
    encoded = enc._hf.encode(DOC)  # pylint: disable=protected-access
    assert enc._hf.decode(encoded.ids) == DOC  # pylint: disable=protected-access


def test_evaluation_compares_held_out_tokens_per_document(tmp_path):
    _corpus(tmp_path)
    paths = sorted(tmp_path.rglob("*.yaml"))
    tk.train(paths, 400, tmp_path / "bpe", progress=False)
    got = tk.evaluate([str(tmp_path / "bpe"), "cl100k_base"], paths[:3], progress=False)
    for r in got.values():
        assert r["docs"] == 3 and r["median"] > 0 and r["chars_per_token"] > 1
    assert tk.evaluate(["cl100k_base"], [], progress=False)["cl100k_base"]["mean"] == 0
    assert "| cl100k_base | 3 |" in tk.table(got)


def test_the_command_line_trains_and_evaluates(tmp_path, capsys):
    _corpus(tmp_path / "c")
    out = tmp_path / "bpe"
    assert tk.main(["train", str(tmp_path / "c"), "--vocab", "300", "--out", str(out)]) == 0
    assert "vocab 300" in capsys.readouterr().out
    assert tk.main(["eval", str(out), str(tmp_path / "c"), "--references", "cl100k_base"]) == 0
    printed = capsys.readouterr().out
    assert "| cl100k_base |" in printed and (out / "eval.json").is_file()


def test_a_reference_hf_tokenizer_is_fetched_by_repository(tmp_path, monkeypatch):
    import huggingface_hub  # pylint: disable=import-outside-toplevel

    _corpus(tmp_path)
    tk.train(sorted(tmp_path.rglob("*.yaml")), 300, tmp_path / "bpe", progress=False)
    asked = []

    def fetch(repo, filename):
        asked.append((repo, filename))
        return str(tmp_path / "bpe" / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    monkeypatch.setitem(tk.REFERENCES, "fake", "org/fake")
    enc = tk.encoder("fake")
    assert asked == [("org/fake", "tokenizer.json")] and enc.count(DOC) > 0
    tk.encoder.cache_clear()
