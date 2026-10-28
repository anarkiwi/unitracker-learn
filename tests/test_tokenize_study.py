"""tools/tokens.py and ``tools/corpus.py forms``: the corpus's document forms, the printer
transforms, and the study's tables on small documents."""

import csv
import json

import pytest

import corpus
import optimiser_docs as D
import shared
import tokens
from test_tokenize import DOC
from unitracker.optimiser import tokenizer as tk
from unitracker.optimiser.fit import TOLERANCES


def _apply(name, text=DOC):
    tree, lexemes = tk.lex(text)
    return tokens.applied(text, tokens.TRANSFORMS[name](text, tree, lexemes))


def test_numbers_are_written_in_hexadecimal():
    text = _apply("hex")
    assert "sr:$F0," in text and " freq:$4D2" in text and "up($12C)" in text
    assert "[0,$84,32,$FF]" in text and "ad:9," in text


def test_tables_are_delta_coded():
    assert "PL:[0,+132,-100,+223]" in _apply("delta-tables")


def test_keys_shorten_to_unique_prefixes():
    text = _apply("short-keys")
    assert "{a:9,sr:240,c:[{c:65,n:+0},{w:1,n:g5}" in text and "D5 f:1234 legato]" in text
    assert "\nme:{title:T,author:Ö}\nv:1\n" in text and "\n - pl:[p1]\n" in text


def test_overlapping_edits_keep_the_first():
    assert tokens.applied("abcdef", [(3, 5, "X"), (1, 4, "Y"), (5, 5, "Z")]) == "aYeZf"


@pytest.fixture(name="forms_root")
def _forms_root(tmp_path):
    root = tmp_path / "forms"
    docs = {"a": DOC, "b": D.TEMPLATE, "c": D.MERGE}
    for form in ("lifted", "named"):
        (root / form).mkdir(parents=True)
        for doc, text in docs.items():
            (root / form / f"{doc}.yaml").write_text(text, encoding="utf-8")
    (root / "named" / "c.yaml").unlink()
    (root / "meta").mkdir()
    for doc in docs:
        (root / "meta" / f"{doc}.json").write_text(json.dumps({"ticks": 100}))
    (root / "lifted" / "d.yaml").write_text("not a document\n")
    return root


def test_the_study_measures_forms_categories_and_transforms(forms_root, capsys):
    assert tokens.main([str(forms_root), "--forms", "lifted", "named"]) == 0
    printed = capsys.readouterr().out
    assert "2 tunes with every form of lifted, named" in printed
    assert "d: " in (forms_root / "study.md").read_text().split("## failed")[1]
    results = {r["doc"]: r for r in json.loads((forms_root / "study.json").read_text())}
    assert set(results) == {"a", "b", "c"} and set(results["c"]["forms"]) == {"lifted"}
    one = results["a"]["forms"]["lifted"]
    assert one["chars"] == len(DOC) and one["events"] == 8 and sum(one["cat_chars"]) == len(DOC)
    for name in tk.REFERENCES:
        assert sum(one[name]["cats"]) == one[name]["tokens"]
        assert one[name]["transforms"]["short-keys"] < one[name]["tokens"]
    row = next(l for l in printed.splitlines() if l.startswith("| lifted | cl100k_base |"))
    mean = (
        one["cl100k_base"]["tokens"] + results["b"]["forms"]["lifted"]["cl100k_base"]["tokens"]
    ) / 2
    assert f"| {mean:.0f} |" in row
    assert "| note |" in printed and "| short-keys |" in printed
    top = next(l for l in printed.splitlines() if l.startswith("| note | `"))
    assert "`,-` " in top and "`:g` " in top and "`\\n` " in printed


def test_forms_are_the_optimisers_and_lossy_when_it_has_it(monkeypatch):
    table = corpus.forms()
    base = ["lifted", "named", "single", "all", "lossy", "lossy-single", "lossy-all"]
    assert list(corpus.forms(ablations=False)) == base
    assert list(table) == base + [
        "lossy-nophrases",
        "lossy-nophrases-single",
        "lossy-nophrases-all",
    ]
    assert [p for p, _ in table["lossy-nophrases"][1]["order"]] == [
        p for p, _ in corpus.PASSES if p != "phrases"
    ]
    assert table["lossy"] == ("lifted", {"lossy": dict(TOLERANCES)})
    assert table["lossy-all"] == ("lossy", {"inline": "all"})
    assert corpus.with_bases(["lossy-all", "single"]) == [
        "lifted",
        "named",
        "single",
        "lossy",
        "lossy-all",
    ]

    def exact(text, ticks=None, *, inline="named"):  # pylint: disable=unused-argument
        return text

    monkeypatch.setattr(corpus, "optimise", exact)
    assert list(corpus.forms()) == ["lifted", "named", "single", "all"]


def test_a_songs_length_is_its_end_or_its_first_loop():
    assert corpus.song_ticks(D.DEFAULTS, 1000) == len(corpus.play(D.DEFAULTS))
    looping = corpus.song_ticks(D.LOOPING, 1000)
    assert 0 < looping < 1000 and corpus.song_ticks(D.LOOPING, 3) == 3


def _opts(out, **kw):
    return {
        "out": str(out),
        "hvsc": None,
        "timeout": 0,
        "subtune": 0,
        "ticks": 200,
        "limit": 1000,
        "retry": False,
        "forms": ["lifted", "named", "single"],
    } | kw


def test_a_tunes_forms_are_optimised_from_its_cached_lifted_document(tmp_path):
    (tmp_path / "lifted").mkdir()
    (tmp_path / "lifted" / "t.yaml").write_text(D.HOIST, encoding="utf-8")
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "t.json").write_text(json.dumps({"player": "p", "ticks": 7}))
    res = corpus.forms_one({"path": "x/t.sid"}, _opts(tmp_path))
    assert res == {
        "doc": "t",
        "lifted": "ok",
        "named": "ok",
        "single": "ok",
        "player": "p",
        "ticks": 7,
    }
    named = (tmp_path / "named" / "t.yaml").read_text()
    assert len(named) < len(D.HOIST) and not list(tmp_path.rglob("*.err"))


def test_a_lossy_form_is_the_lifted_document_optimised_lossily(tmp_path):
    (tmp_path / "lifted").mkdir()
    (tmp_path / "lifted" / "t.yaml").write_text(D.EXACT, encoding="utf-8")
    forms = corpus.with_bases(["lossy-all"])
    res = corpus.forms_one({"path": "x/t.sid"}, _opts(tmp_path, forms=forms))
    assert res["lossy"] == "ok" and res["lossy-all"] == "ok" and "named" not in res
    lossy = (tmp_path / "lossy" / "t.yaml").read_text()
    assert ":|\n" in D.EXACT and ":|\n" not in lossy
    inlined = (tmp_path / "lossy-all" / "t.yaml").read_text()
    assert "\nins:" in lossy and "\nins:" not in inlined


def test_a_tune_that_does_not_lift_has_no_forms_and_stays_failed(tmp_path):
    sid = tmp_path / "junk.sid"
    sid.write_bytes(b"not a sid")
    out = tmp_path / "out"
    res = corpus.forms_one({"path": str(sid)}, _opts(out))
    assert res["lifted"].startswith("FormatError") and res["named"] == "no lifted form"
    (out / "lifted" / "junk.err").write_text("Earlier: failure\n")
    assert corpus.forms_one({"path": str(sid)}, _opts(out))["lifted"] == "Earlier: failure"
    assert corpus.forms_one({"path": str(sid)}, _opts(out, retry=True))["lifted"].startswith(
        "FormatError"
    )


def test_the_forms_command_lifts_a_directory_and_tabulates(metal_warrior, tmp_path, capsys):
    tunes = tmp_path / "tunes"
    (tunes / "x").mkdir(parents=True)
    (tunes / "x" / "junk.sid").write_bytes(b"not a sid")
    (tunes / metal_warrior.name).write_bytes(metal_warrior.read_bytes())
    rows = shared.rows([tunes])
    assert [r["path"] for r in rows] == [str(tunes / metal_warrior.name), str(tunes / "x/junk.sid")]
    out = tmp_path / "out"
    argv = ["forms", str(tunes), "--out", str(out), "--jobs", "1", "--forms", "lifted"]
    argv.append("--unchecked")
    assert corpus.main([*argv, "--limit", "500"]) == 0
    assert "lifted: 1/2 {'goattracker2': 1}" in capsys.readouterr().out
    table = {r["doc"].split("-", 1)[1]: r for r in csv.DictReader((out / "forms.csv").open())}
    assert table[metal_warrior.stem]["ticks"] == "500" and table["junk"]["lifted"].startswith(
        "FormatError"
    )
