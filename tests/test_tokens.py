"""unitracker_learn.tokens: the printer transforms and the study's tables on small documents."""

import json

import pytest

import optimiser_docs as D
from test_tokenizer import DOC
from unitracker_learn import tokenizer as tk
from unitracker_learn import tokens


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


def test_a_tunes_forms_are_measured_in_place_of_the_worker_pool(forms_root):
    paths = {f: forms_root / f / "a.yaml" for f in ("lifted", "named")}
    res = tokens.study_one("a", paths, ["cl100k_base"])
    assert res["doc"] == "a" and set(res["forms"]) == {"lifted", "named"}
    one = res["forms"]["lifted"]
    assert one["chars"] == len(DOC) and len(one["top"]) == len(tk.CATEGORIES)
    assert sum(one["cl100k_base"]["cats"]) == one["cl100k_base"]["tokens"]
