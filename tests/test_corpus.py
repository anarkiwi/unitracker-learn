"""unitracker_learn.corpus: the document forms of a tune, cached, verified and listed."""

import csv
import json
from importlib import metadata
from types import SimpleNamespace

from unitracker_player.player import play

import optimiser_docs as D
from unitracker.optimiser.fit import TOLERANCES
from unitracker_learn import corpus


def test_versions_are_the_checkouts_commit_and_the_libraries(monkeypatch):
    monkeypatch.delenv("UNITRACKER_LEARN_COMMIT", raising=False)
    ran = []

    def git(cmd, **_):
        ran.append(cmd)
        return SimpleNamespace(stdout="f" * 40 + "\n")

    monkeypatch.setattr(corpus.subprocess, "run", git)
    corpus.versions.cache_clear()
    got = corpus.versions()
    assert got["unitracker-learn"] == "f" * 40 and set(corpus.LIBRARIES) <= set(got)
    assert ran[0][-2:] == ["rev-parse", "HEAD"]
    assert got["unitracker"] == metadata.version("unitracker")
    corpus.versions.cache_clear()


def test_versions_take_the_launchers_commit_else_the_package_else_none(monkeypatch):
    monkeypatch.setenv("UNITRACKER_LEARN_COMMIT", "abc")
    corpus.versions.cache_clear()
    assert corpus.versions()["unitracker-learn"] == "abc"
    monkeypatch.delenv("UNITRACKER_LEARN_COMMIT")
    monkeypatch.setattr(corpus.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError))
    corpus.versions.cache_clear()
    assert corpus.versions()["unitracker-learn"] == metadata.version("unitracker-learn")

    def missing(lib):
        raise metadata.PackageNotFoundError(lib)

    monkeypatch.setattr(corpus.metadata, "version", missing)
    corpus.versions.cache_clear()
    assert corpus.versions() == {"unitracker-learn": None}
    corpus.versions.cache_clear()


def test_a_lifted_song_is_verified_and_its_versions_recorded(tmp_path, lifter, monkeypatch):
    opts = {"out": str(tmp_path), "hvsc": None, "timeout": 0, "limit": 200, "retry": False}
    opts |= {"forms": ["lifted"]}
    row = {"path": str(tmp_path / "w.sid"), "subtune": 1}
    (tmp_path / "w.sid").write_bytes(b"w")
    got = corpus.forms_one(row, opts)
    assert got["lifted"] == "ok" and got["verified"] is True and got["bad_cells"] == 0
    assert lifter.lifted == [1] and lifter.verified == [(1, got["ticks"])]
    assert got["versions"] == corpus.versions() and got["ticks"] == corpus.song_ticks(D.HOIST, 200)
    meta = json.loads((tmp_path / "meta" / "w-s1.json").read_text())
    assert meta == {k: got[k] for k in meta}

    def fail(*_):
        raise RuntimeError("no grid")

    monkeypatch.setattr(corpus.cli, "verify", fail)
    got = corpus.forms_one(row | {"subtune": 0}, opts)
    assert got["lifted"] == "ok" and got["verified"] is False
    assert got["verify_error"] == "RuntimeError: no grid" and "bad_cells" not in got


def test_a_failed_forms_traceback_follows_its_line(tmp_path):
    sid = tmp_path / "junk.sid"
    sid.write_bytes(b"not a sid")
    opts = {"out": str(tmp_path), "hvsc": None, "timeout": 0, "limit": 10, "retry": False}
    opts["forms"] = ["lifted", "named"]
    got = corpus.forms_one({"path": str(sid)}, opts)
    assert got["lifted"].startswith("FormatError") and got["named"] == "no lifted form"
    text = (tmp_path / "lifted" / "junk.err").read_text()
    assert text.startswith(got["lifted"] + "\nTraceback")
    assert corpus.forms_one({"path": str(sid)}, opts)["lifted"] == got["lifted"]


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
    assert corpus.song_ticks(D.DEFAULTS, 1000) == len(play(D.DEFAULTS))
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


def test_the_forms_command_lifts_a_directory_and_tabulates(tmp_path, lifter, capsys):
    root = tmp_path / "tunes"
    (root / "x").mkdir(parents=True)
    (root / "x" / "a.sid").write_bytes(b"one")
    (root / "b.sid").write_bytes(b"two")
    out = tmp_path / "out"
    argv = [str(root), "--out", str(out), "--jobs", "1", "--forms", "lifted", "named"]
    assert corpus.main([*argv, "--limit", "500", "--ticks", "100"]) == 0
    assert "named: 2/2 {'stub': 2}" in capsys.readouterr().out
    table = {r["doc"].split("-", 1)[1]: r for r in csv.DictReader((out / "forms.csv").open())}
    assert set(table) == {"a", "b"} and table["a"]["named"] == "ok" and table["a"]["ticks"] != ""
    assert lifter.lifted == [0, 0] and (out / "meta").is_dir()
    assert corpus.main([*argv, "--subtune", "all", "--min-seconds", "1"]) == 0


def test_multi_sid_tunes_are_excluded_unless_asked_for(tmp_path, lifter, monkeypatch, capsys):
    monkeypatch.setattr(corpus.catalog, "multi_sid", lambda raw: True)
    (tmp_path / "m.sid").write_bytes(b"x")
    argv = [str(tmp_path / "m.sid"), "--out", str(tmp_path / "out"), "--jobs", "1"]
    argv += ["--forms", "lifted"]
    assert corpus.main(argv) == 0 and "1 multi-SID tunes excluded" in capsys.readouterr().out
    assert not (tmp_path / "out" / "lifted" / "m.err").exists() and not lifter.lifted
    assert corpus.main([*argv, "--multi-sid"]) == 0 and lifter.lifted == [0]


def test_a_form_that_raises_is_recorded_as_failed(tmp_path, lifter, monkeypatch):
    def slow(*_):
        raise TimeoutError("late")

    monkeypatch.setattr(corpus, "_form", slow)
    opts = _opts(tmp_path, forms=["lifted", "named"])
    (tmp_path / "t.sid").write_bytes(b"t")
    res = corpus.forms_one({"path": str(tmp_path / "t.sid")}, opts)
    assert res["named"] == "TimeoutError: late" and lifter.lifted == [0]
