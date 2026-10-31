"""unitracker_learn.catalog: tune lists, their songs and names, and multi-SID detection."""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from unitracker_learn import catalog


def _sid(version: int, at: int = 0, value: int = 0) -> bytes:
    """A PSID header of ``version`` with byte ``at`` set to ``value``."""
    out = bytearray(0x7C + 4)
    out[0:4], out[4:6], out[6:8] = b"PSID", version.to_bytes(2, "big"), (0x7C).to_bytes(2, "big")
    out[0x0E:0x12] = (1).to_bytes(2, "big") * 2
    if at:
        out[at] = value
    return bytes(out)


@pytest.mark.parametrize(
    "version, at, multi",
    [(2, 0x7A, False), (3, 0x7A, True), (3, 0x7B, False), (4, 0x7B, True), (4, 0x7C, False)],
)
def test_a_tune_is_multi_sid_by_its_header_sid_addresses(version, at, multi, monkeypatch):
    monkeypatch.setattr(catalog, "find_format", lambda raw: SimpleNamespace(name="other"))
    assert catalog.multi_sid(_sid(version, at, 0x42)) == multi


@pytest.mark.parametrize("voices, multi", [((0, 7, 14), False), ((0, 7, 0x20), True)])
def test_a_goattracker_build_past_three_voices_is_multi_sid(monkeypatch, voices, multi):
    playback = SimpleNamespace(voices=voices)
    dec = SimpleNamespace(info=SimpleNamespace(playback=playback))
    monkeypatch.setattr(catalog, "find_format", lambda raw: SimpleNamespace(name="goattracker2"))
    monkeypatch.setattr(catalog, "decompile_sid", lambda raw: dec)
    assert catalog.multi_sid(_sid(2)) == multi


def test_unreadable_tunes_are_single_sid():
    assert not catalog.multi_sid(b"PSID" + bytes(9))


def _csv(path: Path, fields: list, rows: list) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(i, player, songs=1):
    return {"path": f"d/t{i}.sid", "md5": f"{i:032x}", "player": player, "songs": songs}


def test_tune_lists_filter_by_player_and_subtunes_and_name_their_outputs(tmp_path):
    rows = [_row(1, "GoatTracker_V2.x"), _row(2, "GoatTracker_V2.x", 3), _row(3, "DefMon")]
    lst = _csv(tmp_path / "tunes.csv", ["path", "md5", "player", "songs"], rows)
    assert [r["path"] for r in catalog.rows([lst], "goat", 1)] == ["d/t1.sid"]
    assert len(catalog.rows([lst, "x/y.sid"])) == 4
    assert catalog.name(rows[0]) == "00000000-t1" and catalog.name({"path": "a/b.sid"}) == "b"
    assert catalog.tune(lst) == lst


def test_a_directorys_tunes_are_its_sid_files_with_their_md5(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "b.sid").write_bytes(b"b")
    (tmp_path / "a.sid").write_bytes(b"a")
    (tmp_path / "note.txt").write_text("not a tune")
    got = catalog.rows([tmp_path])
    assert [Path(r["path"]).name for r in got] == ["a.sid", "b.sid"]
    assert got[0]["md5"] == "0cc175b9c0f1b6a831c399e269772661"


def test_a_tune_is_found_under_hvsc_else_by_name(tmp_path, monkeypatch):
    (tmp_path / "M").mkdir()
    (tmp_path / "M" / "a.sid").write_bytes(b"a")
    assert catalog.tune("M/a.sid", tmp_path) == tmp_path / "M" / "a.sid"
    monkeypatch.setattr("pysidtracker.testing.resolve_tune", lambda name: None)
    with pytest.raises(FileNotFoundError, match="no such tune"):
        catalog.tune("M/none.sid", tmp_path)
    monkeypatch.setattr("pysidtracker.testing.resolve_tune", lambda name: str(tmp_path / name))
    assert catalog.tune("M/none.sid") == tmp_path / "M" / "none.sid"


def test_a_tunes_songs_are_its_subtunes_but_those_too_short():
    rows = [_row(1, "x", 3) | {"lengths": "30 5.5 12"}, _row(2, "x", 2), {"path": "a/b.sid"}]
    got = catalog.songs(rows, "all", 10)
    assert [catalog.name(r) for r in got] == [
        "00000000-t1",
        "00000000-t1-s2",
        "00000000-t2",
        "00000000-t2-s1",
        "b",
    ]
    assert [r["subtune"] for r in catalog.songs(rows, 1)] == [1, 1, 1]
    assert len(catalog.songs(rows)) == 3 and catalog.songs(rows, "all", 31) == got[2:]
