"""tools/learn.py: the composer split, a tiny experiment end to end on synthetic documents on the
CPU, its status, and the container's launch commands."""

import csv
import json

import learn
import numpy as np
import pytest
from unitracker_player.player import play

import optimiser_docs as D

NOTES = ("C4", "D4", "E4", "G4", "A4", "C5")
FORMS = ("lifted", "named")
TOKENIZERS = ("bpe", "cl100k_base")


def _doc(i: int) -> str:
    notes = ",".join(NOTES[(i * k) % len(NOTES)] for k in range(1, 9))
    return D.doc(
        "      ins:\n"
        "        i1:{ad:9,sr:240,ctrl:c1,pw:2048}\n"
        "      ctrl:\n"
        "        c1:[{wait:0,ctrl:65,note:+0},{hold:1,ctrl:64},end]\n"
        "      phrase:\n"
        f"        p1:[C4.i1,{notes}]\n",
        head="voices:1\ntempo:6\n",
        mods="",
    )


@pytest.fixture(name="base", scope="session")
def _base(tmp_path_factory):
    """A tiny untrained pretrained-format model over cl100k_base's ids."""
    from transformers import Qwen2Config, Qwen2ForCausalLM  # pylint: disable=C0415

    path = tmp_path_factory.mktemp("base")
    shape = Qwen2Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=100352,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        eos_token_id=100257,
    )
    Qwen2ForCausalLM(shape).save_pretrained(path)
    return str(path)


@pytest.fixture(name="exp")
def _experiment(tmp_path, base):
    """An experiment over 12 synthetic tunes by 6 composers, its corpus written."""
    rows = [
        {"path": f"MUSICIANS/A/C{i % 6}/t{i}.sid", "md5": f"{i:032x}", "player": "DefMon"}
        for i in range(12)
    ] + [{"path": "GAMES/G/g.sid", "md5": "f" * 32, "player": "Other"}]
    with (tmp_path / "list.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg = learn.config("smoke", [])
    cfg |= {"tunes": 0, "forms": list(FORMS), "tokenizers": list(TOKENIZERS), "vocab": 300}
    cfg |= {"hidden": 32, "layers": 1, "heads": 2, "ctx": 32, "batch": 4, "micro": 2}
    cfg |= {"steps": 4, "warmup": 1, "eval_every": 2, "val_windows": 2, "batch_eval": 3}
    cfg |= {"samples": 2, "sample_tokens": 40, "play_ticks": 50, "held": [0.15, 0.15]}
    cfg |= {"ft_model": base, "ft_tokenizer": "cl100k_base", "ft_batch_eval": 2}
    exp = learn.Experiment(cfg, tmp_path / "out" / "x", tunes=tmp_path / "list.csv")
    for i, row in enumerate(rows[:-1]):
        doc = learn.shared.name(row)
        for form in FORMS:
            exp.path(form, doc).parent.mkdir(parents=True, exist_ok=True)
            exp.path(form, doc).write_text(_doc(i + (form == "named")), encoding="utf-8")
        (exp.corpus / "meta").mkdir(exist_ok=True)
        (exp.corpus / "meta" / f"{doc}.json").write_text(json.dumps({"ticks": 100 + i}))
    exp.path("named", learn.shared.name(rows[0])).unlink()
    (exp.out / "stages").mkdir(parents=True)
    (exp.out / "stages" / "corpus").write_text("0\n")
    return exp


def test_a_tunes_composer_is_its_musician_directory_else_the_tune():
    assert learn.composer("MUSICIANS/C/Cadaver/x.sid") == "MUSICIANS/Cadaver"
    assert learn.composer("MUSICIANS/C/Cadaver/sub/x.sid") == "MUSICIANS/Cadaver"
    assert learn.composer("GAMES/A-F/x.sid") == "GAMES/A-F/x.sid"


@pytest.mark.parametrize("sizes", [{"a": 1, "b": 1, "c": 1}, {f"g{i}": i + 1 for i in range(40)}])
def test_groups_fill_test_then_val_to_their_fractions_the_rest_train(sizes):
    got = learn.split(sizes, [0.1, 0.2])
    assert got == learn.split(dict(reversed(sizes.items())), [0.1, 0.2])
    docs = {s: sum(n for g, n in sizes.items() if got[g] == s) for s in learn.SPLITS}
    total = sum(sizes.values())
    assert docs["test"] >= 0.2 * total and docs["val"] >= 0.1 * total and docs["train"]
    if len(sizes) > 3:
        assert docs["train"] > 0.6 * total


def test_presets_take_json_or_string_overrides_of_their_keys():
    cfg = learn.config("full", ["steps=7", 'forms=["all"]', "player=GoatTracker"])
    assert cfg["steps"] == 7 and cfg["forms"] == ["all"] and cfg["player"] == "GoatTracker"
    assert learn.PRESETS["full"]["steps"] != 7
    with pytest.raises(SystemExit):
        learn.config("smoke", ["nope=1"])


def test_windows_score_every_position_once_with_up_to_a_context_of_history(exp):
    for n in (2, 31, 32, 33, 100):
        chunks = exp.chunks(n)
        scored = [p for b, e, f in chunks for p in range(f, e)]
        assert scored == list(range(1, n))
        assert all(e - b <= exp.cfg["ctx"] and b < f for b, e, f in chunks)


def test_an_experiment_runs_end_to_end_and_resumes(exp, caplog):  # pylint: disable=R0914
    caplog.set_level("INFO")
    exp.run()
    split = exp.docs()
    assert len(split) == 11 and {v["split"] for v in split.values()} == set(learn.SPLITS)
    groups = {}
    for v in split.values():
        groups.setdefault(v["group"], set()).add(v["split"])
    assert all(len(s) == 1 for s in groups.values())
    for tok in TOKENIZERS:
        vocab = np.load(exp.data(tok) / "vocab.npy")
        enc = exp.encoder(tok)
        for which in learn.SPLITS:
            flat = np.load(exp.data(tok) / f"lifted.{which}.npy")
            index = json.loads((exp.data(tok) / f"lifted.{which}.json").read_text())
            assert flat[0] == flat[-1] == learn.EOS and (flat == learn.EOS).sum() == len(index) + 1
            for d in index:
                ids = flat[d["start"] + 1 : d["start"] + 1 + d["tokens"]]
                assert flat[d["start"]] == learn.EOS and learn.EOS not in ids
                text = exp.path("lifted", d["doc"]).read_text()
                assert enc.decode(vocab[ids - 1]) == text and d["chars"] == len(text)
    rows = exp.results()
    assert len(rows) == 5 and rows[-1]["model"] == exp.cfg["ft_model"]
    forms = json.loads((exp.out / "finetune.json").read_text())
    best = min((r for r in rows[:4]), key=lambda r: r["bits_per_tick"])
    assert forms == [best["form"]] and rows[-1]["form"] == best["form"]
    ft = exp.rundir(learn.Run(best["form"], "cl100k_base", exp.cfg["ft_model"]))
    assert (ft / "best.pt").exists() and not (ft / "last.pt").exists()
    assert len(list((ft / "samples").glob("*.yaml"))) == 2
    for r in rows:
        assert r["params"] > 0 and r["bits_per_tick"] > 0 and r["bits_per_token"] > 0
        assert 0 <= r["parses"] <= 1 and r["seen_tokens"] == 4 * 4 * 32
    logged = [
        json.loads(ln) for ln in (exp.rundir(learn.Run("lifted", "bpe", "")) / "log.jsonl").open()
    ]
    assert [r["step"] for r in logged] == [2, 4]
    table = (exp.out / "results.md").read_text()
    assert table.count("\n") == 7 and "| lifted | bpe | scratch |" in table
    assert "stage report done" in caplog.text
    caplog.clear()
    exp.run()
    assert "stage" not in caplog.text


def test_training_resumes_from_its_last_checkpoint(exp, caplog):
    exp.cfg["tokenizers"] = ["bpe"]
    for name, fn in exp.stages()[:5]:
        if name != "corpus":
            fn()
    exp.cfg["steps"] = 2
    exp.train(learn.Run("lifted", "bpe", ""))
    (exp.rundir(learn.Run("lifted", "bpe", "")) / "train.json").unlink()
    exp.cfg["steps"] = 4
    caplog.set_level("INFO")
    exp.train(learn.Run("lifted", "bpe", ""))
    assert "resumed at step 2" in caplog.text
    rec = json.loads((exp.rundir(learn.Run("lifted", "bpe", "")) / "train.json").read_text())
    assert rec["seen_tokens"] == 4 * 4 * 32 and rec["best_val"] > 0


def test_samples_that_are_documents_parse_and_play(exp, monkeypatch):
    exp.cfg["tokenizers"] = ["bpe"]
    exp.cfg["forms"] = ["lifted"]
    exp.run()
    text = _doc(3)
    play(text, ticks=50)
    vocab = np.load(exp.data("bpe") / "vocab.npy")
    ids = np.searchsorted(vocab, exp.encoder("bpe").encode([text])[0]) + 1
    monkeypatch.setattr(exp, "generate", lambda m, n, eos: np.stack([np.append(ids, [0, 5])] * n))
    rd = exp.rundir(learn.Run("lifted", "bpe", ""))
    (rd / "sample.json").unlink()
    exp.sample(learn.Run("lifted", "bpe", ""))
    rec = json.loads((rd / "sample.json").read_text())
    assert rec["ended"] == rec["parses"] == rec["plays"] == 1.0
    assert (rd / "samples" / "0.yaml").read_text() == text


def test_status_reports_stages_corpus_runs_and_the_log(exp):
    assert "not started" in learn.status(exp.out)
    exp.cfg["tokenizers"] = ["bpe"]
    exp.run()
    (exp.out / "learn.log").write_text("first\nlast line\n")
    got = learn.status(exp.out)
    assert "report     done" in got and "corpus lifted     12/12 written" in got
    assert "corpus named      11/12 written" in got and '"step": 4' in got
    assert got.endswith("last line")


def test_the_tune_list_is_filtered_and_sampled(exp):
    exp.cfg["player"] = "DefMon"
    exp.cfg["tunes"] = 5
    exp.select()
    with (exp.out / "tunes.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 and {r["player"] for r in rows} == {"DefMon"}


def test_launch_builds_then_runs_detached_on_the_gpus(tmp_path):
    build, rm, run = learn.launch(["--preset", "full"], tmp_path / "x", tmp_path / "h")
    assert build[:2] == ["docker", "build"] and rm == ["docker", "rm", "-f", "unitracker-learn-x"]
    assert run[:3] == ["docker", "run", "-d"] and "nvidia.com/gpu=all" in run
    assert f"{tmp_path}:{tmp_path}" in run and f"{tmp_path / 'h'}:{tmp_path / 'h'}" in run
    assert run[-1].startswith("python tools/learn.py run --preset full >> ")


def test_the_command_line_runs_and_reports_status(exp, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(learn.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    out = str(exp.out)
    assert learn.main(["launch", "--out", out, "--hvsc", out]) == 0
    assert len(calls) == 3 and "--out" in calls[2][-1]
    assert learn.main(["status", "--out", out]) == 0
    assert "not started" in capsys.readouterr().out
    monkeypatch.setattr(learn.Experiment, "run", lambda self, redo: calls.append(redo))
    assert learn.main(["run", "--out", out, "--redo", "eval"]) == 0
    assert calls[-1] == ["eval"]
