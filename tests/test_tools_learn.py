"""tools/learn.py: the composer split, a tiny experiment end to end on synthetic documents on the
CPU, its status, and the container's launch commands."""

import csv
import hashlib
import json
from collections import Counter

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
        head=f"voices:1\ntempo:{4 + i}\n",
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
    cfg |= {"ft_ctx": 64, "ft_batch": 2, "ft_micro": 1, "prompt_tokens": 8, "doc_starts": 0.5}
    exp = learn.Experiment(cfg, tmp_path / "out" / "x", tunes=tmp_path / "list.csv")
    for i, row in enumerate(rows[:-1]):
        doc = learn.shared.name(row)
        for form in FORMS:
            exp.path(form, doc).parent.mkdir(parents=True, exist_ok=True)
            exp.path(form, doc).write_text(_doc(i + (form == "named")), encoding="utf-8")
        (exp.corpus / "meta").mkdir(exist_ok=True)
        (exp.corpus / "meta" / f"{doc}.json").write_text(
            json.dumps({"ticks": 100 + i, "verified": i % 3 > 0})
        )
    exp.path("named", learn.shared.name(rows[0])).unlink()
    (exp.out / "stages").mkdir(parents=True)
    (exp.out / "stages" / "corpus").write_text("0\n")
    return exp


def test_a_tunes_composer_is_its_musician_directory_else_the_tune():
    assert learn.composer("MUSICIANS/C/Cadaver/x.sid") == "MUSICIANS/Cadaver"
    assert learn.composer("MUSICIANS/C/Cadaver/sub/x.sid") == "MUSICIANS/Cadaver"
    assert learn.composer("GAMES/A-F/x.sid") == "GAMES/A-F/x.sid"


def test_a_groups_split_is_its_hash_against_the_held_fractions_whatever_the_others():
    groups = [f"g{i}" for i in range(4000)]
    got = learn.split(groups, [0.1, 0.2])
    assert got == {
        g: s for g, s in learn.split(groups[::-1] + ["new"], [0.1, 0.2]).items() if g != "new"
    }
    share = Counter(got.values())
    assert {s: round(share[s] / len(groups), 1) for s in learn.SPLITS} == {
        "train": 0.7,
        "val": 0.1,
        "test": 0.2,
    }
    u = {g: int(hashlib.sha1(g.encode()).hexdigest(), 16) / 16**40 for g in groups}
    span = {s: [u[g] for g in groups if got[g] == s] for s in learn.SPLITS}
    assert max(span["test"]) < 0.2 <= min(span["val"]) <= max(span["val"]) < 0.3
    assert min(span["train"]) >= 0.3


def test_songs_differing_only_in_meta_are_split_once(exp, caplog):
    exp.select()
    rows = sorted(exp.songs(), key=learn.shared.name)
    first, dup = (learn.shared.name(r) for r in rows[1:3])
    text = exp.path("lifted", first).read_text()
    exp.path("lifted", dup).write_text(text.replace("author:A", "author:B"))
    caplog.set_level("INFO")
    exp.split()
    assert first in exp.docs() and dup not in exp.docs() and len(exp.docs()) == 10
    assert "1 duplicates" in caplog.text


def test_the_corpus_lifts_each_song_of_the_selection(exp, monkeypatch):
    calls = []
    monkeypatch.setattr(learn.corpus, "main", calls.append)
    exp.forms()
    assert calls[0][calls[0].index("--subtune") + 1] == "all"
    exp.cfg |= {"subtunes": False, "min_seconds": 10}
    exp.forms()
    assert calls[1][calls[1].index("--subtune") :][:4] == ["--subtune", "0", "--min-seconds", "10"]
    rows = [{"path": "a/t.sid", "songs": 3, "lengths": "20 5 30"}, {"path": "b.sid", "songs": 2}]
    with (exp.out / "tunes.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, ["path", "songs", "lengths"])
        writer.writeheader()
        writer.writerows(rows)
    assert [learn.shared.name(r) for r in exp.songs()] == ["t", "b"]
    exp.cfg["subtunes"] = True
    assert [learn.shared.name(r) for r in exp.songs()] == ["t", "t-s2", "b", "b-s1"]

    cfg = learn.config("full", ["steps=7", 'forms=["all"]', "player=GoatTracker"])
    assert cfg["steps"] == 7 and cfg["forms"] == ["all"] and cfg["player"] == "GoatTracker"
    assert learn.PRESETS["full"]["steps"] != 7
    with pytest.raises(SystemExit):
        learn.config("smoke", ["nope=1"])


def test_windows_score_every_position_once_with_up_to_a_context_of_history(exp):
    for n in (2, 31, 32, 33, 100):
        chunks = exp.chunks(n, 32)
        scored = [p for b, e, f in chunks for p in range(f, e)]
        assert scored == list(range(1, n))
        assert all(e - b <= exp.cfg["ctx"] and b < f for b, e, f in chunks)


def _constrained_samples_are_grammar_prefixes(exp):
    for r in exp.runs() + exp.runs(scratch=False):
        rec = json.loads((exp.rundir(r) / "sample.json").read_text())
        for kind in ("header", "continue"):
            constrained = rec.get(f"{kind}_constrained", [])
            assert bool(constrained) == (r.tok == "bpe"), (r, kind)
            assert all(s["valid"] == 1.0 for s in constrained if s["ended"]), (r, kind)
            assert not any(s.get("stuck") for s in constrained), (r, kind)
            assert all(s["prefix"] == 1.0 for s in constrained), (r, kind)
            if r.tok == "bpe":
                assert all(0 <= s["prefix"] <= 1 for s in rec[kind])


def _verified_bits_are_the_verified_test_documents(exp, rows):
    test = exp.docs("test")
    ok = [d for d in test if exp.meta(d)["verified"]]
    for r in rows:
        run = learn.Run(r["form"], r["tokenizer"], "" if r["model"] == "scratch" else r["model"])
        per_doc = json.loads((exp.rundir(run) / "eval.json").read_text())["per_doc"]
        assert r["verified_docs"] == len(ok) < len(test)
        want = sum(per_doc[d] for d in ok) / sum(test[d]["ticks"] for d in ok)
        assert r["bits_per_tick_verified"] == pytest.approx(want)


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
                text = exp.text("lifted", d["doc"])
                assert learn.META in text and "author" not in text
                assert enc.decode(vocab[ids - 1]) == text and d["chars"] == len(text)
    rows = exp.results()
    assert len(rows) == 5 and rows[-1]["model"] == exp.cfg["ft_model"]
    forms = json.loads((exp.out / "finetune.json").read_text())
    best = min((r for r in rows[:4]), key=lambda r: r["bits_per_tick"])
    assert forms == [best["form"]] and rows[-1]["form"] == best["form"]
    ft = exp.rundir(learn.Run(best["form"], "cl100k_base", exp.cfg["ft_model"]))
    assert (ft / "best.pt").exists() and not (ft / "last.pt").exists()
    assert len(list((ft / "samples").glob("*.yaml"))) == 4
    _verified_bits_are_the_verified_test_documents(exp, rows)
    for r in rows:
        assert r["params"] > 0 and r["bits_per_tick"] > 0 and r["bits_per_token"] > 0
        assert 0 <= r["header_valid"] <= 1 and 0 <= r["continue_parses"] <= 1
        assert ("header_prefix" in r) == (r["tokenizer"] == "bpe")
        assert r["seen_tokens"] == (2 * 64 if r["model"] != "scratch" else 4 * 32) * 4
    logged = [
        json.loads(ln) for ln in (exp.rundir(learn.Run("lifted", "bpe", "")) / "log.jsonl").open()
    ]
    assert [r["step"] for r in logged] == [2, 4]
    table = (exp.out / "results.md").read_text()
    _constrained_samples_are_grammar_prefixes(exp)
    assert "| lifted | bpe | scratch |" in table and "| lifted | " in table.split("zstd_dict")[1]
    base = json.loads((exp.out / "baselines.json").read_text())
    assert (
        0 < base["lifted"]["zstd_dict"] < base["lifted"]["xz"]
        and "unigram_cl100k_base" in base["lifted"]
    )
    for r in exp.runs() + exp.runs(scratch=False):
        probe = json.loads((exp.rundir(r) / "probe.json").read_text())
        cats = probe["categories"]
        assert cats["key"]["tokens"] > 0 and all(c["bits"] > 0 for c in cats.values())
        ctx = exp.hp(r, "ctx")
        hist = {str(h) for h in (16, 64, 256, ctx // 2, 3 * ctx // 4) if h <= 3 * ctx // 4}
        assert set(probe["context"]) == hist and all(v > 0 for v in probe["context"].values())
    assert "stage report done" in caplog.text
    caplog.clear()
    exp.run()
    assert "stage" not in caplog.text


def test_fine_tunes_take_the_forms_named(exp):
    exp.cfg |= {"tokenizers": ["bpe", "cl100k_base"], "ft_forms": ["lifted"], "ft_steps": 2}
    exp.run()
    assert json.loads((exp.out / "finetune.json").read_text()) == ["lifted"]
    assert [r["form"] for r in exp.results() if r["model"] != "scratch"] == ["lifted"]


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


def test_samples_continue_their_prompts_and_documents_parse_and_play(exp, monkeypatch):
    exp.cfg["tokenizers"] = ["bpe"]
    exp.cfg["forms"] = ["lifted"]
    exp.run()
    run, text = learn.Run("lifted", "bpe", ""), learn.META_LINE.sub(learn.META, _doc(3), count=1)
    play(text, ticks=50)
    prompts = exp.prompts(run)
    assert prompts["header"].shape == (2, prompts["header"].shape[1])
    assert prompts["continue"].shape[1] == 9 and (prompts["continue"][:, 0] == learn.EOS).all()
    vocab, enc = np.load(exp.data("bpe") / "vocab.npy"), exp.encoder("bpe")
    assert enc.decode(vocab[prompts["header"][0, 1:] - 1]) == learn.HEADER
    ids = np.searchsorted(vocab, enc.encode([text])[0]) + 1

    def generate(_, __, prompts, stuck=None):  # pylint: disable=unused-argument
        rest = [learn.EOS, 5] if prompts.shape[1] == 9 else ids[prompts.shape[1] - 1 :]
        return np.stack([np.concatenate([p, rest, [learn.EOS, 5]]) for p in prompts])

    monkeypatch.setattr(exp, "generate", generate)
    rd = exp.rundir(run)
    (rd / "sample.json").unlink()
    exp.sample(run)
    rec = json.loads((rd / "sample.json").read_text())
    assert rec["header_ended"] == rec["header_parses"] == rec["header_plays"] == 1.0
    assert rec["header_valid"] == 1.0 and (rd / "samples" / "header-0.yaml").read_text() == text
    assert len(rec["continue"]) == 2 and (rd / "samples" / "continue-1.yaml").exists()


def test_validity_is_the_share_generated_before_the_first_parse_error(exp):
    text = _doc(1)
    cut = text.index("song:")
    broken = text[:cut] + "]]" + text[cut:]
    rec = exp.check(broken, 10)
    assert not rec["parses"] and rec["valid"] == pytest.approx((cut - 10) / (len(broken) - 10))
    assert exp.check(text, 10)["valid"] == 1.0
    rec = exp.check(text.replace("voices:1", "voices:2"), 0)
    assert rec["parses"] != rec["plays"] or rec["valid"] < 1


def test_a_share_of_training_windows_start_at_documents(exp):
    data = np.arange(1, 1001, dtype=np.uint32)
    data[[0, 500]] = learn.EOS
    run = learn.Run("lifted", "bpe", "")
    x, y = exp.windows(run, data, np.random.default_rng(0), 8, np.array([0, 500]))
    assert (x[:4, 0] == learn.EOS).all() and x.shape == y.shape == (8, 32)
    assert (y[:, :-1] == x[:, 1:]).all() and (y[:, -1] == -100).all()
    x, y = exp.windows(run, data[:20], np.random.default_rng(0), 2)
    assert (x[:, 20:] == learn.EOS).all() and (y[:, :19] == x[:, 1:20]).all()
    assert (y[:, 19:] == -100).all()


def test_meta_is_kept_where_asked(exp):
    exp.cfg["meta"] = True
    assert "author:A" in exp.text("lifted", learn.shared.name({"path": "t0", "md5": "0" * 32}))


def test_an_error_is_located_in_its_innermost_open_construct():
    doc = "a:[b,{c:d(1,"
    assert [learn.where(doc, n) for n in (1, 4, 8, 12)] == ["top", "list", "record", "call"]
    assert learn.where("a:[b](c)", 8) == "top"


def test_a_pretrained_models_eos_is_fetched_once(monkeypatch):
    import transformers  # pylint: disable=import-outside-toplevel

    calls = []

    def config(name):
        calls.append(name)
        return type("C", (), {"eos_token_id": [7, 8]})

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config)
    learn.eos_id.cache_clear()
    assert learn.eos_id("m") == learn.eos_id("m") == 7 and calls == ["m"]
    learn.eos_id.cache_clear()


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
    tunes = tmp_path / "t" / "list.csv"
    build, rm, run = learn.launch(["--preset", "full"], tmp_path / "x", tmp_path / "h", tunes)
    assert f"{tunes.parent}:{tunes.parent}" in run
    assert build[:2] == ["docker", "build"] and rm == ["docker", "rm", "-f", "unitracker-learn-x"]
    assert run[:3] == ["docker", "run", "-d"] and "nvidia.com/gpu=all" in run
    assert f"{tmp_path}:{tmp_path}" in run and f"{tmp_path / 'h'}:{tmp_path / 'h'}" in run
    assert run[-1].startswith("python tools/learn.py run --preset full >> ")
    assert f"UNITRACKER_COMMIT={learn.corpus.versions()['unitracker']}" in run


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


def test_the_chunked_loss_is_the_models_own_and_its_gradient(base):
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForCausalLM  # pylint: disable=import-outside-toplevel

    model = AutoModelForCausalLM.from_pretrained(base)
    x = torch.randint(0, 100000, (3, 40))
    want = model(input_ids=x, labels=x).loss
    want.backward()
    grad = model.get_output_embeddings().weight.grad.clone()
    model.zero_grad()
    y = torch.nn.functional.pad(x[:, 1:], (0, 1), value=-100)
    got = learn.nll(model, x, y, chunk=7).sum() / (3 * 39)
    got.backward()
    assert got.item() == pytest.approx(want.item(), rel=1e-5)
    assert torch.allclose(model.get_output_embeddings().weight.grad, grad, atol=1e-6)


def test_training_stops_after_patience_evaluations_without_a_better(exp, monkeypatch):
    exp.cfg |= {"tokenizers": ["bpe"], "steps": 20, "eval_every": 1, "patience": 2}
    for name, fn in exp.stages()[:5]:
        if name != "corpus":
            fn()
    vals = iter([3.0, 2.0, 2.5, 2.1, 1.0])
    monkeypatch.setattr(exp, "loss", lambda *_: next(vals))
    run = learn.Run("lifted", "bpe", "")
    exp.train(run)
    rec = json.loads((exp.rundir(run) / "train.json").read_text())
    assert rec["steps_run"] == 4 and rec["best_val"] == 2.0 and rec["seen_tokens"] == 4 * 4 * 32


def test_a_lora_fine_tune_trains_and_keeps_only_its_adapters(exp):
    exp.cfg |= {"tokenizers": ["bpe", "cl100k_base"], "ft_forms": ["lifted"], "ft_steps": 2}
    exp.cfg["ft_lora"] = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
    exp.run()
    run = learn.Run("lifted", "cl100k_base", exp.cfg["ft_model"])
    best = __import__("torch").load(exp.rundir(run) / "best.pt")
    assert best and all("lora_" in k for k in best)
    rec = json.loads((exp.rundir(run) / "train.json").read_text())
    assert 0 < rec["trained"] < rec["params"]
    assert json.loads((exp.rundir(run) / "eval.json").read_text())["bits_per_tick"] > 0


def test_a_row_the_grammar_leaves_no_model_id_is_ended_and_stuck():
    import torch  # pylint: disable=import-outside-toplevel
    import constrain  # pylint: disable=import-outside-toplevel

    class Grammar:  # pylint: disable=too-few-public-methods
        """Masks that leave the second row only a native id no model id maps to."""

        def masks(self, _, tensor=False):  # pylint: disable=unused-argument
            return torch.tensor([[True, False, False, False], [False, False, False, True]])

    masked = constrain.Masked(Grammar(), [None, None], np.array([2, 1, 0]), 1, True)
    out = masked(torch.zeros((2, 5), dtype=torch.long), torch.zeros((2, 3)))
    assert masked.stuck == {1} and torch.isinf(out).tolist() == [
        [True, True, False],
        [True, False, True],
    ]
