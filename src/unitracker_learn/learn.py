"""An LLM learnability experiment over document forms and tokenizers (docs/learnability.md).

Per form and tokenizer, a decoder trained from scratch for the same steps on HVSC tunes' forms,
scored in bits per song tick on held-out composers' tunes and sampled for playable documents.
Usage: ``python tools/learn.py {run,status,launch} [--preset smoke|full] [--set KEY=VALUE]``.
"""

import argparse
import csv
import functools
import hashlib
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

import corpus
import shared
from lm import LN2, META, META_LINE, device, nll, progress
from unitracker.optimiser import tokenizer as tk

OUT = shared.ROOT / "out" / "learn"
IMAGE = "unitracker-learn"
SPLITS = ("train", "val", "test")
EOS = 0
Run = namedtuple("Run", "form tok base")
Run.name = property(lambda r: ".".join([r.form, r.tok] + [r.base.split("/")[-1]] * bool(r.base)))
HEADER = "unitracker:2.3\n"
SAMPLES = ("header", "continue", "header_constrained", "continue_constrained")
OPEN = {"(": "call", "[": "list", "{": "record"}
COMMON = {
    "player": "GoatTracker_V2",
    "held": [0.05, 0.1],
    "seed": 0,
    "jobs": max(1, len(os.sched_getaffinity(0)) - 2),
    "lr": 1e-3,
    "batch_eval": 16,
    "play_ticks": 3000,
    "ft_model": "Qwen/Qwen2.5-0.5B",
    "ft_lora": None,
    "patience": 0,
    "ft_tokenizer": "qwen2.5",
    "ft_forms": 2,
    "ft_lr": 5e-5,
    "ft_micro": 1,
    "ft_batch_eval": 1,
    "doc_starts": 0.25,
    "meta": False,
    "probe_docs": 96,
    "constrain": True,
    "subtunes": True,
    "min_seconds": 0,
}
PRESETS = {
    "smoke": COMMON
    | {
        "tunes": 8,
        "forms": ["lossy-single", "lossy-all"],
        "tokenizers": ["bpe", "qwen2.5"],
        "vocab": 512,
        "timeout": 300,
        "hidden": 64,
        "layers": 2,
        "heads": 2,
        "ctx": 256,
        "batch": 4,
        "micro": 4,
        "steps": 40,
        "warmup": 10,
        "eval_every": 20,
        "val_windows": 8,
        "samples": 2,
        "sample_tokens": 512,
        "prompt_tokens": 32,
        "play_ticks": 300,
        "ft_ctx": 512,
        "ft_batch": 2,
        "ft_forms": 1,
        "ft_steps": 4,
        "ft_warmup": 1,
        "ft_eval_every": 2,
    },
    "full": COMMON
    | {
        "tunes": 0,
        "min_seconds": 10,
        "forms": ["lossy", "lossy-single", "lossy-all", "lossy-nophrases-all"],
        "tokenizers": ["bpe", "qwen2.5"],
        "vocab": 8192,
        "timeout": 900,
        "hidden": 512,
        "layers": 8,
        "heads": 8,
        "ctx": 4096,
        "batch": 16,
        "micro": 4,
        "steps": 5000,
        "warmup": 200,
        "eval_every": 250,
        "val_windows": 256,
        "samples": 16,
        "sample_tokens": 16384,
        "ft_forms": ["lossy-all"],
        "ft_model": "Qwen/Qwen2.5-1.5B",
        "ft_lora": {"r": 64, "lora_alpha": 128, "lora_dropout": 0.05},
        "lr": 1e-3,
        "ft_lr": 2e-4,
        "patience": 4,
        "prompt_tokens": 512,
        "ft_ctx": 8192,
        "ft_batch": 8,
        "ft_steps": 2000,
        "ft_warmup": 100,
        "ft_eval_every": 250,
    },
}
log = logging.getLogger("learn")


def composer(path: str) -> str:
    """A tune's split group: its HVSC musician directory, else the tune itself."""
    parts = Path(path).parts
    return "/".join(parts[:3:2]) if parts[0] == "MUSICIANS" and len(parts) > 3 else path


def split(groups, held: list) -> dict:
    """``{group: split}``: a group's SHA-1 as a fraction u of 1 is ``test`` under ``held[1]``,
    ``val`` under ``held[0] + held[1]`` (``held`` ``[val, test]``), else ``train``, whatever the
    other groups; a held split no group falls in takes the training group of least u."""
    u = {g: int(hashlib.sha1(g.encode()).hexdigest(), 16) / 16**40 for g in groups}
    out = {g: "test" if v < held[1] else "val" if v < sum(held) else "train" for g, v in u.items()}
    for name in ("test", "val"):
        train = [g for g in u if out[g] == "train"]
        if train and name not in out.values():
            out[min(train, key=u.get)] = name
    return out


@functools.cache
def eos_id(base: str) -> int:
    """A pretrained model's end of document id, its configuration fetched once."""
    from transformers import AutoConfig  # pylint: disable=import-outside-toplevel,import-error

    eos = AutoConfig.from_pretrained(base).eos_token_id
    return eos[0] if isinstance(eos, list) else eos


def trained_state(model) -> dict:
    """A model's trained weights (all of a model trained whole, a LoRA model's adapters)."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    if len(names) == len(list(model.parameters())):
        return model.state_dict()
    return {k: v for k, v in model.state_dict().items() if k in names}


def where(doc: str, pos: int) -> str:
    """The construct open at ``pos`` of a document: its innermost unclosed bracket's kind (a
    call, list or record), else ``top``."""
    stack = []
    for ch in doc[:pos]:
        if ch in OPEN:
            stack.append(OPEN[ch])
        elif ch in ")]}" and stack:
            stack.pop()
    return stack[-1] if stack else "top"


class Experiment:  # pylint: disable=too-many-public-methods
    """One experiment's directory, configuration and stages."""

    def __init__(self, cfg: dict, out: Path, hvsc: Path | None = None, tunes: Path | None = None):
        self.cfg, self.out, self.hvsc, self.tunes = cfg, out, hvsc, tunes
        self.corpus = out.parent / "corpus"

    def stages(self) -> list:
        """The stages in order, each ``(name, function)``."""
        return [
            ("tunes", self.select),
            ("corpus", self.forms),
            ("split", self.split),
            ("tokenizers", self.train_bpe),
            ("encode", self.encode),
            ("train", lambda: self.each(self.train)),
            ("eval", lambda: self.each(self.evaluate)),
            ("sample", lambda: self.each(self.sample)),
            ("finetune", self.finetune),
            ("probe", self.probe),
            ("report", self.report),
        ]

    def run(self, redo=()) -> None:
        """Each stage not yet done or named in ``redo``, marked done after."""
        (self.out / "stages").mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(self.cfg, indent=1), encoding="utf-8")
        for name, fn in self.stages():
            mark = self.out / "stages" / name
            if mark.exists() and name not in redo:
                continue
            log.info("stage %s", name)
            start = time.time()
            fn()
            mark.write_text(f"{time.time() - start:.0f}\n", encoding="utf-8")
            log.info("stage %s done in %.0fs", name, time.time() - start)

    def runs(self, scratch: bool = True) -> list:
        """The runs: each form and tokenizer's from scratch, or with ``scratch`` False, each
        ``finetune.json`` form's fine-tune of ``ft_model``."""
        if scratch:
            return [Run(f, t, "") for f in self.cfg["forms"] for t in self.cfg["tokenizers"]]
        ft = self.out / "finetune.json"
        forms = json.loads(ft.read_text(encoding="utf-8")) if ft.exists() else []
        return [Run(f, self.cfg["ft_tokenizer"], self.cfg["ft_model"]) for f in forms]

    def each(self, fn) -> None:
        """``fn(run)`` for each from-scratch run."""
        for run in self.runs():
            fn(run)

    def select(self) -> None:
        """``tunes.csv``: the tune list's ``player`` rows, all or a seeded sample of ``tunes``;
        without a list, HVSC surveyed once into ``out/learn/tunes.csv``."""
        source = self.tunes or OUT / "tunes.csv"
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-m", "pysidtracker.survey", "find", str(self.hvsc)]
            subprocess.run(cmd + ["--out", str(source), "--player", self.cfg["player"]], check=True)
        rows = shared.rows([source], self.cfg["player"])
        if self.cfg["tunes"]:
            rng = np.random.default_rng(self.cfg["seed"])
            pick = rng.choice(len(rows), min(self.cfg["tunes"], len(rows)), replace=False)
            rows = [rows[i] for i in sorted(pick)]
        with (self.out / "tunes.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        log.info("%d tunes", len(rows))

    def selection(self) -> tuple:
        """``(subtune, min_seconds)``: every subtune with ``subtunes``, else the first, and the
        shortest song kept."""
        return "all" if self.cfg.get("subtunes") else 0, self.cfg.get("min_seconds", 0)

    def songs(self) -> list:
        """``tunes.csv``'s songs (:func:`shared.songs` of :meth:`selection`)."""
        return shared.songs(shared.rows([self.out / "tunes.csv"]), *self.selection())

    def forms(self) -> None:
        """Each song's forms, cached in the corpus the experiments share."""
        subtune, seconds = self.selection()
        args = ["forms", str(self.out / "tunes.csv"), "--out", str(self.corpus)]
        args += ["--subtune", str(subtune), "--min-seconds", str(seconds)]
        args += ["--forms", *self.cfg["forms"], "--timeout", str(self.cfg["timeout"])]
        args += ["--jobs", str(self.cfg["jobs"])]
        corpus.main(args + (["--hvsc", str(self.hvsc)] if self.hvsc else []))

    def path(self, form: str, doc: str) -> Path:
        """A document's form in the corpus."""
        return self.corpus / form / f"{doc}.yaml"

    def meta(self, doc: str) -> dict:
        """A document's corpus record (:func:`corpus.forms_one`), empty if it has none."""
        path = self.corpus / "meta" / f"{doc}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def split(self) -> None:
        """``split.json``: each song every form has, its split (by composer) and song ticks, of
        songs whose first form is the same but for its meta line only the first by name."""
        songs, out, found = sorted(self.songs(), key=shared.name), {}, 0
        for row in songs:
            doc = shared.name(row)
            ticks = self.meta(doc).get("ticks", 0)
            if not ticks or not all(self.path(f, doc).exists() for f in self.cfg["forms"]):
                continue
            text = META_LINE.sub("", self.path(self.cfg["forms"][0], doc).read_text("utf-8"), 1)
            key = hashlib.sha1(text.encode()).digest()
            out.setdefault(key, (doc, {"group": composer(row["path"]), "ticks": ticks}))
            found += 1
        of = split({v["group"] for _, v in out.values()}, self.cfg["held"])
        dups = found - len(out)
        out = {d: {"split": of[v["group"]]} | v for d, v in sorted(out.values())}
        (self.out / "split.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        counts = {s: sum(v["split"] == s for v in out.values()) for s in SPLITS}
        log.info(
            "%d of %d songs in every form, %d duplicates: %s", len(out), len(songs), dups, counts
        )

    def docs(self, which: str | None = None) -> dict:
        """``split.json``'s documents, those of split ``which`` if given."""
        out = json.loads((self.out / "split.json").read_text(encoding="utf-8"))
        return {d: v for d, v in out.items() if which in (None, v["split"])}

    def text(self, form: str, doc: str) -> str:
        """A document's form as the models read it: its meta line :data:`META` unless ``meta``
        (titles and authors are no part of a tune)."""
        text = self.path(form, doc).read_text(encoding="utf-8")
        return text if self.cfg["meta"] else META_LINE.sub(META, text, count=1)

    def train_bpe(self) -> None:
        """A document BPE trained on the training split of every form."""
        if "bpe" in self.cfg["tokenizers"]:
            docs = [(f, d) for f in self.cfg["forms"] for d in self.docs("train")]
            text = lambda fd: self.text(*fd)  # pylint: disable=unnecessary-lambda-assignment
            tk.train(docs, self.cfg["vocab"], self.out / "bpe", progress=False, text=text)

    def encoder(self, tok: str):
        """Tokenizer ``tok``'s encoder: the document BPE, or a :data:`tk.REFERENCES` one."""
        return tk.encoder(str(self.out / "bpe") if tok == "bpe" else tok)

    def data(self, tok: str) -> Path:
        """A tokenizer's encoded corpus directory."""
        return self.out / "data" / tok

    def encode(self) -> None:  # pylint: disable=too-many-locals
        """Per tokenizer, ``vocab.npy`` (its ids any document uses and every byte's, so that any
        text can be spelled; model id i+1 is ``vocab[i]``,
        0 :data:`EOS`), and per form and split ``<form>.<split>.npy``, each document after an EOS
        with one at the end, indexed by ``<form>.<split>.json``."""
        docs = self.docs()
        names = list(docs)
        for tok in self.cfg["tokenizers"]:
            enc, ids = self.encoder(tok), {}
            for form in progress(self.cfg["forms"], desc=f"encode {tok}", unit="form"):
                texts = [self.text(form, d) for d in names]
                ids[form] = (enc.encode(texts), [len(t) for t in texts])
            used = [np.concatenate(v[0]) for v in ids.values()]
            vocab = np.unique(np.concatenate(used + [enc.byte_ids()]))
            out = self.data(tok)
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "vocab.npy", vocab)
            for form, (seqs, chars) in ids.items():
                for which in SPLITS:
                    pick = [i for i, d in enumerate(names) if docs[d]["split"] == which]
                    parts = [np.append(EOS, np.searchsorted(vocab, seqs[i]) + 1) for i in pick]
                    flat = np.concatenate(parts + [np.array([EOS])]).astype(np.uint32)
                    np.save(out / f"{form}.{which}.npy", flat)
                    starts = np.cumsum([0] + [len(p) for p in parts])
                    index = [
                        {"doc": names[i], "start": int(s), "tokens": len(seqs[i])}
                        | {"chars": chars[i], "ticks": docs[names[i]]["ticks"]}
                        for i, s in zip(pick, starts)
                    ]
                    (out / f"{form}.{which}.json").write_text(json.dumps(index), encoding="utf-8")
            log.info("%s: %d token ids used", tok, len(vocab))

    def rundir(self, run: Run) -> Path:
        """A run's directory."""
        return self.out / "runs" / run.name

    def hp(self, run: Run, key: str):
        """A run's setting ``key``: a fine-tune's ``ft_<key>`` where there is one."""
        return self.cfg.get(f"ft_{key}", self.cfg[key]) if run.base else self.cfg[key]

    def eos(self, run: Run) -> int:
        """A run's end of document id: :data:`EOS`, or its pretrained model's."""
        return eos_id(run.base) if run.base else EOS

    def lut(self, run: Run):
        """Each model id's id for a run's model: for a fine-tune its tokenizer's (EOS its
        model's), else None (the same)."""
        if not run.base:
            return None
        vocab = np.load(self.data(run.tok) / "vocab.npy").astype(np.int64)
        return np.append(self.eos(run), vocab)

    def model(self, run: Run):
        """A run's model: the pretrained one, or an untrained Llama decoder of the configured
        shape over its tokenizer's model ids."""
        # pylint: disable=import-outside-toplevel,import-error
        from transformers import LlamaConfig, LlamaForCausalLM

        if run.base:
            return self.pretrained(run.base)
        c = self.cfg
        shape = LlamaConfig(  # pylint: disable=unexpected-keyword-arg
            vocab_size=len(np.load(self.data(run.tok) / "vocab.npy")) + 1,
            hidden_size=c["hidden"],
            intermediate_size=64 * math.ceil(c["hidden"] * 8 / 3 / 64),
            num_hidden_layers=c["layers"],
            num_attention_heads=c["heads"],
            max_position_embeddings=c["ctx"],
            tie_word_embeddings=True,
            bos_token_id=EOS,
            eos_token_id=EOS,
            pad_token_id=EOS,
        )
        return LlamaForCausalLM(shape)

    def pretrained(self, base: str):
        """A pretrained model, with ``ft_lora`` (peft LoRA settings) its adapters on every
        linear layer over its frozen bf16 weights, else all its weights trained."""
        # pylint: disable=import-outside-toplevel,import-error
        import torch
        from transformers import AutoModelForCausalLM

        lora = self.cfg["ft_lora"]
        dtype = torch.bfloat16 if lora else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
        if not lora:
            return model
        from peft import LoraConfig, get_peft_model

        shape = LoraConfig(target_modules="all-linear", task_type="CAUSAL_LM", **lora)
        return get_peft_model(model, shape)

    def windows(self, run: Run, data, rng, n: int, starts=None):  # pylint: disable=R0914
        """``(x, y)``: ``n`` windows of ``ctx`` tokens of ``data`` (EOS padded) in the run's model's
        ids at random offsets (a ``doc_starts`` share at random documents' ``starts`` if given),
        and each position's next-token target (-100 past the data)."""
        import torch  # pylint: disable=import-outside-toplevel,import-error

        ctx, lut = self.hp(run, "ctx"), self.lut(run)
        out, y = np.full((n, ctx), EOS, np.int64), np.full((n, ctx), -100, np.int64)
        at = rng.integers(0, max(1, len(data) - ctx), n)
        if starts is not None and len(starts):
            k = round(n * self.cfg["doc_starts"])
            at[:k] = rng.choice(starts, k)
        for i, a in enumerate(at):
            w = data[a : a + ctx]
            out[i, : len(w)] = w
            y[i, : len(w) - 1] = out[i, 1 : len(w)]
        if lut is not None:
            out, y = lut[out], np.where(y < 0, y, lut[np.maximum(y, 0)])
        return torch.from_numpy(out), torch.from_numpy(y)

    def train(self, run: Run) -> None:  # pylint: disable=too-many-locals,too-many-statements
        """A run's model trained ``steps`` batches of random windows (``micro`` a pass; AdamW,
        warmup, cosine decay), validated every ``eval_every`` steps on fixed windows and stopped
        after ``patience`` (if any) without a better: ``best.pt`` its trained weights at the best,
        ``last.pt`` to resume from, ``log.jsonl`` and ``train.json`` its record."""
        import torch  # pylint: disable=import-outside-toplevel,import-error

        c, rd, steps = self.cfg, self.rundir(run), self.hp(run, "steps")
        if (rd / "train.json").exists():
            return
        rd.mkdir(parents=True, exist_ok=True)
        dev, amp = device()
        torch.manual_seed(c["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        model = self.model(run).to(dev)
        if run.base:
            model.gradient_checkpointing_enable({"use_reentrant": False})
            model.enable_input_require_grads()
        weights = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(weights, lr=self.hp(run, "lr"), betas=(0.9, 0.95), weight_decay=0.1)

        def rate(s):
            decay = 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
            return min(1.0, (s + 1) / self.hp(run, "warmup")) * (0.1 + 0.9 * decay)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, rate)
        state = {"step": 0, "best": math.inf, "seconds": 0.0, "since": 0}
        if (rd / "last.pt").exists():
            ck = torch.load(rd / "last.pt", map_location=dev)
            model.load_state_dict(ck["model"], strict=False)
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            state = ck["state"]
            log.info("%s resumed at step %d", rd.name, state["step"])
        train = np.load(self.data(run.tok) / f"{run.form}.train.npy", mmap_mode="r")
        val = np.load(self.data(run.tok) / f"{run.form}.val.npy", mmap_mode="r")
        val_x = self.windows(run, val, np.random.default_rng(c["seed"]), c["val_windows"])
        index = json.loads((self.data(run.tok) / f"{run.form}.train.json").read_text("utf-8"))
        starts, batch = np.array([d["start"] for d in index], np.int64), self.hp(run, "batch")
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in weights)
        log.info(
            "%s: %d parameters (%d trained), %d training tokens",
            rd.name,
            params,
            trainable,
            len(train),
        )
        pbar = progress(
            range(state["step"], steps), initial=state["step"], total=steps, desc=rd.name
        )
        with (rd / "log.jsonl").open("a", encoding="utf-8") as fh:
            for step in pbar:
                start = time.time()
                model.train()
                rng = np.random.default_rng([c["seed"], step])
                x, y = self.windows(run, train, rng, batch, starts)
                count = int((y >= 0).sum())
                opt.zero_grad(set_to_none=True)
                loss = 0.0
                micro = self.hp(run, "micro")
                for b, t in zip(x.to(dev).split(micro), y.to(dev).split(micro)):
                    with amp:
                        part = nll(model, b, t).sum() / count
                    part.backward()
                    loss += part.detach()
                torch.nn.utils.clip_grad_norm_(weights, 1.0)
                opt.step()
                sched.step()
                state["seconds"] += time.time() - start
                state["step"] = step + 1
                pbar.set_postfix(loss=f"{loss.item():.3f}", refresh=False)
                if state["step"] % self.hp(run, "eval_every") and state["step"] != steps:
                    continue
                rec = {"step": state["step"], "loss": loss.item()}
                rec["val"] = self.loss(run, model, val_x)
                fh.write(json.dumps(rec | {"seconds": state["seconds"]}) + "\n")
                fh.flush()
                log.info("%s step %d loss %.4f val %.4f", rd.name, *list(rec.values()))
                state["since"] += 1
                if rec["val"] < state["best"]:
                    state["best"], state["since"] = rec["val"], 0
                    torch.save(trained_state(model), rd / "best.pt")
                ck = {"model": trained_state(model), "opt": opt.state_dict()}
                torch.save(ck | {"sched": sched.state_dict(), "state": state}, rd / "last.pt")
                if 0 < self.hp(run, "patience") <= state["since"]:
                    log.info("%s stopped at step %d", rd.name, state["step"])
                    break
        seen = state["step"] * batch * self.hp(run, "ctx")
        rec = {"params": params, "trained": trainable, "train_tokens": len(train)}
        rec |= {"steps_run": state["step"], "seen_tokens": seen}
        rec |= {"best_val": state["best"], "seconds": state["seconds"]}
        rec["tokens_per_second"] = seen / max(state["seconds"], 1e-9)
        (rd / "train.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        if run.base:
            (rd / "last.pt").unlink()

    def loss(self, run: Run, model, windows: tuple) -> float:
        """Mean next-token loss (nats) over :meth:`windows`' targets."""
        import torch  # pylint: disable=import-outside-toplevel,import-error

        dev, amp = device()
        model.eval()
        (x, y), total, step = windows, 0.0, self.hp(run, "batch_eval")
        with torch.no_grad(), amp:
            for b, t in zip(x.split(step), y.split(step)):
                total += nll(model, b.to(dev), t.to(dev)).sum().item()
        return total / int((y >= 0).sum())

    def trained(self, run: Run):
        """A run's best model, for inference."""
        import torch  # pylint: disable=import-outside-toplevel,import-error

        model = self.model(run)
        best = torch.load(self.rundir(run) / "best.pt", map_location="cpu")
        model.load_state_dict(best, strict=not run.base)
        return model.to(device()[0]).eval()

    def chunks(self, n: int, ctx: int) -> list:
        """The windows scoring a document of ``n`` tokens, ``(start, end, first)``, half a
        context apart, ``first`` the first position each scores: every position after the
        first scored once, with up to a context of history."""
        out, first, begin = [], 1, 0
        while True:
            end = min(begin + ctx, n)
            out.append((begin, end, first))
            if end == n:
                return out
            first, begin = end, begin + ctx // 2

    def evaluate(self, run: Run) -> None:  # pylint: disable=too-many-locals
        """``eval.json``: the best model's bits on each test document (every token after its
        leading EOS, the trailing EOS included) and the split's bits per song tick, character
        and token."""
        import torch  # pylint: disable=import-outside-toplevel,import-error

        rd = self.rundir(run)
        if (rd / "eval.json").exists():
            return
        (dev, amp), model, ctx = device(), self.trained(run), self.hp(run, "ctx")
        data = np.load(self.data(run.tok) / f"{run.form}.test.npy", mmap_mode="r")
        index = json.loads((self.data(run.tok) / f"{run.form}.test.json").read_text("utf-8"))
        lut = self.lut(run)
        data = data if lut is None else lut[data]
        jobs = []
        for k, d in enumerate(index):
            toks = data[d["start"] : d["start"] + d["tokens"] + 2]
            jobs += [(k, toks[b:e], f - b) for b, e, f in self.chunks(len(toks), ctx)]
        bits, step = np.zeros(len(index)), self.hp(run, "batch_eval")
        for at in progress(range(0, len(jobs), step), desc=f"eval {rd.name}", unit="batch"):
            batch = jobs[at : at + step]
            x = np.full((len(batch), ctx), self.eos(run), np.int64)
            y = np.full((len(batch), ctx), -100, np.int64)
            for i, (_, w, first) in enumerate(batch):
                x[i, : len(w)] = w
                y[i, first - 1 : len(w) - 1] = w[first:]
            with torch.no_grad(), amp:
                nats = nll(model, torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev))
            np.add.at(bits, [k for k, _, _ in batch], nats.cpu().numpy() / LN2)
        ticks = np.array([d["ticks"] for d in index])
        rec = {
            "docs": len(index),
            "bits_per_tick": bits.sum() / ticks.sum(),
            "bits_per_tick_median": float(np.median(bits / ticks)),
            "bits_per_char": bits.sum() / sum(d["chars"] for d in index),
            "bits_per_token": bits.sum() / sum(d["tokens"] + 1 for d in index),
            "tokens_per_doc": float(np.mean([d["tokens"] for d in index])),
            "per_doc": {d["doc"]: b for d, b in zip(index, bits.tolist())},
        }
        (rd / "eval.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        log.info("%s: %.4f bits/tick", rd.name, rec["bits_per_tick"])

    def generate(self, run: Run, model, prompts: np.ndarray, stuck: set | None = None):
        """Each prompt (a row of model ids) continued by sampling up to ``sample_tokens`` or an
        EOS (the rest EOS), half a context at a time after the last half a context; given
        ``stuck``, each kept to the document grammar (:mod:`constrain`), the rows it left no
        model id added to ``stuck``."""
        # pylint: disable=too-many-locals
        import torch  # pylint: disable=import-outside-toplevel,import-error

        ctx, limit, eos = self.hp(run, "ctx"), self.cfg["sample_tokens"], self.eos(run)
        seq = torch.from_numpy(np.asarray(prompts, np.int64)).to(device()[0])
        n, p = seq.shape
        torch.manual_seed(self.cfg["seed"])
        if stuck is not None:
            import constrain  # pylint: disable=import-outside-toplevel

            con, cols = self.constraint(run, model)
            native = prompts[:, 1:] if cols is None else cols[prompts[:, 1:]]
            matchers = [con.matcher(row) for row in native]
        while seq.shape[1] < p + limit and not bool((seq[:, p:] == eos).any(1).all()):
            tail = seq[:, -(ctx // 2) :]
            new = min(ctx - tail.shape[1], p + limit - seq.shape[1])
            extra = {}
            if stuck is not None:
                masked = constrain.Masked(con, matchers, cols, eos, seq.shape[1] == p)
                extra["logits_processor"] = [masked]
            out = model.generate(
                tail,
                attention_mask=torch.ones_like(tail),
                max_new_tokens=new,
                do_sample=True,
                top_k=0,
                eos_token_id=eos,
                pad_token_id=eos,
                **extra,
            )[:, tail.shape[1] :]
            if stuck is not None:
                stuck |= masked.stuck
            ended = (seq[:, p:] == eos).any(1, keepdim=True)
            pad = torch.full((n, new - out.shape[1]), eos, dtype=torch.long, device=seq.device)
            seq = torch.cat([seq, torch.cat([out, pad], 1).masked_fill(ended, eos)], 1)
        return seq.cpu().numpy()

    def constraint(self, run: Run, model) -> tuple:
        """``(constraint, cols)``: the document grammar over the run's tokenizer (its EOS the
        pretrained model's, else one past its vocabulary), and each model id's native id (None
        where they are the same)."""
        from constrain import Constraint  # pylint: disable=import-outside-toplevel

        tok = str(self.out / "bpe") if run.tok == "bpe" else run.tok
        if not run.base:
            con = Constraint(tok)
            vocab = np.load(self.data(run.tok) / "vocab.npy").astype(np.int64)
            return con, np.append(con.eos, vocab)
        n_vocab = model.get_output_embeddings().weight.shape[0]
        return Constraint(tok, eos=self.eos(run), n_vocab=n_vocab), None

    def prompts(self, run: Run) -> dict:
        """``{kind: prompts}`` in the run's model's ids, each after an EOS: ``samples`` of the
        header (as the test documents start), and the openings (``prompt_tokens``) of the
        first ``samples`` test documents at least twice that long."""
        n, p, lut = self.cfg["samples"], self.cfg["prompt_tokens"], self.lut(run)
        data = np.load(self.data(run.tok) / f"{run.form}.test.npy", mmap_mode="r")
        index = json.loads((self.data(run.tok) / f"{run.form}.test.json").read_text("utf-8"))
        k = len(self.encoder(run.tok).encode([HEADER])[0]) + 1
        opens = [data[d["start"] : d["start"] + p + 1] for d in index if d["tokens"] >= 2 * p]
        out = {"header": np.tile(data[:k], (n, 1)), "continue": np.array(opens[:n], np.int64)}
        return {kind: v if lut is None else lut[v] for kind, v in out.items() if len(v)}

    def sample(self, run: Run) -> None:  # pylint: disable=too-many-locals
        """``samples/<kind>-<i>.yaml``, the best model's continuations of :meth:`prompts`, and
        ``sample.json``: per kind, the share that end, parse and play ``play_ticks`` ticks,
        ``valid`` (the share of generated characters before the first parse error) and where
        the grammar is over the tokenizer, ``prefix``: the share of generated tokens a document
        could still be completed after, truncation no error."""
        rd = self.rundir(run)
        if (rd / "sample.json").exists():
            return
        vocab, enc = np.load(self.data(run.tok) / "vocab.npy"), self.encoder(run.tok)
        eos, model, out = self.eos(run), self.trained(run), {}
        (rd / "samples").mkdir(exist_ok=True)

        def text(ids):
            return enc.decode(ids if run.base else vocab[np.asarray(ids) - 1])

        kinds = [(k, v, False) for k, v in self.prompts(run).items()]
        con, cols = (None, None)
        if run.tok == "bpe" or tk.REFERENCES.get(run.tok):
            con, cols = self.constraint(run, model)
        if self.cfg["constrain"] and con is not None:
            kinds += [(f"{k}_constrained", v, True) for k, v, _ in kinds]
        native = (lambda x: x) if cols is None else (lambda x: cols[x])
        for kind, prompts, constrained in kinds:
            recs, p, stuck = [], prompts.shape[1], set() if constrained else None
            for i, ids in enumerate(self.generate(run, model, prompts, stuck)):
                gen = ids[p:]
                ended = bool((gen == eos).any()) and i not in (stuck or ())
                gen = gen[: np.argmax(gen == eos)] if ended else gen
                head, doc = text(ids[1:p]), text(np.concatenate([ids[1:p], gen]))
                (rd / "samples" / f"{kind}-{i}.yaml").write_text(doc, encoding="utf-8")
                rec = {"tokens": len(gen), "ended": ended} | self.check(doc, len(head))
                rec |= {"stuck": True} if i in (stuck or ()) else {}
                if con is not None:
                    taken = con.matcher(native(ids[1:p])).valid(native(gen))
                    rec["prefix"] = taken / len(gen) if len(gen) else 1.0
                recs.append(rec)
            keys = ("ended", "parses", "plays", "valid") + ("prefix",) * (con is not None)
            out |= {f"{kind}_{k}": float(np.mean([r[k] for r in recs])) for k in keys}
            out[f"{kind}_where"] = dict(Counter(r["where"] for r in recs if "where" in r))
            out[kind] = recs
        (rd / "sample.json").write_text(json.dumps(out, indent=1), "utf-8")
        log.info("%s: %s", rd.name, {k: v for k, v in out.items() if k not in SAMPLES})

    def check(self, doc: str, start: int) -> dict:
        """Whether a sampled document parses and plays ``play_ticks`` ticks, its first error,
        and ``valid``: the share of its characters from ``start`` before its first parse error
        (all of them when it parses)."""
        # pylint: disable=import-outside-toplevel
        from pysidtracker.jobs import deadline
        from unitracker_grammar.document import read
        from unitracker_player.player import play

        rec = {"parses": False, "plays": False, "valid": 1.0}
        try:
            read(doc)
            rec["parses"] = True
            with deadline(60):
                play(doc, ticks=self.cfg["play_ticks"])
            rec["plays"] = True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            msg = (str(exc).splitlines() or [""])[0]
            rec["error"] = f"{type(exc).__name__}: {msg}"
            at = re.match(r"line (\d+) column (\d+)", msg) if not rec["parses"] else None
            if at:
                line, col = int(at[1]), int(at[2])
                pos = sum(len(ln) + 1 for ln in doc.split("\n")[: line - 1]) + col - 1
                rec["valid"] = min(1.0, max(0.0, (pos - start) / max(1, len(doc) - start)))
                rec["where"] = where(doc, pos)
        return rec

    def finetune(self) -> None:
        """``finetune.json``, the ``ft_forms`` forms (a list, or that many whose from-scratch runs
        score the fewest bits per tick), then each one's fine-tune of ``ft_model`` trained,
        evaluated and sampled."""
        if self.cfg["ft_tokenizer"] not in self.cfg["tokenizers"]:
            log.info("no fine-tune: %s is not encoded", self.cfg["ft_tokenizer"])
            return
        best = {}
        for row in self.results():
            if "bits_per_tick" in row:
                best[row["form"]] = min(best.get(row["form"], math.inf), row["bits_per_tick"])
        forms = self.cfg["ft_forms"]
        if isinstance(forms, int):
            forms = sorted(best, key=best.get)[:forms]
        (self.out / "finetune.json").write_text(json.dumps(forms), encoding="utf-8")
        log.info("fine-tuning %s on %s", self.cfg["ft_model"], forms)
        for run in self.runs(scratch=False):
            self.train(run)
            self.evaluate(run)
            self.sample(run)

    def probe(self) -> None:
        """The analyses of :mod:`probe`: ``baselines.json`` and per run ``probe.json``."""
        import probe  # pylint: disable=import-outside-toplevel

        probe.analyse(self)

    def verified(self, per_doc: dict) -> dict:
        """Bits per song tick over the test documents of ``per_doc`` (``{doc: bits}``) that play
        their native player's grid, and their count."""
        docs = self.docs("test")
        ok = [d for d in per_doc if self.meta(d).get("verified")]
        ticks = sum(docs[d]["ticks"] for d in ok)
        bits = {"bits_per_tick_verified": sum(per_doc[d] for d in ok) / ticks} if ok else {}
        return {"verified_docs": len(ok)} | bits

    def results(self) -> list:
        """Each run's row: its form, tokenizer, base model, its stages' records' scalars and
        its bits per tick over the verified test documents."""
        rows = []
        for run in self.runs() + self.runs(scratch=False):
            row, rd = {"form": run.form, "tokenizer": run.tok}, self.rundir(run)
            row["model"] = run.base or "scratch"
            for name in ("train", "eval", "sample"):
                if (rd / f"{name}.json").exists():
                    rec = json.loads((rd / f"{name}.json").read_text(encoding="utf-8"))
                    row |= {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
                    row |= self.verified(rec["per_doc"]) if "per_doc" in rec else {}
            rows.append(row)
        return rows

    def report(self) -> str:
        """``results.json`` and ``results.md``, a table of the runs; the table."""
        rows = self.results()
        (self.out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        cols = [
            ("form", "form", "{}"),
            ("tokenizer", "tokenizer", "{}"),
            ("model", "model", "{}"),
            ("params", "params", "{:,}"),
            ("tokens_per_doc", "tokens/doc", "{:.0f}"),
            ("best_val", "val loss", "{:.4f}"),
            ("bits_per_tick", "bits/tick", "{:.4f}"),
            ("bits_per_tick_median", "median", "{:.4f}"),
            ("bits_per_tick_verified", "verified bits/tick", "{:.4f}"),
            ("verified_docs", "verified docs", "{}"),
            ("bits_per_char", "bits/char", "{:.4f}"),
            ("header_prefix", "header prefix", "{:.0%}"),
            ("header_parses", "header parses", "{:.0%}"),
            ("continue_prefix", "continued prefix", "{:.0%}"),
            ("continue_parses", "continued parses", "{:.0%}"),
            ("continue_plays", "continued plays", "{:.0%}"),
            ("header_constrained_parses", "constrained header parses", "{:.0%}"),
            ("header_constrained_plays", "constrained header plays", "{:.0%}"),
            ("continue_constrained_parses", "constrained continued parses", "{:.0%}"),
            ("continue_constrained_plays", "constrained continued plays", "{:.0%}"),
        ]
        lines = [
            "| " + " | ".join(c[1] for c in cols) + " |",
            "|---|---|---|" + "---:|" * (len(cols) - 3),
        ]
        for r in rows:
            lines.append("| " + " | ".join(f.format(r[k]) if k in r else "" for k, _, f in cols))
            lines[-1] += " |"
        if (self.out / "baselines.json").exists():
            base = json.loads((self.out / "baselines.json").read_text(encoding="utf-8"))
            names = list(next(iter(base.values())))
            lines += ["", "| form | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
            lines += [
                f"| {f} | " + " | ".join(f"{b[k]:.3f}" for k in names) + " |"
                for f, b in base.items()
            ]
        text = "\n".join(lines) + "\n"
        (self.out / "results.md").write_text(text, encoding="utf-8")
        log.info("results\n%s", text)
        return text


def status(out: Path, tail: int = 5) -> str:
    """An experiment's progress: stages done, corpus forms written, each run's last record and
    the log's last lines."""
    if not (out / "config.json").exists():
        return f"{out}: not started"
    exp = Experiment(json.loads((out / "config.json").read_text(encoding="utf-8")), out)
    lines = [f"{out}:"]
    for name, _ in exp.stages():
        mark = out / "stages" / name
        lines.append(
            f"  {name:10} " + (f"done {mark.read_text().strip()}s" if mark.exists() else "-")
        )
    if (out / "tunes.csv").exists():
        names = [shared.name(r) for r in exp.songs()]
        for form in exp.cfg["forms"]:
            have = sum(exp.path(form, d).exists() for d in names)
            err = sum(exp.path(form, d).with_suffix(".err").exists() for d in names)
            lines.append(f"  corpus {form:10} {have}/{len(names)} written, {err} failed or running")
    for run in exp.runs() + exp.runs(scratch=False):
        rd = exp.rundir(run)
        recs = (rd / "log.jsonl").read_text().splitlines() if (rd / "log.jsonl").exists() else []
        lines.append(f"  run {rd.name:24} {recs[-1] if recs else '-'}")
    if (out / "learn.log").exists():
        lines += ["  " + ln[-200:] for ln in (out / "learn.log").read_text().splitlines()[-tail:]]
    return "\n".join(lines)


def launch(argv: list, out: Path, hvsc: Path | None, tunes: Path | None = None) -> list:
    """The commands that build the image and run ``learn.py run argv`` detached in it on the
    host's GPUs as this user, named for ``out``, appending to ``out/learn.log`` (``out``'s
    parent, the corpus's, and the tune list's directory mounted)."""
    root, name = str(shared.ROOT), f"{IMAGE}-{out.name}"
    build = ["docker", "build", "-q", "-t", IMAGE, "-f", f"{root}/docker/learn/Dockerfile", root]
    env = ["HOME=/tmp", f"HF_HOME={root}/.cache/hf", "PYTHONPATH=src:tools", "PYTHONUNBUFFERED=1"]
    env += ["PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
    env += [f"UNITRACKER_COMMIT={corpus.versions()['unitracker'] or ''}"]
    env += [f"{v}=2" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")]
    cmd = shlex.join(["python", "tools/learn.py", "run", *argv])
    run = ["docker", "run", "-d", "--name", name, "--device", "nvidia.com/gpu=all", "--ipc=host"]
    run += ["--user", f"{os.getuid()}:{os.getgid()}", "-w", root]
    run += [x for e in env for x in ("-e", e)]
    mounts = [root, str(out.parent)] + [str(p) for p in (hvsc, tunes and tunes.parent) if p]
    run += [x for m in mounts for x in ("-v", f"{m}:{m}")]
    run += [IMAGE, "sh", "-c", f"{cmd} >> {shlex.quote(str(out / 'learn.log'))} 2>&1"]
    return [build, ["docker", "rm", "-f", name], run]


def config(preset: str, sets: list) -> dict:
    """A preset's configuration with ``KEY=VALUE`` overrides (JSON values, else strings)."""
    cfg = dict(PRESETS[preset])
    for item in sets:
        key, value = item.split("=", 1)
        if key not in cfg:
            raise SystemExit(f"{key}: not a configuration key ({', '.join(cfg)})")
        try:
            cfg[key] = json.loads(value)
        except json.JSONDecodeError:
            cfg[key] = value
    return cfg


def main(argv=None) -> int:
    """Parse the command line and run the command."""
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("command", choices=("run", "status", "launch"))
    ap.add_argument("--preset", choices=list(PRESETS), default="smoke")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", type=Path, help="the experiment's directory (out/learn/PRESET)")
    ap.add_argument("--hvsc", type=Path, default=os.environ.get("HVSC"))
    ap.add_argument("--tunes", type=Path, help="a tune list (default: HVSC surveyed)")
    ap.add_argument("--redo", nargs="*", default=[], help="stages to run again")
    args = ap.parse_args(argv)
    out = (args.out or OUT / args.preset).resolve()
    if args.command == "status":
        print(status(out))
        return 0
    if args.command == "launch":
        out.mkdir(parents=True, exist_ok=True)
        inner = [a for a in argv if a != "launch"] + ["--out", str(out)]
        for cmd in launch(inner, out, args.hvsc, args.tunes and args.tunes.resolve()):
            subprocess.run(cmd, check=cmd[1] != "rm", capture_output=cmd[1] == "rm")
        print(f"python tools/learn.py status --out {out}")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    Experiment(config(args.preset, args.set), out, args.hvsc, args.tunes).run(args.redo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
