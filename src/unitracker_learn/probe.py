"""The learnability experiment's analyses (docs/learnability.md, the probe stage): the test
split's bits per tick under structure-free baselines, and per run the best model's bits per token
by grammar category and by length of history."""

import json
import logging
import lzma

import numpy as np

from unitracker_learn import tokenizer as tk
from unitracker_learn.lm import LN2, device, nll, progress

log = logging.getLogger("learn")


def analyse(exp) -> None:
    """``baselines.json`` (:func:`baselines`) and per run ``probe.json``
    (:func:`probe_run`)."""
    if not (exp.out / "baselines.json").exists():
        rec = {form: baselines(exp, form) for form in exp.cfg["forms"]}
        (exp.out / "baselines.json").write_text(json.dumps(rec, indent=1), "utf-8")
    for run in exp.runs() + exp.runs(scratch=False):
        probe_run(exp, run)


def baselines(exp, form: str) -> dict:
    """The test split's bits per song tick under models that know no structure: per
    tokenizer its training split's unigram frequencies, xz, and zstd with a size-capped part of the
    training split (seeded order) as a raw content dictionary, within its window."""
    import zstandard  # pylint: disable=import-outside-toplevel

    test = list(exp.docs("test"))
    ticks = sum(v["ticks"] for v in exp.docs("test").values())
    texts = [exp.text(form, d).encode() for d in test]
    train = [exp.text(form, d).encode() for d in exp.docs("train")]
    order = np.random.default_rng(exp.cfg["seed"]).permutation(len(train))
    raw = b"".join(train[i] for i in order)[: 8 << 20]
    zdict = zstandard.ZstdCompressionDict(raw, dict_type=zstandard.DICT_TYPE_RAWCONTENT)
    zc = zstandard.ZstdCompressor(level=19, dict_data=zdict)
    out = {
        "raw": 8 * sum(map(len, texts)) / ticks,
        "xz": 8 * sum(len(lzma.compress(t, preset=9 | lzma.PRESET_EXTREME)) for t in texts),
        "zstd_dict": 8 * sum(len(zc.compress(t)) for t in texts) / ticks,
    }
    out["xz"] /= ticks
    for tok in exp.cfg["tokenizers"]:
        uni = unigram(exp, tok, form)
        data = np.load(exp.data(tok) / f"{form}.test.npy", mmap_mode="r")
        out[f"unigram_{tok}"] = float(uni[data[1:]].sum()) / ticks
    log.info("%s baselines (bits/tick): %s", form, out)
    return out


def unigram(exp, tok: str, form: str) -> np.ndarray:
    """Each model id's surprisal (bits) under the training split's token frequencies
    (add-half smoothed)."""
    data = np.load(exp.data(tok) / f"{form}.train.npy", mmap_mode="r")
    size = len(np.load(exp.data(tok) / "vocab.npy")) + 1
    count = np.bincount(data, minlength=size) + 0.5
    return -np.log2(count / count.sum())


def probe_run(exp, run) -> None:  # pylint: disable=too-many-locals
    """``probe.json``: the best model's bits per token by grammar category (beside the
    unigram's) over the first context of ``probe_docs`` test documents, and over their
    last quarter context, the bits per token with each length of history."""
    import torch  # pylint: disable=import-outside-toplevel,import-error

    rd = exp.rundir(run)
    if (rd / "probe.json").exists() or not (rd / "best.pt").exists():
        return
    model, ctx, (dev, amp) = exp.trained(run), exp.hp(run, "ctx"), device()
    data = np.load(exp.data(run.tok) / f"{run.form}.test.npy", mmap_mode="r")
    index = json.loads((exp.data(run.tok) / f"{run.form}.test.json").read_text("utf-8"))
    index = index[:: max(1, len(index) // exp.cfg["probe_docs"])]
    uni, enc, lut = unigram(exp, run.tok, run.form), exp.encoder(run.tok), exp.lut(run)
    size = len(tk.CATEGORIES)
    cats = {"tokens": np.zeros(size), "bits": np.zeros(size), "unigram": np.zeros(size)}
    start = 3 * ctx // 4
    hist = [h for h in (16, 64, 256, ctx // 2, start) if h <= start]
    context = {h: [0.0, 0] for h in hist}

    for d in progress(index, desc=f"probe {rd.name}", unit="doc"):
        toks = np.asarray(data[d["start"] : d["start"] + d["tokens"] + 1], np.int64)
        x = toks[:ctx]
        x_t = torch.from_numpy(x if lut is None else lut[x]).to(dev)[None]
        y_t = torch.nn.functional.pad(x_t[:, 1:], (0, 1), value=-100)
        with torch.no_grad(), amp:
            hidden = model.get_decoder()(input_ids=x_t).last_hidden_state
            logits = model.get_output_embeddings()(hidden).float()[0, :-1]
        per = torch.nn.functional.cross_entropy(logits, y_t[0, :-1], reduction="none")
        per = per.cpu().numpy() / LN2
        text = exp.text(run.form, d["doc"])
        starts, ends = enc.spans(text)
        cat = tk.attribute(tk.categories(text), starts, ends)[: len(per)]
        np.add.at(cats["tokens"], cat, 1)
        np.add.at(cats["bits"], cat, per[: len(cat)])
        np.add.at(cats["unigram"], cat, uni[x[1 : len(cat) + 1]])
        if len(toks) < ctx:
            continue
        for h in hist:
            w = toks[start - h : ctx]
            y = np.full(len(w), -100, np.int64)
            y[h - 1 : -1] = w[h:]
            xw = w if lut is None else lut[w]
            yw = y if lut is None else np.where(y < 0, y, lut[np.maximum(y, 0)])
            xw, yw = torch.from_numpy(xw).to(dev)[None], torch.from_numpy(yw).to(dev)[None]
            with torch.no_grad(), amp:
                context[h][0] += nll(model, xw, yw).item() / LN2
            context[h][1] += ctx - start
    rec = {
        "categories": {
            c: {k: float(v[i]) for k, v in cats.items()}
            for i, c in enumerate(tk.CATEGORIES)
            if cats["tokens"][i]
        },
        "context": {str(h): b / max(n, 1) for h, (b, n) in context.items()},
    }
    (rd / "probe.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log.info("%s probe: %s", rd.name, rec["context"])
