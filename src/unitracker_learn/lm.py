"""What the learnability experiment's stages and analyses share: the models' document text,
progress, the device, and a causal LM's loss."""

import math
import re
import sys

from tqdm import tqdm

LN2 = math.log(2)
META = "meta:{title:-}"
META_LINE = re.compile(r"(?m)^meta:\{.*\}$")


def progress(it, **kw):
    """``tqdm``, a line a minute when its output is not a terminal (a log file)."""
    tty = sys.stderr.isatty()
    return tqdm(it, mininterval=0.1 if tty else 60, ascii=not tty, **kw)


def device():
    """``(device, autocast)``: the GPU with bf16 autocast when there is one, else the CPU."""
    import torch  # pylint: disable=import-outside-toplevel,import-error

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dev, torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda")


def nll(model, x, y, chunk: int = 1024):
    """Each window's summed loss (nats) of its targets ``y`` (-100 ignored) under a causal LM,
    its output head applied ``chunk`` positions at a time (recomputed in the backward pass), so
    no window's logits over the vocabulary are held at once."""
    # pylint: disable=import-outside-toplevel,import-error
    import torch
    from torch.nn.functional import cross_entropy
    from torch.utils.checkpoint import checkpoint

    hidden, head = model.get_decoder()(input_ids=x).last_hidden_state, model.get_output_embeddings()

    def part(h, t):
        logits = head(h).float()
        flat = cross_entropy(logits.flatten(0, 1), t.flatten(), reduction="none")
        return flat.view(t.shape).sum(1)

    total = 0
    for at in range(0, x.shape[1], chunk):
        args = (hidden[:, at : at + chunk], y[:, at : at + chunk])
        grad = torch.is_grad_enabled()
        total = total + (checkpoint(part, *args, use_reentrant=False) if grad else part(*args))
    return total
