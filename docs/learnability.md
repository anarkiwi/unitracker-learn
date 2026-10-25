# Learnability

How well language models learn unitracker documents: which document form and tokenizer a model
learns most from, and whether a pretrained LLM (Qwen2.5) learns to write tunes that play.

## Questions

| question | runs | measure |
|---|---|---|
| which form and tokenizer is most learnable, without prior knowledge | a decoder trained from scratch per form and tokenizer, the same shape, steps and tokens per step | held-out bits per song tick |
| does a pretrained LLM learn to write unitracker tunes, and how much does its prior help | Qwen2.5-0.5B fine-tuned on the forms the from-scratch runs score best | the same, beside the from-scratch runs on the same form |
| are the documents a model writes tunes | every run's samples | the share that end, parse, and play on unitracker-player |

Bits per song tick (a document's total next-token surprisal over its song length,
`tools/corpus.py` `song_ticks`) compares across forms and tokenizers, where loss per token does
not: a form or tokenizer that spends fewer tokens on a tune is not penalised for each token
carrying more. The lossy form's documents play a different grid; their bits are of a different
(approximated) tune.

## Method

`python tools/learn.py run` runs the stages in order, each resuming from what an earlier run wrote
(a stage done is marked in `stages/`; `--redo STAGE ...` runs one again):

| stage | does | writes (under `out/learn/<preset>/`) |
|---|---|---|
| `tunes` | the HVSC tunes SIDId names GoatTracker 2, SID-Wizard or defMON (`pysidtracker.survey find`, once into `out/learn/tunes.csv`), all or a seeded sample | `tunes.csv` |
| `corpus` | each tune's forms (`tools/corpus.py forms`), cached per tune in `out/learn/corpus/`, shared by the experiments | `../corpus/<form>/<tune>.yaml` |
| `split` | the tunes every form has, split by composer (the HVSC musician directory; a tune elsewhere is its own group): groups in SHA-1 order fill test (10% of tunes), then validation (5%), the rest train, so no composer's style crosses splits | `split.json` |
| `tokenizers` | the document BPE (`unitracker.optimiser.tokenizer`) trained on the training split of every form | `bpe/` |
| `encode` | each form and split's documents per tokenizer, each after an end-of-document token, the ids remapped to those the corpus uses | `data/<tokenizer>/` |
| `train` | per form and tokenizer, a Llama-shaped decoder from scratch: random training windows, AdamW with warmup and cosine decay, the best checkpoint by validation loss | `runs/<form>.<tokenizer>/` |
| `eval` | each test document's bits under the best checkpoint, every token scored once with up to a context of history (windows half a context apart) | `eval.json` |
| `sample` | documents sampled unconditionally (temperature 1, a context window sliding half a context at a time), parsed and played | `samples/`, `sample.json` |
| `finetune` | `ft_model` fine-tuned on the `ft_forms` forms with the fewest from-scratch bits per tick (its own tokenizer's ids), then evaluated and sampled as above | `finetune.json`, `runs/<form>.<tokenizer>.<model>/` |
| `report` | a table of every run | `results.md`, `results.json` |

Presets (`--preset`, each key overridable with `--set KEY=VALUE`, JSON values):

| preset | tunes | forms | from scratch | fine-tune |
|---|---|---|---|---|
| `smoke` | 8 | `lifted`, `named` | 2 layers of 64, context 256, 40 steps | the best form, 4 steps |
| `full` | all | `lifted`, `named`, `all`, `lossy-all` | 8 layers of 512 (26M parameters), context 2048, 5000 steps of 32 windows | Qwen2.5-0.5B on the best 2 forms, 2000 steps of 32 windows, learning rate 5e-5 |

Both tokenizers are the document BPE (`vocab`) and Qwen2.5's (`qwen2.5`), whose ids the fine-tune
reads.

## Running

The stages run in a Docker image (`docker/learn/Dockerfile`: the PyTorch CUDA image with this
project's dependencies; the repository is mounted, not copied) on the host's GPUs:

```sh
python tools/learn.py launch --preset smoke --hvsc $HVSC        # minutes: every stage end to end
python tools/learn.py launch --preset full --hvsc $HVSC         # detached; hours
python tools/learn.py status --preset full                      # stages, corpus, runs, log tail
```

`launch` builds the image and starts the container `unitracker-learn-<out>` detached, its output
appended to `out/learn/<preset>/learn.log`; launched again it resumes. `--out DIR` puts an
experiment elsewhere (its corpus beside it, in `DIR/../corpus`).

## Results

Pending the `full` run.
