# Learnability

How well language models learn unitracker documents in their intended form, lossy and inline
(`lossy-single`, `lossy-all`, beside ablations of it): which form and tokenizer a model learns
most from, and whether a pretrained LLM (Qwen2.5) learns to write tunes that play. The lossless forms are left out: their
code modulators are the native players' 6502 routines, the same boilerplate in every tune of a
player.

## Questions

| question | runs | measure |
|---|---|---|
| which form and tokenizer is most learnable, without prior knowledge | a decoder trained from scratch per form and tokenizer, the same shape, steps and tokens per step | held-out bits per song tick |
| what in the representation helps | ablations of the intended form from scratch: definitions named in sections (`lossy`), at first use (`lossy-single`), all inline (`lossy-all`), and all inline without phrase templates (`lossy-nophrases-all`, the optimiser's passes but `phrases`) | the same, and where samples' parse errors fall |
| does a pretrained LLM learn to write unitracker tunes, and how much does its prior help | Qwen2.5 fine-tuned (LoRA) on the intended form | | the same, beside the from-scratch runs on the same form |
| are the documents a model writes tunes | every run's samples, from the document header and continuing held-out tunes' openings | the share that end, parse, and play on unitracker-player, and the share of generated characters before the first parse error |

Bits per song tick (a document's total next-token surprisal over its song length,
`tools/corpus.py` `song_ticks`) compares across forms and tokenizers, where loss per token does
not: a form or tokenizer that spends fewer tokens on a tune is not penalised for each token
carrying more. The lossy forms' documents play an approximation of the native grid, within the
lossy mode's tolerances; their bits are of that tune.

## Method

`python tools/learn.py run` runs the stages in order, each resuming from what an earlier run wrote
(a stage done is marked in `stages/`; `--redo STAGE ...` runs one again):

| stage | does | writes (under `out/learn/<preset>/`) |
|---|---|---|
| `tunes` | the HVSC tunes SIDId names GoatTracker 2, SID-Wizard or defMON (`pysidtracker.survey find`, once into `out/learn/tunes.csv`), all or a seeded sample | `tunes.csv` |
| `corpus` | each tune's forms (`tools/corpus.py forms`), cached per tune in `out/learn/corpus/`, shared by the experiments | `../corpus/<form>/<tune>.yaml` |
| `split` | the tunes every form has, split by composer (the HVSC musician directory; a tune elsewhere is its own group): groups in SHA-1 order fill test (10% of tunes), then validation (5%), the rest train, so no composer's style crosses splits | `split.json` |
| `tokenizers` | the document BPE (`unitracker.optimiser.tokenizer`) trained on the training split of every form | `bpe/` |
| `encode` | each form and split's documents per tokenizer, the model's vocabulary the ids they use and every byte's (so any text the grammar allows can be spelled) (the models' documents throughout have their meta line (titles and authors, no part of the tune) replaced by `meta:{title:-}`; `--set meta=true` keeps it), each after an end-of-document token, the ids remapped to those the corpus uses | `data/<tokenizer>/` |
| `train` | per form and tokenizer, a Llama-shaped decoder from scratch: training windows at random offsets, a quarter of them at document starts (`doc_starts`), AdamW with warmup and cosine decay, stopped after `patience` validations without a better, the best checkpoint by validation loss; the loss is applied to the output head a chunk of positions at a time, so no window's logits over a large vocabulary are held at once | `runs/<form>.<tokenizer>/` |
| `eval` | each test document's bits under the best checkpoint, every token scored once with up to a context of history (windows half a context apart) | `eval.json` |
| `sample` | documents sampled (temperature 1, a context window sliding half a context at a time) after two kinds of prompt: `header`, the document header `unitracker:2.3`; `continue`, the first `prompt_tokens` of a test tune. Each kind is also sampled grammar-constrained (`*_constrained`, `tools/constrain.py`: llguidance masks each step's logits to the tokens the document grammar allows). Each sample is parsed and played; `valid` is the share of its generated characters before the first parse error and `where` the construct (call, list, record, top level) open there; `prefix` the share of its generated tokens after which a document could still be completed (so a sample cut off at `sample_tokens` is not an error), 100% for a constrained one by construction | `samples/<kind>-<i>.yaml`, `sample.json` |
| `finetune` | `ft_model` fine-tuned on the `ft_forms` forms (named, or a number: that many with the fewest from-scratch bits per tick) in its own tokenizer's ids, its weights all trained or (`ft_lora`, peft settings) LoRA adapters on every linear layer over its frozen bf16 weights, then evaluated and sampled as above | `finetune.json`, `runs/<form>.<tokenizer>.<model>/` |
| `probe` | the test split's bits per tick under models that know no structure (each tokenizer's training unigram, xz, zstd with up to 8 MiB of training text as its dictionary), and per run the best model's bits per token by grammar category beside the unigram's, and the bits per token of a context's last quarter with each length of history | `baselines.json`, `runs/*/probe.json` |
| `report` | a table of every run and the baselines | `results.md`, `results.json` |

Presets (`--preset`, each key overridable with `--set KEY=VALUE`, JSON values):

| preset | tunes | forms | from scratch | fine-tune |
|---|---|---|---|---|
| `smoke` | 8 | `lossy-single`, `lossy-all` | 2 layers of 64, context 256, 40 steps | the best form, 4 steps |
| `full` | all | `lossy`, `lossy-single`, `lossy-all`, `lossy-nophrases-all` | 8 layers of 512 (26M parameters), context 4096, at most 5000 steps of 16 windows, patience 4 validations (1000 steps) | Qwen2.5-1.5B on `lossy-all`, LoRA rank 64 on every linear layer, context 8192, at most 2000 steps of 8 windows, learning rate 2e-4, patience 4 |

Both tokenizers are the document BPE (`vocab`) and Qwen2.5's (`qwen2.5`), whose ids the fine-tune
reads. Both kinds of run see the same tokens per step (65536).

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
