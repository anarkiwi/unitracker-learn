# unitracker-learn

Learnability studies, tokenization and constrained decoding for
[unitracker](https://github.com/anarkiwi/unitracker) documents: how well language models learn the
documents per form and tokenizer, a document BPE, and grammar-constrained sampling.

## Install

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -e ".[dev]"
```

`unitracker` is pinned to a git ref in `pyproject.toml`.

## Commands

| command | does |
|---|---|
| `unitracker-corpus TUNES... [--out DIR] [--forms ...]` | each tune's lifted and optimised document forms, cached and verified |
| `unitracker-tokens [DIR] [--forms ...] [--encoders ...]` | tokens per form, tokenizer, lexical category and printer transform |
| `unitracker-tokenizer {train,eval}` | train and evaluate a document BPE |
| `unitracker-constrain {check,sample,bench,hazards} TOKENIZER [DOCS...]` | grammar masks over a tokenizer; sample documents the parser takes |
| `unitracker-learn {launch,status,run} [--preset smoke\|full] [--set KEY=VALUE ...]` | the learnability experiment; `launch` runs it detached in Docker on the host's GPUs |

## Docs

- [docs/learnability.md](docs/learnability.md): how well models learn each form and tokenizer
- [docs/tokenization.md](docs/tokenization.md): how documents tokenize, and the document BPE

## Development

```sh
black --check src tests && pylint src && pytest -n auto --cov=unitracker_learn
```

CI builds `docker/learn/Dockerfile` (stages `ml`, `deps`, `test`) and runs the same checks in it.
