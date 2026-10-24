# Tokenization

How canonical documents (format 2.1) tokenize under LLM tokenizers, what costs the tokens, which
printer changes would save them, and a BPE trained on documents.

## Method

| step | tool |
|---|---|
| corpus: each cached tune a direct lifter reads, subtune 0, in every optimiser form, cached under `out/tokenize/<form>/<tune>.yaml` (a failure in `<tune>.err`, a form's timeout from its start) with `meta/<tune>.json` (format, song length) and `forms.csv` | `python tools/corpus.py forms .tunecache --out out/tokenize [--forms ...] [--timeout S]` |
| study: tokens per form and tokenizer, per category, per transform; `study.md`, `study.json` | `python tools/tokens.py out/tokenize [--forms ...] [--encoders ...]` |
| a document BPE: train on the training split, evaluate on the held-out split | `python -m unitracker.optimiser.tokenizer {train CORPUS --vocab N --out DIR, eval DIR CORPUS}` |

- **Forms** (`corpus.forms()`): `lifted` (the direct lifter's document), `named`
  (`optimise(text, ticks=3000)`), `single` and `all` (the named form's `--inline` modes), and
  where the optimiser has its lossy mode, `lossy` (`lossy=fit.TOLERANCES`) and `lossy-single`,
  `lossy-all`. A form is listed with its base form, so a new optimiser mode is a new entry.
- **Song length**: the ticks unitracker-player plays to the song's end or until a row starts at
  or before the row started last (its loop), at most 100000. **Events**: the lifted document's
  `event`s. Both are per tune, the same for every form.
- **Tokenizers**: tiktoken `cl100k_base`, `o200k_base`, and Qwen2.5's `tokenizer.json`
  (`Qwen/Qwen2.5-0.5B`, fetched without auth into `$HF_HOME`, `.cache/hf` for the tools).
- **Categories** (`tokenizer.categories`): the document is parsed by unitracker-grammar's LALR
  parser keeping every terminal; each character takes its terminal's category: `note`
  (`NOTE`, `NOTE_ATOM`), `number` (an integer of 1-2 digits), `long number` (3 or more, a call's
  arguments included), `name` (record names, templates, calls, parameters), `key` (the letters of
  the grammar's literals: `ad: `, `!freq:`, `end`), `table` (a `tables:`/`regs:` list, whole),
  `code` (a code block or `asm{}`, whole), `expr` (an expression's operators and symbols), `text`
  (meta values), `unicode` (any non-ASCII character), and otherwise by the character `punct`,
  `space`, or `indent` (a newline and the indentation after it, a line wrap's too). A token takes
  the category most of its characters have (ties to the earlier category in that order).
- **Transforms** (`tools/tokens.py`): each is a set of edits on the grammar's terminals (so
  reversible by a printer that knows it), applied to every document and re-tokenized; the saving
  is the corpus's total tokens before and after.
- **Comparisons** are over the tunes every form has.
