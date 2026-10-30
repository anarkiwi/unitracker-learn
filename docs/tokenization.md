# Tokenization

How canonical documents (format 2.1) tokenize under LLM tokenizers, what costs the tokens, which
printer changes would save them (all adopted as format 2.2), and a BPE trained on documents.

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
  the grammar's literals: `ad:`, ` freq:`, `end`), `table` (a `tables:`/`regs:` list, whole),
  `code` (a code block or `asm{}`, whole), `expr` (an expression's operators and symbols), `text`
  (meta values), `unicode` (any non-ASCII character), and otherwise by the character `punct`,
  `space`, or `indent` (a newline and the space or tab after it). A token takes
  the category most of its characters have (ties to the earlier category in that order).
- **Transforms** (`tools/tokens.py`): each is a set of edits on the grammar's terminals (so
  reversible by a printer that knows it), applied to every document and re-tokenized; the saving
  is the corpus's total tokens before and after.
- **Comparisons** are over the tunes every form has.

## Corpus

The 2040 `.sid` of `.tunecache`; 1064 are no direct lifter's format. Lifted subtune 0:

| format | tunes | lifted | failures (the largest causes) |
|---|---:|---:|---|
| GoatTracker 2 | 699 | 666 | 13 wavetable command, 12 `SidParseError` (no song table, play call, 1.x player), 6 lift timeouts, 2 `PlayError` |
| defMON | 57 | 51 | 3 not a recognised replay build, 3 others |
| Music Assembler | 38 | 3 | 24 play routine not the player's entry, 8 note control, 3 others |

Of the 829 lifted, `named`, `single` and `all` are written for 827 (2 named timeouts at 900 s)
and the lossy forms for 825 (4 lossy timeouts); the tables below are over those 825.

## Study

| form | tokenizer | tokens/doc mean | median | tokens/tick | tokens/event | chars/token |
|---|---|---:|---:|---:|---:|---:|
| lifted | cl100k_base | 16051 | 11055 | 1.409 | 10.92 | 1.97 |
| lifted | o200k_base | 16059 | 11002 | 1.410 | 10.93 | 1.97 |
| lifted | qwen2.5 | 17332 | 11891 | 1.522 | 11.79 | 1.82 |
| named | cl100k_base | 14382 | 10277 | 1.263 | 9.78 | 1.97 |
| named | o200k_base | 14391 | 10216 | 1.264 | 9.79 | 1.96 |
| named | qwen2.5 | 15521 | 11030 | 1.363 | 10.56 | 1.82 |
| single | cl100k_base | 14135 | 10103 | 1.241 | 9.62 | 2.01 |
| single | o200k_base | 14143 | 10038 | 1.242 | 9.62 | 2.00 |
| single | qwen2.5 | 15214 | 10769 | 1.336 | 10.35 | 1.86 |
| all | cl100k_base | 13767 | 9728 | 1.209 | 9.37 | 2.03 |
| all | o200k_base | 13773 | 9679 | 1.209 | 9.37 | 2.03 |
| all | qwen2.5 | 14846 | 10415 | 1.303 | 10.10 | 1.88 |
| lossy | cl100k_base | 13786 | 9806 | 1.210 | 9.38 | 1.99 |
| lossy | o200k_base | 13773 | 9749 | 1.209 | 9.37 | 1.99 |
| lossy | qwen2.5 | 14895 | 10569 | 1.308 | 10.13 | 1.84 |
| lossy-single | cl100k_base | 13539 | 9595 | 1.189 | 9.21 | 2.03 |
| lossy-single | o200k_base | 13524 | 9485 | 1.187 | 9.20 | 2.03 |
| lossy-single | qwen2.5 | 14588 | 10331 | 1.281 | 9.92 | 1.88 |
| lossy-all | cl100k_base | 13254 | 9311 | 1.164 | 9.02 | 2.06 |
| lossy-all | o200k_base | 13238 | 9257 | 1.162 | 9.01 | 2.06 |
| lossy-all | qwen2.5 | 14305 | 9998 | 1.256 | 9.73 | 1.91 |

tokens/tick and tokens/event are corpus totals over total song ticks and lifted events.
cl100k_base and o200k_base agree within 0.3%; Qwen2.5 takes 8% more, mostly in digits (it
splits every number into single digits: 1.00 tokens per digit).

## Categories

Share of tokens by category (cl100k_base unless named), characters' share and tokens per
character:

| category | chars (named) | lifted | named | all | lossy-all | named o200k | named qwen2.5 | tokens/char cl100k | tokens/char qwen2.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| key | 24.2% | 15.7% | 14.9% | 15.6% | 16.2% | 14.8% | 13.8% | 0.31 | 0.31 |
| name | 10.2% | 11.6% | 12.5% | 11.7% | 11.6% | 12.4% | 13.6% | 0.62 | 0.73 |
| note | 9.3% | 14.5% | 16.0% | 16.7% | 17.3% | 16.0% | 14.8% | 0.88 | 0.88 |
| number | 5.5% | 8.9% | 8.3% | 8.7% | 8.1% | 8.3% | 9.9% | 0.77 | 1.00 |
| long number | 1.7% | 1.2% | 1.2% | 1.2% | 1.3% | 1.2% | 3.2% | 0.35 | 1.00 |
| table | 1.4% | 1.6% | 1.8% | 1.9% | 1.8% | 1.8% | 2.5% | 0.66 | 0.98 |
| code | 10.5% | 8.0% | 8.9% | 7.6% | 5.9% | 8.9% | 8.4% | 0.43 | 0.44 |
| expr | 0.0% | 0.1% | 0.1% | 0.1% | 0.3% | 0.1% | 0.1% | 1.00 | 1.00 |
| text | 0.2% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.37 | 0.41 |
| unicode | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.66 | 0.65 |
| punct | 23.8% | 33.7% | 32.7% | 33.1% | 33.8% | 33.0% | 30.3% | 0.70 | 0.70 |
| space | 8.7% | 2.4% | 1.4% | 1.4% | 1.5% | 1.4% | 1.3% | 0.08 | 0.08 |
| indent | 4.5% | 2.1% | 2.1% | 1.9% | 2.0% | 2.1% | 2.0% | 0.24 | 0.24 |

What tokenizes badly (the lifted form's most frequent tokens per category, cl100k_base):

- **punct**, a third of all tokens at 0.70 per character: `,` (31%), `:` (25%) and `!` (16%) are
  tokens of their own, because the space after them (`, C4`, `: 9`) or the word after them
  (`!legato`, `!tick.freq:`) does not merge with them.
- **note**, 16% at 0.88 per character: `C4` is ` C` + `4`, `A#2` three tokens (` A`, `#`, `2`);
  digits never merge with letters; rests (` -*`, ` -,`) are a fifth of note tokens.
- **number**: `0` alone is 30% of them (`note: +0`, `gate: 0`); `: 9` costs `:`, ` `, `9`.
- **code**, 9%: each line's `\n` and 3-space indentation are two tokens (40% of code tokens)
  before ` lda`.
- **key** is cheap (0.31) except `.freq` (14% of key tokens: `tick.freq`, `wave.freq`) and
  `legato` (`leg` + `ato` after `!`).
- **name**: `.i` + digit for every instrument change (13% of name tokens).
- φ is gone in format 2.1 (`phi`); the only non-ASCII left is meta text. Long numbers and tables
  are 3% of tokens; hexadecimal and delta coding do not pay (below).

## Recommendations

Each transform is applied to every document and re-tokenized; the saving is of total tokens
(`named` form; `lossy-all` cl100k_base in the last column). All are printer changes the grammar
could read back; none loses information.

| change | transform | cl100k | o200k | qwen2.5 | lossy-all cl100k |
|---|---|---:|---:|---:|---:|
| commands introduced by a space, not `!` (`C4 legato tick.freq:none`) | `command-space` | 6.1% | 5.9% | 5.6% | 6.6% |
| no space after a flow comma (`[C4,-,D4]`, `{ad: 9,sr: 240}`) | `comma-space` | 5.0% | 4.9% | 4.6% | 5.0% |
| a phrase note without its octave where it repeats the previous note's (`C4, E, G, C5`) | `octave-elide` | 3.1% | 3.1% | 2.8% | 3.3% |
| that and sharps as lower case letters (`A#2`: `a2`) | `notes` | 3.8% | 3.8% | 3.5% | 4.1% |
| no space after a key's colon (`{ad:9}`) | `colon-space` | 2.8% | 2.9% | 2.6% | 2.7% |
| no line wraps (one line per record) | `no-wrap` | 1.1% | 1.1% | 1.1% | 1.6% |
| one space of indentation per level | `indent-1` | 1.0% | 1.0% | 1.0% | 0.4% |
| code lines indented by a tab | `code-tab` | 0.7% | 0.7% | 0.6% | 0.0% |
| **together**: command-space, comma-space, colon-space, no-wrap, notes, code-tab | `combined` | **19.5%** | **19.3%** | **18.0%** | **20.0%** |

Measured and not recommended:

| change | transform | cl100k | o200k | qwen2.5 |
|---|---|---:|---:|---:|
| integers of 3+ digits in `$` hexadecimal | `hex` | -0.6% | -0.7% | +1.4% |
| `tables:` lists delta-coded | `delta-tables` | 0.1% | 0.1% | 0.2% |
| keys shortened to unique prefixes | `short-keys` | -0.1% | -0.1% | -0.1% |

Whole words are single tokens in every vocabulary, so shortening keys and moving numbers to hex
cost tokens; hex only pays for Qwen2.5, whose digits are one token each. The savings are of
separators: tokenizers merge a word with one leading space or punctuation character, so a
format whose separators are single spaces before words (`command-space`) or vanish
(`comma-space`) gets the separator free.

## Format 2.2

Format 2.2 is every recommended change, `indent-1` included; the transforms it implements are
gone from `tools/tokens.py`. Measured with `tools/tokens.py` on a sample: 320 `.sid` drawn from
`.tunecache` (seed 22), the 109 a direct lifter reads, in the `lifted`, `named` and `all` forms,
the 2.1 documents from the 2.1 tools and the 2.2 ones from the 2.2 tools (the same documents,
their models printed in 2.2, take within 0.1% of the same tokens). Saving of total tokens:

| form | cl100k_base | o200k_base | qwen2.5 | 2.2 tokens/doc mean (cl100k_base) |
|---|---:|---:|---:|---:|
| lifted | 22.0% | 21.8% | 20.5% | 10633 |
| named | 21.8% | 21.5% | 20.3% | 9624 |
| all | 21.1% | 20.8% | 19.6% | 9279 |

## A document tokenizer

`python -m unitracker.optimiser.tokenizer` trains a byte-level BPE (Hugging Face `tokenizers`)
whose pre-tokenizer (`tokenizer.LEXEME`) splits a document into lexemes: a newline with its
indentation, a note, a key or name with the space before it and its `:`, a number, a
punctuation run with its space, spaces. No token crosses a lexeme. The split is by the file
stem's SHA-1 (1 in 10 held out), so a tune's forms fall in one split.

Trained on the training split of the `lifted`, `named`, `single` and `all` forms (3045
documents), evaluated on the held-out 268:

| vocabulary | tokens/doc mean | median | chars/token | vs cl100k_base | vs o200k_base | vs qwen2.5 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 11820 | 8565 | 2.49 | -19.1% | -19.1% | -24.8% |
| 2048 | 9877 | 7108 | 2.98 | -32.4% | -32.4% | -37.2% |
| 8192 | 9808 | 7074 | 3.00 | -32.8% | -32.8% | -37.6% |
| 32768 (stops at 10811) | 9808 | 7074 | 3.00 | -32.8% | -32.8% | -37.6% |

cl100k_base 14605, o200k_base 14605, qwen2.5 15727 tokens/doc on the same documents. The
vocabulary saturates at 10811 tokens: every lexeme of the corpus is one token, so a larger
vocabulary needs a pre-tokenizer that lets tokens span lexemes (`, C4`, `{ctrl: `).

```sh
python tools/corpus.py forms .tunecache --out out/tokenize --timeout 900
python tools/tokens.py out/tokenize
python -m unitracker.optimiser.tokenizer train out/tokenize/{lifted,named,single,all} --vocab 8192 --out out/bpe/8192
python -m unitracker.optimiser.tokenizer eval out/bpe/8192 out/tokenize/{lifted,named,single,all}
```
