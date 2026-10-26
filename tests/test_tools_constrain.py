"""tools/constrain.py: grammar masks over a document BPE's and Qwen2.5's ids take every valid
document whole with its EOS, refuse broken syntax as the grammar's parser does, and sample
documents the parser takes."""

import constrain
import numpy as np
import pytest
import test_tools_learn
from lark.exceptions import LarkError
from unitracker_grammar.canonical import parser
from unitracker_grammar.document import read

import optimiser_docs as D

SECTIONS = (
    "unitracker:2.3\n"
    "meta:{title:-}\n"
    "clock:{cycles_per_tick:19656,clock_hz:985000}\n"
    "slots:{freq:{wave:{reset:[row.note,row.freq]},tick:{hold:[step-1,row]}},"
    "pw:{pulse:{hold:[read,note-1,last-1]}}}\n"
    "order:[events,ctrl,pulse,filter,freq,pw,cut,read]\n"
    "state:{pw:{r:1,q:1}}\n"
    "tables:\n"
    " LAA:[2048,1792,1856,0,64]\n"
    " LAB:[0,1868]\n"
    "mod:\n"
    " seq(last):LAA[min(phi,last)]\n"
    " steps(last):v + LAB[min(phi,last)] - 1856\n"
    " pitch():F(N)\n"
    " note():v + (-v + F(N))*(phi < 1)\n"
    " ptab(b):|\n"
    "\tlda phi\n\tbne go\n\tlda #b\n\tsta r\n\tgo:\n\tldx r\n\tlda LAA.lo,x\n\tsta v\n\trts\n"
    "restart:\n"
    " hr(ad,sr):[{hold:2,gate:0,ad:15,sr:0},{ctrl:9,gate:1,ad:ad,sr:sr,run:c2},{},end]\n"
    "ins:\n"
    " i7:{ad:224,sr:248,ctrl:c1,filter:f1,wave.freq:note(),restart:hr(224,248)}\n"
    " i1:{ad:0,sr:229,ctrl:c3,pw:ptab(3),wave.freq:note(),tick.freq:none,restart:hr(0,229)}\n"
    "ctrl:\n"
    " c1:[{ctrl:129,note:+0},end]\n"
    " c2:[end]\n"
    " c3:[{ctrl:65},end]\n"
    " c4:[{ctrl:129,note:B7},{ctrl:17,note:g3},{ctrl:16,note:C2},end]\n"
    "filter:\n"
    " f1:[{cut:128,res:15,route:3,mode:64},{cut:v + 8},{loop:1}]\n"
    " f2:[{cut:v - 8},{loop:0}]\n"
    "phrase:\n"
    " p0:[-,- tempo:6]\n"
    " p1:[C3.i7 tick.freq:pitch(),-*14,E2.i1 run:f2]\n"
    " p2:ph_1(F2,E2)\n"
    " p4:[[C5.i1|-*2,D sr:76|-,f4 sr:76],D5 sr:76,c tick.freq:pitch() legato,A4]\n"
    " ph_1(x,z):[f1.i1 run:f2,x ad:144,[[f1,f2|f ad:144,z],-|f1]]\n"
    "section:\n"
    " s1:[play:[p4,p2,p1],[play:[~,p2,~]]x2]\n"
    "init:[-.i1 vol:15 tempo:2,-.i1,-.i1]\n"
    "song:\n"
    " - play:[p0,p0,p0]\n"
    " - s1x2\n"
    " - [play:[p1,p4,p2]|play:[p2,p1,p4]]\n"
    " - {loop:1}\n"
)
INLINE = (
    "unitracker:2.3\n"
    "meta:{title:-}\n"
    "tempo:[6,5]\n"
    "tables:\n"
    " PL:[1,2,3]\n"
    "init:[-.i1={ad:9,sr:218,ctrl:c4=[{ctrl:65,note:+0},end],pw:ptabs(b)=asm{lda phi; ora phi+1; "
    "bne go; lda #b; sta r; go:; ldx r; lda PL.lo,x; sta v; rts}(5),"
    "wave.freq:note()='v + (-v + F(N))*(phi < 1)'(),restart:hr(ad,sr)=[{hold:2,gate:0,ad:15,"
    "sr:0},{ctrl:9,gate:1,ad:ad,sr:sr,run:c2=[end]},{},end](0,229)} vol:15 tempo:2 "
    "tick.freq:cycle(last,period)='PL[max(0,phi - last) % period + min(phi,last)]'(0,8),"
    "-.i1,-.i1]\n"
    "song:\n"
    " - s1=[play:[p3=[C1.i5={ad:8,sr:201,ctrl:c5=[{ctrl:97,note:D3},{note:+0},end],"
    "filter:f3=[{cut:384,res:15,route:1,mode:48},{cut:v},end]},-,[- cut:24,f.i1,-|D.i5|C],"
    "a3.i1 sr:122,d tick.freq:pitch()='F(N)'()! legato,-*3],p8=[D4.i11=ins_1(t,t2,t3)={ad:0,"
    "sr:140,ctrl:arp(t,t2,t3)=[{ctrl:65},{wait:2,note:+t},{wait:2,note:+t2},{wait:2,note:+t3},"
    "{loop:1}](t,t2,t3)}(3,sus4,7),-*7,off],~]]\n"
    " - play:[p3,p8,p3+2]\n"
    " - s1x2\n"
    " - end\n"
)
HANDWRITTEN = [v for v in vars(D).values() if isinstance(v, str) and "\nsong:\n" in v]
LEARN = [test_tools_learn._doc(i) for i in range(3)]  # pylint: disable=protected-access
DOCS = HANDWRITTEN + LEARN + [SECTIONS, INLINE]
# one edit of a document: (old, new) in D.DEFAULTS, whether the grammar's parser takes it or not
EDITS = [
    ("i1:{ad:9,sr:240", "i1:{sr:240,ad:9"),
    ("[C4.i1,", "[C4.i1}"),
    ("-*3]", "-*3]]"),
    ("ctrl:c1,", "ctrl:c1,,"),
    ("unitracker:2.3", "unitracker:2.4"),
    ("title:T", "titel:T"),
    ("D,-*3", "H,-*3"),
    ("D,-*3", "dd,-*3"),
    ("D,-*3", "on,-*3"),
    ("D,-*3", "c4x,-*3"),
    (" - end\n", " - {loop:0}\n"),
    (" - end\n", " - end"),
    ("legato pw filter", "pw legato filter"),
    ("tick.freq:none", "tick.freq:'v + 1'"),
    ("tick.freq:none", "tick.freq:v + 1"),
    ("tick.freq:none", "tick.freq:'min(v,1)'"),
    ("tick.freq:none", "tick.freq:min(v,1)"),
    ("tick.freq:none", "tick.freq:f(v)"),
    ("tick.freq:none", "tick.freq:f(2)"),
    ("tick.freq:none", "tick.freq:N[1]"),
    ("tick.freq:none", "tick.freq:Nx[1]"),
    ("tick.freq:none", "tick.freq:pw1(2)"),
    ("tick.freq:none", "tick.freq:pwx(2)"),
    ("tick.freq:none", "tick.freq:min(2)"),
    ("tick.freq:none", "tick.freq:abs(v)"),
    ("tick.freq:none", "tick.freq:abs(vx)"),
    ("tick.freq:none", "tick.freq:max(vx,1)"),
    ("tick.freq:none", "tick.freq:min(a,b)='min(a,b)'(1,2)"),
    ("tick.freq:none", "tick.freq:abs(a)=asm{lda a}(1)"),
    ("tick.freq:none", "tick.freq:-12"),
    ("tick.freq:none", "tick.freq:-x"),
    ("tick.freq:none", "tick.freq:'-1 * -v'"),
    ("tick.freq:none", "tick.freq:asm{lda v,x; sta v+1}"),
    ("tick.freq:none", "tick.freq:asm{lda N.lo}"),
    ("tick.freq:none", "tick.freq:asm{lda Ab.lo}"),
    ("tick.freq:none", "tick.freq:g(a)='v + a'(3)"),
    ("tick.freq:none", "tick.freq:g(v)='v'(3)"),
    ("tick.freq:none", "tick.freq:g(phix)='phix'(3)"),
    ("tick.freq:none", "tick.freq:g(none)='v'(3)"),
    ("tick.freq:none", "tick.freq:g(c1)='v'(3)"),
    ("tick.freq:none", "tick.freq:g(c1x)='c1x'(3)"),
    ("tick.freq:none", "tick.freq:3!"),
    ("{hold:1,ctrl:64}", "{hold:1,ctrl:64,run:pw[end]}"),
    ("{hold:1,ctrl:64}", "{hold:1,ctrl:64,run:pw1}"),
    ("{hold:1,ctrl:64}", "{hold:1,ctrl:64,run:pw}"),
    ("{hold:1,ctrl:64}", "{hold:1,ctrl:64,note:+x}"),
    ("{hold:1,ctrl:64}", "{hold:1,ctrl:64,note:+v}"),
    ("- legato pw filter", "- legatox pw filter"),
    ("- legato pw filter", "- pwm.pw:3"),
    ("- legato pw filter", "- legato.pw:3"),
    ("- legato pw filter", "- pw:3 pw"),
    ("- legato pw filter", "- tempo:6/5 vtempo:2"),
]


@pytest.fixture(name="bpe", scope="module")
def _bpe(tmp_path_factory):
    """A document BPE trained on the documents, its EOS appended."""
    from unitracker.optimiser import tokenizer as tk  # pylint: disable=import-outside-toplevel

    out = tmp_path_factory.mktemp("bpe")
    tk.train(DOCS, 400, out, progress=False, text=lambda d: d)
    return constrain.Constraint(out)


@pytest.fixture(name="qwen", scope="module")
def _qwen():
    """Qwen2.5's tokenizer, its ``<|endoftext|>`` the EOS."""
    tok = constrain.tokenizer("qwen2.5")
    return constrain.Constraint(tok, tok.token_to_id("<|endoftext|>"))


@pytest.fixture(name="con", params=["bpe", "qwen"])
def _con(request):
    return request.getfixturevalue(request.param)


def parses(doc: str) -> bool:
    try:
        parser().parse(doc)
        return True
    except LarkError:
        return False


def refused(con, prefix: str, more: str) -> bool:
    """Whether a token of ``more`` falls outside its mask after ``prefix`` (which ends at a
    token boundary of ``prefix + more``)."""
    head, ids = con.encode(prefix), con.encode(prefix + more)
    assert ids[: len(head)] == head
    m = con.matcher(head)
    for t in ids[len(head) :]:
        if not m.mask()[t]:
            return True
        m.feed([t])
    return False


def test_every_document_is_taken_token_by_token_with_its_eos_at_the_end(con):
    assert all(map(parses, DOCS)) and all(read(d) for d in (SECTIONS, INLINE))
    assert con.check(DOCS) == [-1] * len(DOCS)


def test_any_tokenization_is_taken_not_only_the_canonical_one(con):
    chars = [sum((con.encode(c) for c in d), []) for d in (D.DEFAULTS, INLINE)]
    assert all(len(c) > len(con.encode(d)) for c, d in zip(chars, (D.DEFAULTS, INLINE)))
    assert con.check(chars) == [-1, -1]
    m = con.matcher(con.encode("unitracker:2.3\n"))
    assert "m" in {con.decode([t]) for t in np.flatnonzero(m.mask())}
    forced = constrain.Constraint(con.tok, con.eos, forcing=True)
    assert forced.check(chars) != [-1, -1] and forced.check([D.DEFAULTS]) == [-1]


def test_eos_is_allowed_only_where_the_document_may_end(con):
    ids = con.encode(SECTIONS)
    m = con.matcher()
    for t in ids:
        assert not m.mask()[con.eos] and not m.accepting
        m.feed([t])
    assert m.accepting and np.flatnonzero(m.mask()).tolist() == [con.eos]
    assert not con.matcher(ids[:-1]).feed([con.eos])
    assert m.feed([con.eos]) and m.done
    assert np.flatnonzero(m.mask()).tolist() == [con.eos] and m.feed([con.eos])
    assert not m.feed(ids[:1])


@pytest.mark.parametrize("old,new", EDITS)
def test_a_document_is_taken_whole_exactly_where_the_grammars_parser_takes_it(bpe, old, new):
    doc = D.DEFAULTS.replace(old, new, 1)
    assert doc != D.DEFAULTS
    assert (bpe.check([doc]) == [-1]) == parses(doc)


def test_masks_exclude_tokens_that_break_syntax(con):
    phrase = D.DEFAULTS[: D.DEFAULTS.index("p1:[C4.i1") + len("p1:[C4.i1")]
    assert refused(con, phrase, "}") and not refused(con, phrase, "]")
    ins = D.DEFAULTS[: D.DEFAULTS.index("{ad:9,sr:240,") + len("{ad:9,sr:240,")]
    assert refused(con, ins, "ad:9}") and not refused(con, ins, "ctrl:c1}")
    assert refused(con, "unitracker:2.3\n", "song:") and not refused(
        con, "unitracker:2.3\n", "meta:"
    )
    assert refused(con, D.HEAD, "voices:1,") and not refused(con, D.HEAD, "voices:1\n")


def test_batched_masks_are_each_matchers_own(con):
    ids = con.encode(INLINE)
    ms = [con.matcher(ids[:k]) for k in (0, 5, 40, len(ids))]
    batch = con.masks(ms)
    assert batch.shape == (4, con.n_vocab) and batch.dtype == bool
    for m, row in zip(ms, batch):
        assert np.array_equal(m.mask(), row)
    assert len({r.tobytes() for r in batch}) == 4
    assert np.array_equal(con.masks(ms, tensor=True).numpy(), batch)
    bits = con.bitmask(ms)
    t = np.arange(con.n_vocab)
    assert np.array_equal((bits[:, t >> 5] >> (t & 31)) & 1, batch)


def test_an_eos_past_the_vocabulary_is_added_and_the_mask_is_padded(bpe):
    size = bpe.tok.get_vocab_size()
    assert bpe.eos == size - 1 and bpe.tok.id_to_token(bpe.eos).startswith("<|eod")
    wide = constrain.Constraint(bpe.tok, size + 2, n_vocab=size + 100)
    assert wide.n_vocab == size + 100 and wide.check([D.DEFAULTS]) == [-1]
    assert not wide.masks([wide.matcher()])[0, size:].any()


def test_a_prompt_the_grammar_refuses_is_an_error(bpe):
    with pytest.raises(ValueError):
        bpe.matcher(bpe.encode("unitracker:3"))
    assert bpe.matcher(bpe.encode("unitracker:2.3\nmeta:{ti")).valid(bpe.encode("tle:")) > 0


def test_lookarounds_become_intersections_or_are_dropped():
    words = r"(?!(?:v|phi)\b|[a-g](?:-?[0-9]+)?\b)[a-z][a-z_]*\b"
    assert constrain.regex(words) == r"(/[a-z][a-z_]*/ & ~/(?:v|phi)|[a-g](?:-?[0-9]+)?/)"
    assert constrain.regex(r"(?![ij][0-9]+\()[a-z]+\(") == r"(/[a-z]+\(/ & ~/[ij][0-9]+\((?s:.*)/)"
    assert constrain.regex(r" pw(?![\w.:])") == "/ pw/"
    assert constrain.regex(r"(?:a|b)(?=\[)") == "/(?:a|b)/"
    assert not constrain.regex(r"(?!\()")
    with pytest.raises(ValueError):
        constrain.regex(r"(?<=a)b")
    with pytest.raises(ValueError):
        constrain.regex(r"a\bb")
    with pytest.raises(ValueError):
        constrain.regex(r"(?!a")
    out = constrain.lark('A.2: "x // y" // note\nb: A -> c\n')
    assert out == 'A: "x // y" \nb: A\n'


def test_python_classes_are_spelled_out_as_pythons_unicode_tables_have_them():
    digits = constrain.regex(r"[^\s,]\d")
    assert digits.startswith(r"/[^\x{9}-\x{d}\x{1c}-\x{20}") and r"\x{30}-\x{39}" in digits
    assert r"\x{16130}" not in digits and "\\d" in constrain.regex(r"\\d")


def test_only_the_safe_lexing_hazards_remain(bpe):
    assert set(constrain.hazards(bpe)) == constrain.SAFE


def test_samples_under_the_mask_that_end_parse(bpe):
    tally, refusals = constrain.agreement(bpe, DOCS, 8, 1500, 20)
    assert 4 <= tally["ended"] == tally["syntax"], refusals
    assert tally["read"] <= tally["parse"] <= tally["syntax"]


def test_the_command_line_checks_samples_and_times(bpe, tmp_path, capsys):
    bpe.tok.save(str(tmp_path / "tokenizer.json"))
    docs = tmp_path / "docs"
    docs.mkdir()
    for i, d in enumerate(DOCS[:6]):
        (docs / f"{i}.yaml").write_text(d.replace("meta:{title:T,author:A}", "meta:{x}"))
    assert constrain.main(["check", str(tmp_path), str(docs), "-n", "4", "--batch", "3"]) == 0
    assert "4/4 documents accepted" in capsys.readouterr().out
    (docs / "bad.yaml").write_text(D.DEFAULTS.replace("-*3]", "-*3]]"))
    assert constrain.main(["check", str(tmp_path / "tokenizer.json"), str(docs)]) == 1
    assert "6/7 documents accepted" in capsys.readouterr().out
    args = [str(tmp_path), str(docs / "0.yaml"), "--batch", "2", "--limit", "60"]
    constrain.main(["sample", *args, "--wander", "5"])
    assert "'samples': 2" in capsys.readouterr().out
    assert constrain.main(["bench", *args]) == 0
    assert "mask" in capsys.readouterr().out
