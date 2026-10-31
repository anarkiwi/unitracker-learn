"""Small hand-written documents for the optimiser's tests."""

HEAD = "unitracker:2.3\nmeta:{title:T,author:A}\n"
MODS = "mod:\n note():F(N)\n vib(d):v + d*(phi % 4)\n wob(e):v + e*(phi % 4)\n"


TOP = ("restart", "ins", "ctrl", "pw", "filter", "phrase")


def doc(defs: str, play: str = "[p1]", head: str = "voices:1\n", mods: str = MODS) -> str:
    """A one-row ending document of ``defs`` (sections at 6 spaces, records at 8) playing
    ``play``: each section top-level, after the templates, in the grammar's order."""
    sections, kind = {}, None
    for line in defs.splitlines(keepends=True):
        if not line.startswith("        "):
            kind = line.strip()[:-1]
            sections[kind] = ""
        else:
            sections[kind] += line
    top = "".join(
        f"{k}:\n" + "".join(l[7:] for l in sections[k].splitlines(keepends=True))
        for k in TOP
        if k in sections
    )
    return f"{HEAD}{head}{mods}{top}song:\n - play:{play}\n - end\n"


DEFAULTS = doc(
    "      ins:\n"
    "        i1:{ad:9,sr:240,ctrl:c1,pw:2048,tick.freq:none}\n"
    "      ctrl:\n"
    "        c1:[{wait:0,ctrl:65,note:+0},{hold:1,ctrl:64},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1,- legato pw filter*1,D,-*3]\n",
    head="voices:1\ntempo:6\nslots:{freq:{base:{},tick:{}}}\n",
    mods="",
)

DEFAULTS_BOUND = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:c1,tick.freq:none}\n"
    "      ctrl:\n"
    "        c1:[{ctrl:65},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1 tick.freq:vib(3),-,D,-*3]\n",
    head="voices:1\nslots:{freq:{base:{},tick:{}}}\n",
    mods="mod:\n vib(d):v + d*(phi % 4)\n",
)

MERGE = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:c1}\n"
    "        i2:{ad:9,ctrl:c2}\n"
    "      ctrl:\n"
    "        c1:[{ctrl:65},end]\n"
    "        c2:[{ctrl:65},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1 freq:vib(2),-,D.i2 freq:wob(2),-*3]\n"
    "        p2:[C4.i1 freq:vib(2),-,D.i2 freq:wob(2),-*3]\n",
    play="[p1]\n - play:[p2]",
)

MERGE_PHI = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end]}\n"
    "      phrase:\n"
    "        p1:[C4.i1 freq:vib(2),- freq:wob(2),-*3]\n",
)

HOIST = doc(
    "      ins:\n"
    + "".join(f"        i{k}:{{ad:{k},ctrl:c1,pw:2048 + 0}}\n" for k in range(1, 6))
    + "      ctrl:\n"
    "        c1:[{ctrl:65},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1,-,D.i2,E.i3,F.i4,G.i5,-*3]\n",
    mods="",
)

HOIST_LATE = HOIST.replace("p1:[C4.i1,", "p1:[-,C4.i1,")

LOOPING = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end],pw:2048}\n"
    "        i2:{ad:9,ctrl:[{ctrl:65},end],pw:2048}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-,D.i2,-]\n",
    mods="",
).replace(" - end\n", " - {loop:0}\n")

CODE = (
    "mod:\n note():F(N)\n vib(d):v + d*(phi % 4)\n wob(e):v + e*(phi % 4)\n"
    " up(c):|\n\tlda v\n\tclc\n\tadc #c\n\tsta v\n\trts\n"
)
LOOPING_CODE = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end],pw:up(3)}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-,D,-]\n",
    mods=CODE,
).replace(" - end\n", " - {loop:0}\n")

TEMPLATE = doc(
    "      ins:\n"
    "        i1:{ad:9,sr:240,ctrl:c1,"
    "restart:[{hold:2,gate:0,ad:15,sr:0},{ad:9,sr:240},end]}\n"
    "        i2:{ad:8,sr:120,ctrl:c2,restart:[{gate:0,ad:15,sr:0},{ad:8,sr:120},end]}\n"
    "        i3:{ad:7,sr:100,ctrl:c3,restart:[{gate:0,ad:15,sr:0},{ad:7,sr:100},end]}\n"
    "      ctrl:\n"
    "        c1:[{ctrl:65,note:+7},{wait:1,note:+4},{wait:1,note:+0},{loop:0}]\n"
    "        c2:[{ctrl:65,note:+7},{wait:1,note:+3},{wait:1,note:+0},{loop:0}]\n"
    "        c3:[{ctrl:65,note:+5},{wait:1,note:+2},{wait:1,note:+0},{loop:0}]\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*3,D.i2,-*3,E.i3,-*3,F.i1 run:c2,-*3]\n",
    mods="",
)

# code modulators that are exactly expressions: a 16-bit slide, a note on φ 0 kept after, a
# square-wave vibrato (two calls) on a second slot and a pulse width sweep
EXACT_CODE = (
    "mod:\n"
    " ramp(c):|\n\tclc\n\tlda v\n\tadc #<c\n\tsta v\n\tlda v+1\n\tadc #>c\n"
    "\tsta v+1\n\trts\n"
    " start():|\n\tlda phi\n\tora phi+1\n\tbne done\n\tsta s\n\tldy nn\n"
    "\tlda F.lo,y\n\tsta v\n\tlda F.hi,y\n\tsta v+1\n\tdone:\n\trts\n"
    " wob(c):|\n\tlda phi\n\tand #2\n\tbne dn\n\tclc\n\tlda v\n\tadc #c\n"
    "\tsta v\n\tlda v+1\n\tadc #0\n\tsta v+1\n\trts\n\tdn:\n\tsec\n\tlda v\n"
    "\tsbc #c\n\tsta v\n\tlda v+1\n\tsbc #0\n\tsta v+1\n\trts\n"
)
EXACT = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:c1,pw:ramp(40),a.freq:start(),b.freq:wob(8)}\n"
    "        i2:{ad:9,ctrl:c1,pw:ramp(24),a.freq:start(),b.freq:wob(16)}\n"
    "      ctrl:\n"
    "        c1:[{ctrl:65},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*9,E.i2,-*9,G.i1,-*9]\n",
    head="voices:1\nslots:{freq:{a:{},b:{}}}\nstate:{freq:{s:1}}\n",
    mods=EXACT_CODE,
)

# a pulse width of 2048 plus 0 to 7 of a shift register's noise
NOISY = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end],pw:jit()}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*30]\n",
    head="voices:1\nstate:{pw:{r:1}}\n",
    mods="mod:\n jit():|\n\tlda r\n\tbne go\n\tlda #1\n\tgo:\n\tasl\n\tbcc nx\n"
    "\teor #29\n\tnx:\n\tsta r\n\tand #7\n\tsta v\n\tlda #8\n\tsta v+1\n\trts\n",
)

# put() keeps the frequency but stores the note that get() adds: put alone fits and is refused,
# while ramp() on the pulse width and the cutoff is kept
COUPLED = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:c1,pw:ramp(8),filter:f1,a.freq:put(),b.freq:get()}\n"
    "      ctrl:\n"
    "        c1:[{ctrl:65},end]\n"
    "      filter:\n"
    "        f1:[{cut:ramp(3),res:12,route:1,mode:16},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*5,E.i1,-*5]\n",
    head="voices:1\nslots:{freq:{a:{},b:{}}}\nstate:{freq:{s:1}}\n",
    mods="mod:\n put():|\n\tlda nn\n\tsta s\n\trts\n"
    " get():|\n\tclc\n\tlda v\n\tadc s\n\tsta v\n\tlda v+1\n\tadc #0\n\tsta v+1\n"
    "\trts\n" + EXACT_CODE[5 : EXACT_CODE.index(" start")],
)

# p2 is p1 two semitones up, p3 is p1 but for one interval
TRANSPOSE = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end]}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-,E4,G4,-,C5,-*3]\n"
    "        p2:[D4.i1,-,f4,A4,-,D5,-*3]\n"
    "        p3:[C4.i1,-,F,G,-,C5,-*3]\n",
    play="[p1]\n - play:[p2-1]\n - play:[p3]",
    mods="",
)

# a legato note bound to a modulator that jumps to the note: alone on its slot (dropped), after
# an instrument's binding on an earlier slot (that slot's none), not at the note (kept), and
# writing state a later binding reads (refused)
SNAP_MODS = (
    "mod:\n nt():(phi == 0)*F(N) + (phi > 0)*v\n snap():F(n)\n up():F(n) + 5\n"
    " snapc():|\n\tlda ne\n\tsta s\n\ttay\n\tlda F.lo,y\n\tsta v\n\tlda F.hi,y\n"
    "\tsta v+1\n\trts\n"
    " get():|\n\tclc\n\tlda v\n\tadc s\n\tsta v\n\tlda v+1\n\tadc #0\n\tsta v+1\n"
    "\trts\n"
)
SNAP = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end]}\n"
    "        i2:{ad:9,ctrl:[{ctrl:65},end],wave.freq:nt()}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-,E tick.freq:snap() legato,-,D.i2,-,G tick.freq:snap() legato,-,"
    "D.i2,-,E tick.freq:up() legato,-*3]\n",
    head="voices:1\nslots:{freq:{wave:{},tick:{}}}\n",
    mods=SNAP_MODS,
)

# SNAP with a repeat group before and around the binding events: the compiled phrase's events
# are not the document phrase's
SNAP_GROUP = SNAP.replace(
    "p1:[C4.i1,-,E tick.freq:snap() legato,-,D.i2,-,G tick.freq:snap() legato,-,"
    "D.i2,-,E tick.freq:up() legato,-*3]",
    "p1:[[C4.i1,-]x2,[D.i2,-,E tick.freq:snap() legato,-]x2,D.i2,-,G tick.freq:snap() legato,-]",
)

SNAP_STATE = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end]}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-,E freq:snapc() legato,- freq:none,- freq:get(),-*3]\n",
    head="voices:1\nstate:{freq:{s:1}}\n",
    mods=SNAP_MODS,
)

# i1's none ends no binding at its notes, i2's ends the one C4's command made
INS_NONE = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:c1,freq:none}\n"
    "        i2:{ad:8,ctrl:c1,freq:none}\n"
    "      ctrl:\n"
    "        c1:[{hold:2,ctrl:65},{},{ctrl:64},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1 freq:vib(2),-,D.i2,-*3,E.i1,-*3]\n",
    mods="mod:\n vib(d):v + d*(phi % 4)\n",
)

# a restart whose last row is empty: the row before it falls a tick before the note
RESTART_EMPTY = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end],restart:[{gate:0,ad:15},"
    "{hold:2,ad:9,sr:240},{},end]}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*3,D4,-*3]\n",
    mods="",
)

JOINT_CODE = (
    "mod:\n"
    " set(d):|\n\tlda phi\n\tora phi+1\n\tbne done\n\tlda #d\n\tsta s\n\tdone:\n"
    "\trts\n"
    " add():|\n\tclc\n\tlda v\n\tadc s\n\tsta v\n\tlda v+1\n\tadc #0\n\tsta v+1\n"
    "\trts\n"
)

# set() stores the step add() adds each tick: neither fits alone, together a ramp per instrument
JOINT = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:c1,a.freq:set(8),b.freq:add()}\n"
    "        i2:{ad:9,ctrl:c1,a.freq:set(16),b.freq:add()}\n"
    "      ctrl:\n"
    "        c1:[{ctrl:65},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*5,E.i2,-*5,G.i1,-*5]\n",
    head="voices:1\nslots:{freq:{a:{},b:{}}}\nstate:{freq:{s:1}}\n",
    mods=JOINT_CODE,
)

# JOINT with an expression slot firing between the two code slots
INTERLEAVED = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:c1,a.freq:set(8),m.freq:v + 1,b.freq:add()}\n"
    "      ctrl:\n"
    "        c1:[{ctrl:65},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*5]\n",
    head="voices:1\nslots:{freq:{a:{},m:{},b:{}}}\nstate:{freq:{s:1}}\n",
    mods=JOINT_CODE,
)

ONCE = doc(
    "      ins:\n"
    "        i1:{ad:9,sr:240}\n"
    "      phrase:\n"
    "        p1:[C4.i1 pw:'800 + 0',-*3,- pw:none,D pw:'v + 8',-*2,- pw:none]\n"
)

# templates of the document's own that play as prelude calls (code and expressions), and one
# (wob) that plays as none
PRELUDE_MODS = (
    "mod:\n gl(c):v + c\n tr(a,b):(-a + b)*(phi // 4 % 2) + a\n"
    " vb(d):(-abs((phi + 1) % 12 - 6) + 3)*(phi >= 2)*d + F(N)\n"
    " sw(a):min(2000,20*phi + a)\n wob(e):v + e*(phi % 4)\n"
    " up(c):|\n\tclc\n\tlda v\n\tadc #<c\n\tsta v\n\tlda v+1\n\tadc #>c\n\tsta v+1\n\trts\n"
)
PRELUDE = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end],pw:tr(800,1600),freq:vb(5)}\n"
    "        i2:{ad:9,ctrl:[{ctrl:65},end],pw:sw(300),freq:up(40)}\n"
    "        i3:{ad:9,ctrl:[{ctrl:65},end],pw:up(9),freq:gl(-7)}\n"
    "        i4:{ad:9,ctrl:[{ctrl:65},end],freq:wob(3)}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*20,E.i2,-*20,G.i3,-*20,C5.i1,-*9,D.i4,-*9]\n",
    mods=PRELUDE_MODS,
)

# instruments alike but for integers (i1-i3), and one of another shape (i4)
INSTRUMENTS = doc(
    "      restart:\n"
    "        hr(ad,sr):[{gate:0,ad:15,sr:0},{ad:ad,sr:sr},end]\n"
    "      ins:\n"
    "        i1:{ad:9,sr:240,ctrl:c1,pw:800,restart:hr(9,240)}\n"
    "        i2:{ad:8,sr:240,ctrl:c1,pw:900,restart:hr(8,240)}\n"
    "        i3:{ad:7,sr:240,ctrl:c1,pw:1000,restart:hr(7,240)}\n"
    "        i4:{ad:7,sr:240,ctrl:c2,pw:1000,restart:hr(7,240)}\n"
    "      ctrl:\n"
    "        c1:[{ctrl:65},end]\n"
    "        c2:[{ctrl:17},end]\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*3,D.i2,-*3,E.i3,-*3,F.i4,-*3]\n",
    mods="",
)

# phrases alike but for their notes (p1, p2; p3 has another instrument), groups among them
PHRASES = doc(
    "      ins:\n"
    "        i1:{ad:9,ctrl:[{ctrl:65},end]}\n"
    "        i2:{ad:8,ctrl:[{ctrl:65},end]}\n"
    "      phrase:\n"
    "        p1:[C4.i1,-,E4,G4 pw:800,[C5,D5]x2,-*3,E4 pw:900,-,G4 pw:1000,-*2,off]\n"
    "        p2:[D4.i1,-,E4,A4 pw:800,[D5,E5]x2,-*3,E4 pw:900,-,G4 pw:1000,-*2,off]\n"
    "        p3:[F4.i2,-,E4,C5 pw:800,[F5,G5]x2,-*3,E4 pw:900,-,G4 pw:1000,-*2,off]\n",
    play="[p1]\n - play:[p2]\n - play:[p3]",
    mods="",
)

# an arpeggio called with its notes in one order not ascending, directly and through an
# instrument template, and a template whose note parameter also times a row
CHORDS = doc(
    "      ins:\n"
    "        ins_1(ad,t,t2,t3):{ad:ad,ctrl:arp(t,t2,t3)}\n"
    "        i1:ins_1(9,10,3,7)\n"
    "        i2:{ad:9,ctrl:arp(11,4,7)}\n"
    "      ctrl:\n"
    "        arp(t,t2,t3):[{ctrl:65},{note:+t2},{note:+t3},{note:+t},{loop:0}]\n"
    "        sw(t,w):[{ctrl:65},{note:+t},{wait:t,note:+w},{loop:0}]\n"
    "      phrase:\n"
    "        p1:[C4.i1,-*3,E.i2,-*3,G.ins_1(9,11,4,7),-*3,C.i1 run:arp(9,3,7),-*3,"
    "D.{ctrl:sw(4,7)},-*3]\n",
    mods="",
)

# templates and records that are their calls, each referenced once (p2, i2) or more (p1, i1)
CALLS = doc(
    "      ins:\n"
    "        ins_1(ad,p):{ad:ad,ctrl:[{ctrl:65},end],pw:p}\n"
    "        i1:ins_1(9,800)\n"
    "        i2:ins_1(8,1600)\n"
    "      phrase:\n"
    "        ph_1(x,y):[x.i1,-,y.i2,-*3]\n"
    "        p1:ph_1(C4,E4)\n"
    "        p2:ph_1(D4,F4)\n",
    play="[p1]\n - play:[p2]\n - play:[p1]",
    mods="",
)
