#!/usr/bin/env python3
"""asm51_shim.py -- assemble Intel/Keil ASM-51 teaching sources with open-source AS31.

Measured baseline (Ubuntu 24.04: as31 2.3.1, sdcc 4.2.0, uCsim 0.6.4)
--------------------------------------------------------------------
AS31 already understands the *instruction* layer of these files: Intel
mnemonics, `H`/`B`/`O`/`D` number suffixes (`#55H`, `0FFH`, `1010B`), `'c'`
character constants, `;` comments, dotted bit operands (`CLR P0.7` -> `C2 87`),
and non-monotonic `.ORG` (as long as nothing overlaps).

What AS31 rejects is purely the *assembler-language* layer. Each rule below was
decided by a probe, not by assumption -- see PROBE RESULTS at the bottom.

Rewrite rules
-------------
  1. file must end in a newline          (last line is otherwise dropped: probe
                                         `.END` w/o newline -> "syntax error near END")
  2. `//`            -> `;`              (C-style comments; Keil tolerates, AS31 does not)
  3. `#define S V`   -> `.EQU S, V` / `.FLAG S, V` when V is `Pn.x`
  4. `ORG x`         -> `.ORG x`         `END [lbl]` -> `.END`
     `DB list`      -> `.BYTE list`     `DW` -> `.WORD`   `DS n` -> `.SKIP n`
     `S EQU v`      -> `.EQU S, v`      `S BIT b`/`S DATA a` -> `.EQU`/`.FLAG`
  5. `jmp X`         -> `ljmp X`         `call X` -> `lcall X`   (Keil pseudo-mnemonics)
  6. `$`             -> `*`              (location counter; `sjmp $` fails, `sjmp *` passes)
  7. rename labels/refs of reserved words {org,end,byte,word,flag,equ,skip,db,dw,ds}
     by appending `_`  (AS31 also treats the *dotless* directive names as keywords)
  8. `'MULTI'`       -> `"MULTI"` inside data directives (AS31: single quotes are
     character constants only -> "Missing quote in character constant")
  9. `n DUP(x)`      -> `x, x, ...`
 10. vendor bookkeeping (`NAME`,`AREA`,`CSEG`,`RSEG`,`PROC`,`PUBLIC`,...) is
     commented out and counted -- never silently dropped.
 11. `// text` on a line of its own -> `; text` (AS31 never sees `//`).
 12. Keil `$`-directives (`$INCLUDE(f)`, `$NOMOD51`, `$SET`, `$MODEL`) are
     commented out and counted; a dropped `$INCLUDE` is tagged `INCLUDE_DROPPED`
     because the header usually carries symbols the body then references.
 13. operand-position hex that starts with a letter (`MOV F0H, #00H`) -> `0xF0`
     (AS31: "if the first character is non-numerical, then either 0x or 0 must
     be specified"). The mnemonic, `label:` definitions and quoted text are
     never touched, so no branch target can be mangled.
 14. `LOW(x)`/`HIGH(x)` -> `(x - (x / 100H) * 100H)` / `(x / 100H)`. AS31 has
     `+ - * /` but no `AND`/`SHR` (measured), and the desugaring was checked
     numerically: `tab=0x103` -> `74 01`, `75 F0 03`, `90 01 03`.
 15. `PUSH A`/`POP A` -> `PUSH ACC`/`POP ACC` (Keil accepts the mnemonic, AS31
     wants the direct symbol; `POP ACC` -> `D0 E0`).
 16. a definition whose name is one of AS31's *predefined* SFR symbols is
     dropped, not emitted: AS31 answers "Attempt to redefine symbol: T2CON".
     The set was measured by probing every SFR name plus all 966 distinct
     symbol names defined across the 9 repositories (`_p_ops.sh`,
     `_p_names.py`): P0-P3, ACC, B, PSW, SP, DPL, DPH, TCON, TMOD, TL0, TH0,
     TL1, TH1, SCON, SBUF, IE, IP, PCON, T2CON, T2EX, TR2, RCAP2H, RCAP2L
     collide; T2MOD, IE2, SCON2, OKON, OLEN, CCON, CH, CL do NOT.
 17. `x mod y` -> `((x) % (y))` -- AS31 has `%` but not the `mod` keyword
     (measured), and the byte values agree: `t0_data = 65536-5888 = 0xE900`
     gives `MOV TL0, #(t0_data % 256)` -> `75 8A 00`.
 18. `db"text"` -> `db "text"`: without the space AS31 sees one token
     (`db"`), which rule 7 then renamed to `db_` and the line became garbage.
 19. rule 7's rename must advance by the *original* token length: `SKIP:` ->
     `SKIP_:` not `SKIP_ ` -- losing the colon is exactly why `skip:` and
     `BYTE:` still failed after renaming (5 files).

Not papered over (reported by tag instead)
-------------------------------------------
  * AS31 operand forms it genuinely does not have: `CJNE direct, #data, rel`
    (probe c04), `ADDC Rn, A`, `SYM EQU Rn` (register alias), `.ORG` onto
    already-emitted bytes ("Location counter overlaps", probe ov3).
  * Source defects: `INC R2.` (trailing dot), `CJE` (typo for `CJNE`),
    `DB '1","2','3','0'` (mismatched quotes), `MOV WDTCN,...` under `$NOMOD51`
    with no vendor SFR header, `LCALL` of a routine that lives in a sibling
    file. The shim never invents a symbol or an instruction the repository
    did not contain -- an unresolved symbol stays an unresolved symbol.

Usage
-----
  asm51_shim.py --out build --summary step.md file1.asm file2.a51 [--gate name1,name2]

Exit code is 1 only if a `--gate` file fails; otherwise 0 (informational matrix).
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

RESERVED_WORDS = {"org", "end", "byte", "word", "flag", "equ", "skip", "db", "dw", "ds"}

VENDOR_ONLY = {
    "name", "title", "print", "eject", "public", "extern", "externl",
    "cseg", "dseg", "bseg", "xseg", "iseg", "rseg", "area", "proc", "endp",
    "using", "noget", "seg", "endmod", "modtel", "modsrc",
}

# SFR byte addresses whose low 3 bits are individually addressable.
SFR_BITS = {
    "p0": 0x80, "p1": 0x90, "p2": 0xA0, "p3": 0xB0,
    "tcon": 0x88, "scon": 0x98, "ie": 0xA8, "ip": 0xB8,
    "psw": 0xD0, "acc": 0xE0, "a": 0xE0, "b": 0xF0, "scon2": 0x90,
}

RE_LABEL_DEF = re.compile(r"^\s*([A-Za-z_][\w]*)(\s*):\s*(.*)$")
RE_SYMDEF = re.compile(r"^([A-Za-z_][\w]*)\s+(EQU|BIT|DATA|XDATA)\s+(.+?)\s*$", re.I)
RE_DEFINE = re.compile(r"^#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", re.I)
RE_DUP = re.compile(r"\b(\d+)\s+DUP\s*\(([^()]*)\)", re.I)
RE_BITREF = re.compile(r"^([A-Za-z_]\w*)\.([0-7])$")
RE_MULTIQUOTE = re.compile(r"'([^'\r\n]{2,})'")
RE_TOKEN = re.compile(r"[A-Za-z_]\w*|\d[HhDdOoBb]?\w*|0[Xx][0-9A-Fa-f]+|\S")
RE_LOWHIGH = re.compile(r"\b(LOW|HIGH)\s*\(\s*([^()]+?)\s*\)", re.I)
RE_DOLLAR_DIR = re.compile(r"^\$([A-Za-z_]\w*)")
RE_STACK_A = re.compile(r"^(PUSH|POP)\s+A\s*$", re.I)
RE_LEADHEX = re.compile(r"\b([A-Fa-f][0-9A-Fa-f]*)[Hh](?!:)\b")


def split_comment(line):
    """Split (code, comment) on the first ';' or '//' that is outside a string."""
    i, in_dq, in_sq, cmt = 0, False, False, None
    while i < len(line):
        ch = line[i]
        if cmt is None:
            if ch == '"' and not in_sq:
                in_dq = not in_dq
            elif ch == "'" and not in_dq:
                in_sq = not in_sq
            elif not in_dq and not in_sq:
                if ch == ";":
                    cmt = i
                elif ch == "/" and line[i:i + 2] == "//":
                    cmt = i
        i += 1
    if cmt is None:
        return line, ""
    return line[:cmt], line[cmt:]


def parse_int(tok):
    t = (tok or "").strip()
    try:
        if re.fullmatch(r"0[Xx][0-9A-Fa-f]+", t):
            return int(t, 16)
        if re.fullmatch(r"[0-9][0-9A-Fa-f]*[Hh]", t):
            return int(t[:-1], 16)
        if re.fullmatch(r"[0-9]+[Bb]", t):
            return int(t[:-1], 2)
        if re.fullmatch(r"[0-7]+[Oo]", t):
            return int(t[:-1], 8)
        if re.fullmatch(r"\d+[Dd]?", t):
            return int(t[:-1] if t[-1] in "Dd" else t, 10)
    except ValueError:
        return None
    return None


def bit_address(expr):
    """Keil bit expression -> absolute bit address, or None if not computable."""
    m = RE_BITREF.match(expr.strip())
    if not m:
        return None
    base, n = m.group(1).lower(), int(m.group(2))
    if base in SFR_BITS:
        return SFR_BITS[base] + n
    v = parse_int(base)
    if v is not None and 0x20 <= v <= 0x2F:
        return (v - 0x20) * 8 + n
    if v is not None and v % 8 == 0 and 0x80 <= v <= 0xF8:
        return v + n
    return None


def named_bit_ref(expr):
    """True when `expr` is `SFRname.bit`, which AS31's `.FLAG` takes literally.

    Measured: `.FLAG RS, P1.3` assembles; `.FLAG RS, 0x93` does NOT (probe
    p06 vs the corpus cluster), so the dotted symbol form must be preserved.
    """
    m = RE_BITREF.match(expr.strip())
    return bool(m) and (m.group(1).lower() in SFR_BITS)


SFR_NAMES = set(SFR_BITS) | {"tmod", "tl0", "th0", "tl1", "th1", "dpirl", "dpirh",
                             "dpl", "dph", "sp", "dptr", "pcon", "ie", "ip"}

# Measured, not guessed: `.EQU <name>, 0` -> "Attempt to redefine symbol" for
# exactly these (probe `_p_ops.sh` over 50 candidate SFR names, then
# `_p_names.py` over the 966 distinct symbols the 9 repos define).
AS31_PREDEFINED = {
    "P0", "P1", "P2", "P3", "ACC", "B", "PSW", "SP", "DPL", "DPH",
    "TCON", "TMOD", "TL0", "TH0", "TL1", "TH1", "SCON", "SBUF", "IE", "IP",
    "PCON", "T2CON", "T2EX", "TR2", "RCAP2H", "RCAP2L",
}


def hexfix(text, stats):
    """Intel hex literals that start with a letter (`E803H`) need AS31's `0x` form.

    AS31: "if the first character is non-numerical, then either 0x or 0 must be
    specified" -- so `#E803H` and `DB ABH` are rejected while `#0E803H`/`0ABH`
    are accepted. Applied only in immediate (`#`) and data-list positions, never
    to bare tokens, so a label named `E8H` cannot be mangled.
    """
    def rep(m):
        stats["hexfix"] += 1
        return "#0x" + m.group(1)
    text = re.sub(r"#([A-Fa-f][0-9A-Fa-f]*)[Hh]\b", rep, text)
    parts = text.split(",")
    fixed = []
    for p in parts:
        m = re.fullmatch(r"(\s*)([A-Fa-f][0-9A-Fa-f]*)([Hh])(\s*)", p)
        if m:
            stats["hexfix"] += 1
            p = "%s0x%s%s" % (m.group(1), m.group(2), m.group(4))
        fixed.append(p)
    return ",".join(fixed)


def map_outside_quotes(text, fn):
    """Apply `fn` to every run of `text` that is not inside `'...'` or `"..."`."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in ('"', "'"):
            j = text.find(ch, i + 1)
            if j < 0:
                out.append(text[i:])
                break
            out.append(text[i:j + 1])
            i = j + 1
            continue
        j = i
        while j < n and text[j] not in ('"', "'"):
            j += 1
        out.append(fn(text[i:j]))
        i = j
    return "".join(out)


def hex_operand(text, stats):
    """`MOV F0H, #00H` -> `MOV 0xF0, #00H`; the mnemonic token is never touched.

    AS31 needs `0x`/`0` in front of a hex literal whose first character is a
    letter. Doing this only after the first token means a `label:` definition
    cannot be converted, and the `(?!:)` guard covers a reference to one.
    """
    m = re.match(r"^\s*[A-Za-z_]\w*", text)
    head = m.group(0) if m else ""
    rest = text[len(head):]

    def fn(seg):
        new, n = RE_LEADHEX.subn(lambda mm: "0x" + mm.group(1), seg)
        stats["hex_operand"] += n
        return new

    return head + map_outside_quotes(rest, fn)


def lowhigh(text, stats):
    """Keil `LOW(x)`/`HIGH(x)` -> arithmetic AS31 really has (see rule 14)."""
    def rep(m):
        x, hi = m.group(2), m.group(1).lower() == "high"
        stats["keil_lowhigh:" + ("high" if hi else "low")] += 1
        if hi:
            return "(%s / 100H)" % x
        return "(%s - (%s / 100H) * 100H)" % (x, x)

    return map_outside_quotes(text, lambda s: RE_LOWHIGH.sub(rep, s))


RE_MOD = re.compile(r"(#?\s*)([^\s,()#]+)\s+mod\s+([^\s,()]+)", re.I)


def modop(text, stats):
    """`x mod y` -> `((x) % (y))` (rule 17).

    Measured on AS31 2.3.1: the `mod` keyword is rejected ("syntax error near
    t0_data") while `%` is accepted, and both agree numerically --
    `t0_data = 65536-5888 = 0xE900` gives `#(t0_data % 256)` -> `75 8A 00` and
    `#(t0_data / 256)` -> `75 8C E9`. The `#` stays outside the parentheses:
    `#t0_data mod 256` -> `#((t0_data) % (256))`.
    """
    def rep(m):
        stats["modop"] += 1
        return "%s((%s) %% (%s))" % (m.group(1), m.group(2), m.group(3))

    return RE_MOD.sub(rep, text)


def stack_acc(text, stats):
    """`PUSH A`/`POP A` -> `PUSH ACC`/`POP ACC` (rule 15)."""
    m = RE_STACK_A.match(text.strip())
    if not m:
        return text
    stats["push_pop_acc"] += 1
    return "%s ACC" % m.group(1).upper()


def rename_reserved(code, stats):
    """Append '_' to any bare token that AS31 treats as a keyword (outside strings).

    A token preceded by '.' is a directive name we emitted ourselves (`.ORG`,
    `.EQU`, `.BYTE`, `.FLAG`, ...) and must be left exactly as written.
    """
    out, i, in_dq, in_sq = [], 0, False, False
    while i < len(code):
        ch = code[i]
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            out.append(ch); i += 1; continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            out.append(ch); i += 1; continue
        if not in_dq and not in_sq:
            m = re.match(r"[A-Za-z_]\w*", code[i:])
            if m:
                tok = w = m.group(0)
                prev = code[i - 1] if i > 0 else ""
                if prev == ".":
                    out.append(w); i += len(tok); continue
                if w.lower() in RESERVED_WORDS:
                    stats["rename:" + w.lower()] += 1
                    w = w + "_"
                out.append(w); i += len(tok); continue
        out.append(ch); i += 1
    return "".join(out)


def expand_dup(text):
    prev = None
    while prev != text:
        prev = text
        text = RE_DUP.sub(lambda m: ", ".join([m.group(2)] * int(m.group(1))), text)
    return text


def fix_data_strings(text, stats):
    """`'WEEK'` -> `"WEEK"` (only meaningful inside data directives)."""
    def one(m):
        stats["multiquote"] += 1
        return '"%s"' % m.group(1)
    return RE_MULTIQUOTE.sub(one, text)


def data_operand(rest, stats):
    rest = expand_dup(rest)
    return fix_data_strings(rest, stats)


def emit(indent, body, comment):
    tail = (" " + comment) if comment else ""
    # AS31 only knows `;` comments; Keil/student sources use `//` freely.
    if tail.lstrip().startswith("//"):
        tail = " ;" + tail.lstrip()[2:]
    if not body.strip():
        return (indent + tail.strip()) if comment else indent.rstrip()
    return indent + body + tail


def rewrite_line(raw, stats):
    line = raw.rstrip("\r\n")
    code, comment = split_comment(line)
    code_r = code.rstrip()
    stripped = code_r.strip()
    if not stripped:
        # rule 11: a line that is *only* a C-style comment still reaches AS31.
        if comment.startswith("//"):
            stats["comment_only_slash"] += 1
            return re.match(r"^\s*", line).group(0) + ";" + comment[2:]
        return line  # pure `;` comment / blank: untouched

    indent = re.match(r"^\s*", code_r).group(0)
    label = ""
    body = stripped
    mlab = RE_LABEL_DEF.match(stripped)
    if mlab and not re.match(r"(?i)^(org|end|db|dw|ds|equ|bit|data)\b", stripped):
        label = mlab.group(1) + ":"
        body = mlab.group(3).strip()

    # `db"text"` written without a space: AS31's tokenizer needs the directive
    # and its operand separated (measured: `db"` -> "syntax error near db_").
    mglue = re.match(r"^([A-Za-z_.]\w*)([\"'])", body)
    if mglue:
        stats["glued_quote"] += 1
        body = mglue.group(1) + " " + body[len(mglue.group(1)):]

    low_first = body.split(None, 1)[0].lower() if body else ""
    rest = body[len(body.split(None, 1)[0]):].strip() if body else ""

    def out(text):
        pre = (label + " " if label else "")
        return emit(indent, rename_reserved(pre + text, stats), comment)

    # --- C preprocessor style
    mdef = RE_DEFINE.match(body)
    if mdef:
        sym, val = mdef.group(1), mdef.group(2)
        if sym.upper() in AS31_PREDEFINED:
            # `#define p1 0x90` would fight AS31's built-in SFR table
            # (measured: "Attempt to redefine symbol: P1"). Drop it -- the
            # built-in address is the hardware truth either way.
            stats["define_redundant_sfr"] += 1
            return emit(indent, rename_reserved(";[shim] redundant SFR define: " + body, stats), comment)
        if named_bit_ref(val):
            stats["define"] += 1
            return emit(indent, rename_reserved(".FLAG %s, %s" % (sym, val), stats), comment)
        ba = bit_address(val)
        stats["define"] += 1
        if ba is not None:
            return emit(indent, rename_reserved(".EQU %s, 0x%x" % (sym, ba), stats), comment)
        return out(".EQU %s, %s" % (sym, hexfix(val, stats)))
    if body.startswith("#"):
        stats["neutered:#" + low_first] += 1
        return emit(indent, rename_reserved(";[shim] disabled: " + body, stats), comment)

    # --- Keil `$`-directives: no AS31 equivalent (rule 12). `$INCLUDE` is
    # flagged separately because the dropped header often defines symbols the
    # body then references, which shows up later as UNDEFINED.
    mdol = RE_DOLLAR_DIR.match(body)
    if mdol:
        dname = mdol.group(1).upper()
        stats["neutered:$" + dname] += 1
        note = "INCLUDE_DROPPED " if dname.startswith("INCLUDE") else ""
        return emit(indent, rename_reserved(";[shim] %sdisabled Keil $%s directive" % (note, dname), stats)
                    + " -- " + body, comment)

    # --- vendor bookkeeping AS31 cannot express
    if low_first in VENDOR_ONLY:
        stats["neutered:" + low_first] += 1
        return emit(indent, rename_reserved(";[shim] disabled vendor directive: " + body, stats), comment)

    # --- symbol definitions
    msym = RE_SYMDEF.match(body)
    if msym:
        sym, kind, val = msym.group(1), msym.group(2).lower(), msym.group(3)
        if label:  # "NAME: SYM EQU v" is not a thing; treat literally
            return out(body)
        if sym.upper() in AS31_PREDEFINED:
            # rule 16: AS31 already owns this name (measured "Attempt to
            # redefine symbol: T2CON"). Its built-in value is the real SFR
            # address, so the source's EQU carries no extra information.
            stats["symdef_redundant_sfr"] += 1
            return emit(indent, rename_reserved(";[shim] dropped redundant SFR symdef (%s is predefined by as31)" % sym.upper(), stats), comment)
        if named_bit_ref(val):
            stats["symdef:" + kind] += 1
            return emit(indent, rename_reserved(".FLAG %s, %s" % (sym, val), stats), comment)
        ba = bit_address(val)
        stats["symdef:" + kind] += 1
        if ba is not None:
            return emit(indent, rename_reserved(".EQU %s, 0x%x" % (sym, ba), stats), comment)
        if kind == "bit":
            # Keil `SYM BIT n` states the *bit address* itself; it does not have
            # to be in the bit-addressable window (martyniak's clock.asm uses
            # `t0_flag BIT 02h`, i.e. TCON.2 -- sloppy, but it assembles).
            # Measured: neutering that line instead left 10 later `SETB t0_flag`
            # references undefined, so a numeric value must be passed through.
            if parse_int(val) is None:
                stats["unresolved_bit"] += 1
                return emit(indent, rename_reserved(";[shim] unresolved BIT %s -> %s" % (sym, val), stats), comment)
            stats["bit_numeric"] += 1
        return out(".EQU %s, %s" % (sym, hexfix(val, stats)))

    # --- directives
    if low_first == "org":
        stats["ORG"] += 1
        return out(".ORG %s" % hexfix(rest, stats))
    if low_first == "end":
        stats["END"] += 1
        return emit(indent, ".END", comment)
    if low_first == "db":
        stats["DB"] += 1
        return out(".BYTE %s" % hexfix(data_operand(rest, stats), stats))
    if low_first == "dw":
        stats["DW"] += 1
        return out(".WORD %s" % hexfix(data_operand(rest, stats), stats))
    if low_first == "ds":
        stats["DS"] += 1
        return out(".SKIP %s" % hexfix(expand_dup(rest), stats))
    if low_first == "equ":
        stats["equ_nolabel"] += 1
        return out(body)

    # --- Keil pseudo-mnemonics / textbook shorthands
    if low_first == "jmp":
        stats["jmp"] += 1
        return out("ljmp %s" % rest)
    if low_first == "call":
        stats["call"] += 1
        return out("lcall %s" % rest)
    if low_first == "clra":
        stats["clra"] += 1
        return out("clr a")

    # --- plain instruction line: keep bytes, apply $ -> *, LOW/HIGH, PUSH A,
    #     letter-leading hex, hexfix, reserved renames
    conv = re.sub(r"(?<![\w'])\$(?![\w'])", "*", body)
    if conv != body:
        stats["dollar"] += 1
    conv = stack_acc(lowhigh(conv, stats), stats)
    conv = modop(conv, stats)
    return out(hexfix(hex_operand(conv, stats), stats))


def shim_source(text):
    stats = defaultdict(lambda: 0)
    out = [rewrite_line(ln, stats) for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n", dict(stats)


def classify(log):
    if "Location counter overlaps" in log:
        return "ORG_OVERLAP"
    m = re.search(r"Undefined symbol ([\w.$]+)", log)
    if m:
        return "UNDEFINED:" + m.group(1)
    m = re.search(r"Attempt to redefine symbol (\S+)", log)
    if m:
        return "SYMBOL_CLASH:" + m.group(1)
    if "Relative offset exceeds" in log:
        return "REL_RANGE"
    if "Value out of range" in log:
        return "VALUE_RANGE"
    if re.search(r'syntax error near "[^"]*,', log):
        return "UNSUPPORTED_FORM"
    if "Missing quote" in log:
        return "STRING_FORM"
    if "redefine" in log.lower():
        return "SYMBOL_CLASH"
    if "syntax error" in log.lower():
        return "SYNTAX"
    return "FAIL"


def assemble(path, workdir, timeout=90):
    try:
        p = subprocess.run(["as31", os.path.basename(path)], cwd=workdir,
                           capture_output=True, text=True, timeout=timeout, errors="replace")
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, "%s" % e, -1
    log = (p.stdout or "") + (p.stderr or "")
    ok = "aborted" not in log.lower()
    return ok, log, p.returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="build")
    ap.add_argument("--summary")
    ap.add_argument("--stats", help="write the shim stats json here")
    ap.add_argument("--report", help="write a machine-readable json report of every file")
    ap.add_argument("--gate", default="")
    ap.add_argument("--gate-list", help="file with one gated source path per line")
    ap.add_argument("--keep-hex", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    gate = {x.strip() for x in args.gate.split(",") if x.strip()}
    if args.gate_list:
        # gate.txt holds repo-relative paths, but rows are keyed by basename.
        for ln in open(args.gate_list, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                gate.add(os.path.basename(ln))
    rows, gate_fail = [], []

    for src in args.files:
        name = os.path.basename(src)
        try:
            raw = open(src, "rb").read().decode("latin-1")
        except OSError as e:
            rows.append((name, "READ_ERROR", str(e)[:40], {}, name, ""))
            continue
        shimmed, stats = shim_source(raw)
        stem = re.sub(r"\.(asm|a51)$", "", name, flags=re.I)
        dst = os.path.join(args.out, stem + ".asm")
        with open(dst, "w", newline="") as f:
            f.write(shimmed)
        ok, log, rc = assemble(dst, args.out)
        status = "PASS" if ok else classify(log)
        first = ""
        first_src = ""
        if not ok:
            for l in log.splitlines():
                if l.lower().startswith(("warning", "error")):
                    first = l
                    mn = re.search(r"line (\d+)", l)
                    if mn:
                        try:
                            lines = shimmed.splitlines()
                            k = int(mn.group(1)) - 1
                            first_src = lines[k] if 0 <= k < len(lines) else "<past EOF>"
                        except Exception:
                            pass
                    break
            detail = "%s rc=%d" % (first[:78], rc)
            # Measured on ven0m-hue's MotorControl.asm: as31 aborts yet still
            # leaves a header-only .hex behind, which sim_run.py then "ran"
            # (words=0, all-zero flash = NOPs). A failed build must not hand
            # the simulator an image, so drop the artefacts it did not earn.
            for ext in (".hex", ".lst"):
                junk = os.path.join(args.out, stem + ext)
                if os.path.exists(junk):
                    os.remove(junk)
        else:
            hx = os.path.join(args.out, stem + ".hex")
            detail = "hex=%dB" % (os.path.getsize(hx) if os.path.exists(hx) else 0)
            if not args.keep_hex:
                for ext in (".lst",):
                    p = os.path.join(args.out, stem + ext)
                    if os.path.exists(p):
                        os.remove(p)
        rows.append((name, status, detail, stats, stem + ".asm", first_src))
        if name in gate and status != "PASS":
            gate_fail.append((name, log))

    passed = sum(1 for r in rows if r[1] == "PASS")
    print("assembled %d/%d files" % (passed, len(rows)))
    for n, s, d, _, _, _ in sorted(rows):
        print("  %-64s %-15s %s" % (n[:64], s, d))
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write("\n### as31 build matrix (%d/%d PASS)\n\n| source | result | detail |\n|---|---|---|\n" % (passed, len(rows)))
            for n, s, d, _, _, _ in sorted(rows):
                f.write("| `%s` | **%s** | %s |\n" % (n, s, d.replace("|", "\\|")))
    if args.stats:
        import json
        agg = defaultdict(int)
        for r in rows:
            for k, v in r[3].items():
                agg[k] += v
        json.dump(dict(agg), open(args.stats, "w"), indent=1, sort_keys=True)
    if args.report:
        import json
        json.dump({"total": len(rows), "passed": passed,
                   "files": [{"name": n, "status": s, "detail": d, "shimmed": sh,
                              "first_bad_line": fs, "stats": st}
                             for n, s, d, st, sh, fs in rows]},
                  open(args.report, "w"), indent=1)
    seen_gate = gate & {r[0] for r in rows}
    if args.gate_list and gate and not seen_gate:
        print("GATE FAILURE: gate list has %d entries but none of those sources "
              "exist on this branch -- refusing to report a green run for a "
              "branch that was not built." % len(gate), file=sys.stderr)
        return 1
    if gate:
        print("gated sources present: %d/%d" % (len(seen_gate), len(gate)))
    if gate_fail:
        for n, log in gate_fail:
            print("GATE FAILURE in %s:\n%s" % (n, log), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
