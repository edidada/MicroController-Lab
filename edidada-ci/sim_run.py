#!/usr/bin/env python3
"""sim_run.py -- execute the images as31 produced on uCsim (`s51`) and report.

Measured invocation (sdcc-ucsim 4.2.0+dfsg-1, Ubuntu noble, /usr/bin/s51):

    printf 'step N\\nregs\\nquit\\n' | s51 -t 8051 file.hex
      Loading from file.hex
      9 words read from file.hex
      Stop at 0x000007: (109) stepped 228 ticks
           R0 R1 R2 R3 R4 R5 R6 R7
           08 70 d4 b2 8a 29 55 aa
      0x0007    80 fe    SJMP   0x0007
      F 0x000007
      Simulated 228 ticks (2.062e-05 sec)

Also measured, so the assertions below do not depend on them:
  * `s51 --version` is NOT an option ("invalid option -- '-'"); use `-V`-free
    probing or just the banner.
  * `-p FILE` is a pipe/port option, it does not read a command script -- the
    commands have to come in on stdin.
  * `set cpu m51` and `sw` are not valid in this build.
  * One trailing "Parser error: syntax error" line comes from the piped `quit`
    while rc is still 0 -- so success is judged on `Stop at` + `Simulated`,
    never on "no stderr".

What this proves, and what it does not: the image loads and the CPU retires
instructions at the addresses the linker put there. Whether the program is
*semantically* right (LCD shows "HELLO", motor turns clockwise) is not checked
here; the register/PC dump is printed for a human to eyeball.

Usage:
  sim_run.py --build build [--steps 400] [--timeout 60] [--report sim.json]
             [--summary $GITHUB_STEP_SUMMARY] [--must-run a.hex,b.hex]
Exit 1 only if an image in --must-run fails, or if no image ran at all while
hex files were present.
"""

import argparse
import json
import os
import re
import subprocess
import sys

RE_STEPS = re.compile(r"stepped (\d+) ticks")
RE_READ = re.compile(r"(\d+) words? read from")
RE_STOPAT = re.compile(r"Stop at (0x[0-9A-Fa-f]+)")
RE_FLINE = re.compile(r"^F (0x[0-9A-Fa-f]+)", re.M)
RE_REGS = re.compile(r"\n\s+R0 R1 R2 R3 R4 R5 R6 R7\n\s+([0-9a-f]{2} )+", re.M)
RE_ACC = re.compile(r"ACC= (0x[0-9A-Fa-f]+)")


def run_one(hexpath, steps, timeout):
    cmds = "step %d\nregs\nquit\n" % steps
    try:
        p = subprocess.run(["s51", "-t", "8051", os.path.basename(hexpath)],
                           cwd=os.path.dirname(hexpath) or ".",
                           input=cmds, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
    except subprocess.TimeoutExpired:
        return "TIMEOUT", {}
    out = (p.stdout or "") + (p.stderr or "")
    info = {}
    m = RE_READ.search(out)
    info["words"] = int(m.group(1)) if m else 0
    m = RE_STEPS.search(out)
    info["ticks"] = int(m.group(1)) if m else 0
    m = RE_STOPAT.search(out) or RE_FLINE.search(out)
    info["pc"] = m.group(1) if m else ""
    m = RE_REGS.search(out)
    info["regs"] = m.group(0).strip().replace("\n", " ") if m else ""
    if "Stop at" not in out or "Simulated" not in out:
        return "SIM_ERROR", info
    if info["words"] == 0:
        # Nothing was loaded: an image left behind by an aborted as31 run
        # (header-only .hex). Executing all-zero flash proves nothing.
        return "EMPTY_IMAGE", info
    if p.returncode != 0:
        return "SIM_RC%d" % p.returncode, info
    return "RAN", info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--report")
    ap.add_argument("--summary")
    ap.add_argument("--must-run", default="",
                    help="comma separated hex basenames that MUST run")
    args = ap.parse_args(argv)

    hexes = sorted(f for f in os.listdir(args.build) if f.endswith(".hex"))
    must = {x.strip() for x in args.must_run.split(",") if x.strip()}
    rows = []
    for h in hexes:
        st, info = run_one(os.path.join(args.build, h), args.steps, args.timeout)
        rows.append((h, st, info))
        if st != "RAN" and not info:
            print("%-40s %s" % (h[:40], st))

    ran = sum(1 for r in rows if r[1] == "RAN")
    print("simulated %d/%d images (s51 / uCsim 0.6.4, %d steps each)"
          % (ran, len(rows), args.steps))
    for h, st, info in rows:
        print("  %-44s %-9s words=%-5s pc=%-10s ticks=%s"
              % (h[:44], st, info.get("words", 0), info.get("pc", ""),
                 info.get("ticks", 0)))

    bad_must = [h for h, st, _ in rows if h in must and st != "RAN"]
    if hexes and ran == 0:
        print("FAIL: images were produced but none could be simulated", file=sys.stderr)
        return 1
    if bad_must:
        for h in bad_must:
            print("FAIL: required image %s did not run" % h, file=sys.stderr)
        return 1

    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write("\n### uCsim run matrix (%d/%d images executed, %d steps each)\n\n"
                    % (ran, len(rows), args.steps))
            if not hexes:
                f.write("_No images: this branch has no assembly sources._\n")
            else:
                f.write("| image | result | bytes loaded | final PC | ticks |\n"
                        "|---|---|---|---|---|\n")
                for h, st, info in rows:
                    f.write("| `%s` | %s | %s | %s | %s |\n"
                            % (h, st, info.get("words", 0), info.get("pc", "-"),
                               info.get("ticks", 0)))
    if args.report:
        json.dump({"images": len(rows), "ran": ran,
                   "rows": [{"hex": h, "status": st, **i} for h, st, i in rows]},
                  open(args.report, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
