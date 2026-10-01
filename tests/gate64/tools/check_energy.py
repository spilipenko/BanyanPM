#!/usr/bin/env python3
"""Compare a run's Ekin at every statistics epoch against a reference.

Used twice per run, with two different references, because they answer different questions:

  1. vs `ref/energy_cpu_gadget.txt` -- GADGET-2.0.7, an INDEPENDENT code, same ICs and parameters.
     This is the external oracle: it is evidence about the port rather than about itself. It is
     necessarily LOOSE, because the port and GADGET genuinely disagree by up to 1.9% at 64^3 (both
     codes are resolution-limited here; every parameter has been diffed and matches).

  2. vs `ref/ekin_port_{every,cadence}.txt` -- the mean of known-good runs OF THIS PORT, per epoch.
     This is a regression envelope, and it can be TIGHT, because the 1.9% offset against GADGET is a
     stable property of the configuration while a code change that alters it is what we want to see.

Both tolerances are measured floors, not preferences. The port is not reproducible run to run
(0.42-0.51% spread in Ekin at z=0, from PM CIC atomics -- tickets/T46), so a tolerance tighter than
that fails at random, which is worse than no gate.
"""
import re, sys, argparse

def gpu_epochs(path):
    """(a, Ekin) from the port's own log, first occurrence per epoch.

    The parsed line is the one printed inside compute_energies() -- 'iter=N : time= ... Etot= ...
    Ekin= ...' -- whose Ekin is Ekin_c = Ekin/a^2 and therefore directly comparable to GADGET's
    energy.txt. The [ENERGY-EXACT] line is the same quantity in comoving units; do not mix them.
    """
    out, seen = [], set()
    pat = re.compile(r'time= ([0-9.]+) +Etot= [0-9.eE+-]+ +Ekin= ([0-9.eE+-]+) +Epot=')
    with open(path, errors='replace') as f:
        for line in f:
            m = pat.search(line)
            if m and m.group(1) not in seen:
                seen.add(m.group(1)); out.append((float(m.group(1)), float(m.group(2))))
    return out

def ref_epochs(path):
    """Reference (a, Ekin) pairs, from either reference format.

    - GADGET-2 `energy.txt`: >=4 columns, time in 1, Ekin in 4.
    - this gate's port envelope: 2 columns, a and Ekin.

    Both carry Ekin in the same units (GADGET's Ekin, i.e. the port's Ekin/a^2), which is the whole
    reason one comparator can serve both.
    """
    out = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith('#'): continue
            c = line.split()
            try:
                if len(c) >= 4:   out.append((float(c[0]), float(c[3])))
                elif len(c) == 2: out.append((float(c[0]), float(c[1])))
            except ValueError: pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log'); ap.add_argument('reference')
    ap.add_argument('--tol', type=float, default=0.5, help='percent')
    ap.add_argument('--min-epochs', type=int, default=18)
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    gpu, cpu = gpu_epochs(a.log), ref_epochs(a.reference)
    if not gpu: print("  FAIL  no energy epochs in %s" % a.log); return 1
    matched, worst, worst_a, fails = 0, 0.0, 0.0, []
    for ca, ce in cpu:
        if ce <= 0: continue
        ga, ge = min(gpu, key=lambda t: abs(t[0] - ca))
        if abs(ga - ca) > 0.002: continue
        d = 100.0 * (ge - ce) / ce
        matched += 1
        if abs(d) > abs(worst): worst, worst_a = d, ca
        if abs(d) > a.tol: fails.append((ca, ce, ge, d))
        if not a.quiet: print("      a=%6.4f  CPU %12.6e  GPU %12.6e  %+7.3f%%" % (ca, ce, ge, d))
    if matched < a.min_epochs:
        print("  FAIL  only %d epochs matched (need %d) -- did the run reach z=0?" % (matched, a.min_epochs))
        return 1
    if fails:
        print("  FAIL  %d/%d epochs outside +-%.2f%%; worst %+.3f%% at a=%.4f"
              % (len(fails), matched, a.tol, worst, worst_a))
        return 1
    print("  PASS  %d epochs within +-%.2f%%; worst %+.3f%% at a=%.4f" % (matched, a.tol, worst, worst_a))
    return 0

sys.exit(main())
