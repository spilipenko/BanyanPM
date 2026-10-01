#!/usr/bin/env python3
"""sigma(delta) on a grid from a GADGET format-1 snapshot -- a structure-growth check.

NOT an external oracle. There is no CPU GADGET z=0 snapshot for this IC (the reference run kept only
restart files), so the expected value comes from a known-good run of this port. Treat it as a
REGRESSION check: it catches "structure stopped forming" or "structure changed materially", which is
what a broken or missing long-range force looks like. The external oracle is check_energy.py.

Measured discrimination at 64^3, z=0:
    long-range force present (5 runs):   sigma = 3.677 .. 3.688    (0.31% scatter)
    long-range force absent:             sigma = 0.946             (-74%)
so this separates the failure that cost a day (T42's missing G) from noise by a factor of ~240.

Modes:
    --expect V --tol P     sigma must be within P percent of V
    --differ-from FILE --min-diff P    sigma must differ from FILE's by at least P percent
"""
import struct, sys, argparse
import numpy as np

def read_pos(path):
    with open(path, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        h = f.read(n); f.read(4)
        npart = struct.unpack('6i', h[:24])
        time = struct.unpack('d', h[72:80])[0]
        # BoxSize sits at offset 128: npart[6]=24 + mass[6]=48 + time=8 + redshift=8
        # + flag_sfr/flag_feedback=8 + npartTotal[6]=24 + flag_cooling/num_files=8.
        # Reading it at 88 (flag_sfr) yields 0.0 and silently puts every particle in one cell --
        # which looks like "maximum clustering", not like an error.
        boxsize = struct.unpack('d', h[128:136])[0]
        if not (boxsize > 0):
            sys.exit("  FAIL  BoxSize=%r in %s -- snapshot header not as expected" % (boxsize, path))
        total = sum(npart)
        n = struct.unpack('i', f.read(4))[0]
        if n != total * 3 * 4:
            sys.exit("  FAIL  POS block is %d bytes, expected %d" % (n, total * 3 * 4))
        pos = np.frombuffer(f.read(n), dtype=np.float32).reshape(total, 3)
    return pos, boxsize, time, total

def sigma_delta(path, ngrid):
    pos, box, time, n = read_pos(path)
    idx = np.floor(pos / box * ngrid).astype(np.int64) % ngrid
    flat = (idx[:, 0] * ngrid + idx[:, 1]) * ngrid + idx[:, 2]
    counts = np.bincount(flat, minlength=ngrid ** 3).astype(np.float64)
    if counts.sum() <= 0: sys.exit("  FAIL  no particles binned from %s" % path)
    return float((counts / counts.mean() - 1.0).std()), time, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('snapshot')
    ap.add_argument('--ngrid', type=int, default=32)
    ap.add_argument('--expect', type=float)
    ap.add_argument('--tol', type=float, default=3.0, help='percent')
    ap.add_argument('--differ-from')
    ap.add_argument('--min-diff', type=float, default=20.0, help='percent')
    a = ap.parse_args()

    s, time, n = sigma_delta(a.snapshot, a.ngrid)
    z = 1.0 / time - 1.0 if time > 0 else float('nan')
    print("      sigma(delta)=%.5f  (z=%.3f, N=%d, ngrid=%d)" % (s, z, n, a.ngrid))
    if abs(z) > 0.01:
        print("  FAIL  snapshot is at z=%.3f, not z=0 -- the run did not finish" % z); return 1
    if a.differ_from:
        t, _, _ = sigma_delta(a.differ_from, a.ngrid)
        d = 100.0 * abs(s - t) / t
        if d < a.min_diff:
            print("  FAIL  must differ by >=%.1f%% from %s (sigma %.5f vs %.5f = %.2f%%)"
                  % (a.min_diff, a.differ_from, s, t, d)); return 1
        print("  PASS  differs by %.1f%% from %s (sigma %.5f vs %.5f)" % (d, a.differ_from, s, t))
        return 0
    if a.expect is not None:
        d = 100.0 * abs(s - a.expect) / a.expect
        if d > a.tol:
            print("  FAIL  sigma %.5f is %.2f%% from expected %.5f (tol %.1f%%)"
                  % (s, d, a.expect, a.tol)); return 1
        print("  PASS  sigma %.5f within %.2f%% of %.5f" % (s, d, a.expect))
    return 0

sys.exit(main())
