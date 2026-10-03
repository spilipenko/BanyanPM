#!/usr/bin/env python3
"""Periodic centre of one particle type in a Gadget-2 snapshot, by the same minimum-image method
pm_zoom_periodic_center uses -- so the C++ and this agree by construction, and a disagreement
between the IC and the written snapshot means the recentring shift was not undone correctly.

Subsamples, because the whole point is to detect an O(Mpc/h) frame error, not to measure the centre
to machine precision."""
import struct, sys, numpy as np

def read_header(f):
    f.seek(0)
    blk = struct.unpack('i', f.read(4))[0]
    h = f.read(256)
    struct.unpack('i', f.read(4))
    npart = struct.unpack('6i', h[0:24])
    mass  = struct.unpack('6d', h[24:72])
    time  = struct.unpack('d', h[72:80])[0]
    box   = struct.unpack('d', h[128:136])[0]
    return npart, mass, time, box, blk

def read_pos_for_type(path, want_type, stride):
    with open(path, 'rb') as f:
        npart, mass, time, box, _ = read_header(f)
        ntot = sum(npart)
        # POS block follows the header block
        blk = struct.unpack('i', f.read(4))[0]
        assert blk == 3 * 4 * ntot, "unexpected POS block size %d for ntot %d" % (blk, ntot)
        pos_start = f.tell()
        before = sum(npart[:want_type])
        count  = npart[want_type]
        out = []
        for i in range(0, count, stride):
            f.seek(pos_start + (before + i) * 12)
            out.append(struct.unpack('3f', f.read(12)))
        return np.array(out, dtype=np.float64), box, time, npart

def periodic_centre(p, box):
    x0 = p[0]
    d = p - x0
    d -= box * np.round(d / box)        # minimum image into [-L/2, L/2)
    lo, hi = d.min(axis=0), d.max(axis=0)
    extent = hi - lo
    c = (x0 + 0.5 * (lo + hi)) % box
    return c, extent

if __name__ == '__main__':
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 997
    for path in sys.argv[2:]:
        p, box, time, npart = read_pos_for_type(path, 1, stride)
        c, e = periodic_centre(p, box)
        print("%-62s a=%.6f box=%g  n_type1=%d sampled=%d" % (path, time, box, npart[1], len(p)))
        print("    type-1 periodic centre = (%.5f, %.5f, %.5f)   extent = (%.4f, %.4f, %.4f)"
              % (c[0], c[1], c[2], e[0], e[1], e[2]))
