#!/usr/bin/env python3
"""Keep include/devFunctionDefinitions.h in step with the kernel definitions.

    tools/check_kernel_decls.py            report divergences, exit 1 if any
    tools/check_kernel_decls.py --write    regenerate the header from the definitions

Run from runtime/.

WHY THIS EXISTS. The declarations in devFunctionDefinitions.h are used only for their ADDRESS --
load_kernels.cpp passes &kernel to hipLaunchKernel -- so a wrong signature there compiled, linked
and ran silently for years. When the comparison was first switched on (2026-10-01, T44 item 2) it
found 25 divergences in 44 kernels, including:

    compute_dt            declared with 12 parameters, defined with 25
    correct_particles     13 against 22
    predict_particles      9 against 16
    get_Tnext             float2 *time where the Phase 2 integer timeline made it int2 *
    setActiveGroups       a float second parameter that is an int
    compute_leaf          uint* against ulonglong1*, const real against const float
    three scanKernels     setupParams2, a layout-compatible DUPLICATE of setupParams

None was an active bug, because nothing ever read these types. That is also what made the header
useless as documentation and unusable for type-checking set_args.

Each kernel-defining .cu now includes the header, so the compiler compares declaration against
definition and a divergence is a build error. This tool is for regenerating after adding a kernel;
the compiler is the enforcement.

NOTES ON THE EXTRACTION
  - Comments are stripped BEFORE splitting on commas. A comma inside a comment
    ("// Ti_current, in ticks") otherwise splits one parameter into two and leaves prose in a type.
  - real/real4 are emitted as float/float4. They are `typedef float real; typedef float4 real4;` in
    five headers, and plain `real` is AMBIGUOUS inside compute_propertiesD.cu (which is why that
    file writes ::real). The qualified form must be replaced literally: \\b does not match before
    the ':' in '::real', so r'\\b(?:::)?real\\b' matches only the 'real' and leaves '::float'.
  - dev_approximate_gravity/_let are macro-generated and are carried over by hand.
"""
import re, os, io, sys

CU_DIR = 'CUDAkernels'
HDR    = 'include/devFunctionDefinitions.h'
KEPT   = ('dev_approximate_gravity', 'dev_approximate_gravity_let')

def strip_comments(s):
    s = re.sub(r'/\*.*?\*/', ' ', s, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', s)

def balanced(s, i):
    d, j = 0, i
    while j < len(s):
        if s[j] == '(': d += 1
        elif s[j] == ')':
            d -= 1
            if d == 0: return s[i+1:j], j+1
        j += 1
    raise ValueError('unbalanced parentheses')

def split_top(p):
    d, cur, out = 0, '', []
    for ch in p:
        if ch in '<([': d += 1
        elif ch in '>)]': d -= 1
        if ch == ',' and d == 0: out.append(cur); cur = ''
        else: cur += ch
    out.append(cur)
    return out

def canon(t):
    t = t.replace('::real4', 'float4').replace('::real', 'float')
    t = re.sub(r'\breal4\b', 'float4', t)
    t = re.sub(r'\breal\b',  'float',  t)
    return t

def params_of(inner):
    parts = [re.sub(r'\s+', ' ', x).strip() for x in split_top(strip_comments(inner))]
    return ', '.join(canon(p) for p in parts if p)

def typeonly(decl):
    d = re.sub(r'\s+', ' ', decl.strip())
    m = re.match(r'^(.*?)(\b[A-Za-z_]\w*\s*)(\[\s*\d*\s*\])?$', d)
    if not m or not m.group(1).strip(): return d.replace(' ', '')
    return (m.group(1).strip() + (m.group(3) or '')).replace(' ', '')

def key_of(params):
    return tuple(typeonly(x) for x in split_top(params) if x.strip())

def find_defs():
    defs = {}
    for fn in sorted(os.listdir(CU_DIR)):
        if not fn.endswith('.cu'): continue
        src = io.open(os.path.join(CU_DIR, fn), encoding='utf-8', errors='surrogateescape').read()
        for pat in (r'KERNEL_DECLARE\s*\(\s*(\w+)\s*\)\s*\(',
                    r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\('):
            for m in re.finditer(pat, src):
                inner, _ = balanced(src, m.end()-1)
                defs[m.group(1)] = (fn, params_of(inner))
    return defs

def find_decls():
    src = io.open(HDR, encoding='utf-8', errors='surrogateescape').read()
    out = {}
    for m in re.finditer(r'extern\s+"C"\s+\w+\s+\(?\s*(\w+)\s*\)?\s*\(', src):
        inner, _ = balanced(src, m.end()-1)
        out[m.group(1)] = params_of(inner)
    return out

def preamble():
    src = io.open(HDR, encoding='utf-8', errors='surrogateescape').read()
    i = src.find('// Macro-generated')
    if i < 0: i = src.find('extern "C"')
    return src[:i].rstrip()

def write_header(defs):
    old = find_decls()
    kept = ['extern "C" void  (%s)(%s);' % (k, old[k]) for k in KEPT if k in old]
    out = [preamble(), '',
           '// Macro-generated in dev_approximate_gravity_warp_new.cu; not extracted by the tool.'] \
          + kept + ['']
    for name in sorted(defs):
        fn, params = defs[name]
        out.append('extern "C" void  (%s)(%s);   // %s' % (name, params, fn))
    out += ['', '#endif', '']
    io.open(HDR, 'w', encoding='utf-8', errors='surrogateescape').write('\n'.join(out))
    print('wrote %s: %d declarations (+%d kept by hand)' % (HDR, len(defs), len(kept)))

def main():
    if not os.path.isdir(CU_DIR):
        sys.exit('run this from runtime/ (no %s here)' % CU_DIR)
    defs = find_defs()
    if '--write' in sys.argv:
        write_header(defs); return 0
    decls, bad = find_decls(), 0
    for name in sorted(defs):
        fn, params = defs[name]
        if name not in decls:
            print('  MISSING DECL  %-40s (%s)' % (name, fn)); bad += 1; continue
        if key_of(params) != key_of(decls[name]):
            bad += 1
            print('  MISMATCH      %-40s (%s)' % (name, fn))
            print('      header: %s' % ', '.join(key_of(decls[name])))
            print('      actual: %s' % ', '.join(key_of(params)))
    for name in sorted(decls):
        if name not in defs and name not in KEPT:
            print('  ORPHAN DECL   %-40s' % name)
    print('\n%d divergence(s). The compiler enforces this too: every kernel-defining .cu includes\n'
          'the header, so a mismatch is a build error. Use --write to regenerate.' % bad)
    return 1 if bad else 0

sys.exit(main())
