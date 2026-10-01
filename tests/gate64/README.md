# The 64³ commit gate

Five runs of a 262144-particle cosmological box to z=0, checked against a CPU GADGET-2 reference.
**About 4 minutes total.** T44 item 4, with items 5 and 3 folded in.

```sh
tests/gate64/gate64.sh  path/to/gadget_hip      # PMGRID=64 PERIODIC=ON MAC_SPRINGEL=ON build
```

Exit 0 = pass, 1 = a check failed (the run directory is kept), 2 = cannot run (missing binary, wrong
build configuration, IC checksum mismatch).

## Why it exists

T42's missing factor `G` in the long-range kick took a day to find, because everything before this
harness was a 5–8 hour 512³ run. **Verified, not asserted:** reproducing that bug exactly
(`GADGET_HIP_PM_KICK_SCALE=0.0232519712`, an impulse 1/43.007106 of correct) is rejected by all three
quantitative checks in a 21-second run — Ekin −94.9%, σ(δ) 0.954 against 3.681.

The defects in this port are silent: the run completes, `rc=0`, plausible numbers appear. What they
do not survive is a cheap external oracle, run often.

## What it checks

| check | kind | threshold |
|---|---|---|
| Ekin at 20 epochs vs GADGET-2, both paths | **external oracle** | 2.5% |
| Ekin at 20 epochs vs this port's known-good mean, both paths | regression envelope | **1.0%** |
| σ(δ) at z=0, both paths | regression | 3.6814 ± 3% |
| no long-range force **must** change σ(δ) | must-differ | ≥20% |
| 10× long-range impulse **must** change σ(δ) | must-differ | ≥20% |
| `[PM-ENERGY]`, `[T45-AOLD-PM]` present and not OFF | refusal audit | — |
| a known-bad run **must** be rejected | self-test | — |

"Both paths" is PM-every-step and the PM cadence; both are supported and both have been broken
before.

### Two energy references, because they answer different questions

- **vs `ref/energy_cpu_gadget.txt`** — GADGET-2.0.7, an *independent code*, same ICs and parameters.
  Evidence about the port rather than about itself. Necessarily **loose**: the two codes genuinely
  disagree by up to 1.880% (every-step) and 1.534% (cadence) here, measured over 8 and 3 known-good
  runs. Every parameter has been diffed against the reference run's `parameters-usedvalues` and
  matches, so this is resolution-limited disagreement, not a setup error.
- **vs `ref/ekin_port_{every,cadence}.txt`** — the per-epoch mean of known-good runs of this port.
  A regression envelope, and it can be **tight**, because the ~1.9% offset against GADGET is a
  stable property of the configuration while a code change that moves it is the thing worth seeing.
  Observed headroom on a good run: worst 0.14–0.34% against the 1.0% tolerance.

σ(δ) is a **regression** check, not an oracle: there is no CPU z=0 snapshot for this IC (the
reference run kept only restart files), so its expected value comes from known-good runs of this
port. It is in the gate because it separates the failure that actually happened from noise by a
factor of ~240.

> **T44 said this box "reproduces CPU GADGET-2 to 0.7%". That was wrong** — taken from a partial run.
> Over all 20 epochs to z=0 it is 1.9%. A gate set at 0.7% would have failed on correct code.

## Every threshold is a measured floor

The port is **not reproducible run to run**: PM CIC mass assignment uses `atomicAdd` on float, so the
density grid differs in the last bits and the FFT spreads it across the box (tickets/T46, T16, T3).
Measured at 64³:

```
step counts:          2588 .. 2659        (four nominally identical runs)
Ekin spread at z=0:   0.42% every-step, 0.51% cadence
sigma(delta) spread:  0.31%               (five runs, 3.677 .. 3.688)
```

So: Ekin envelope 1.0% ≈ 2× the floor; σ(δ) 3% ≈ 10× its floor; must-differ ≥20% against measured
effects of 74% and 1140%.

### Must-differ checks need an effect bigger than the noise

T44 item 5 proposed asserting that configurations expected to differ actually do. With a
non-deterministic code *any* two runs differ, so the naive form proves nothing. T44's own suggested
case — quartering the PM interval, which moved the 7th significant digit — is **unusable** here,
being far below the floor. The two cases used were chosen for effects two orders of magnitude above
it.

## Three bugs this harness had while being written

Recorded because they are the same shape as the bugs it exists to catch — a check that silently
checks nothing:

1. `python3 ... | sed ... || fail` takes the exit status of **`sed`**, so the energy checks could
   never fail. Fixed with `out=$(...); rc=$?`.
2. The refusal audit used `grep -E '^\[PM-ENERGY\]'`. The port block-buffers stdout and leaves
   stderr unbuffered, so that line arrives **spliced into the middle** of a stdout line
   (`...Dlo[PM-ENERGY] long-range term...`) and the anchored pattern matched nothing — the audit
   covered one feature while appearing to cover two. Fixed by dropping the anchor *and* by requiring
   each expected line to be **present**: absence is now a failure, since it means either the feature
   did not run or the audit no longer matches it. `src/main.cpp` also now calls
   `setvbuf(stdout, NULL, _IOLBF, 0)`, which removes the splicing at the source.
3. `BoxSize` read at header offset 88 (`flag_sfr`) instead of 128, giving 0.0, putting every particle
   in one cell, and reporting σ(δ)=181 — which looks like *maximum* clustering rather than an error.
   The reader now rejects a non-positive `BoxSize`.

Hence the SELF-TEST step: the gate reproduces a known-bad configuration and fails if any check
accepts it.

## Provenance

- **IC**: `IC_ngenic64_id0` — 64³ = 262144 particles, 100 Mpc/h, z=99, made with N-GenIC from
  `ics/inputspec_IC_ngenic64.txt`. Not stored here (7.3 MB); the gate verifies its sha256 so a
  different IC cannot be silently compared against these energies. Override with `GATE64_IC=`.
  IDs are **0-based**; the CPU reference ran on the 1-based `IC_ngenic64`, the same particles
  renumbered, which GADGET-2 does not care about and the port did — ticket T43.
- **`ref/energy_cpu_gadget.txt`** — GADGET-2.0.7, `PMGRID=64`, parameters as `ref/gate64.param.in`,
  on CPU. 20 epochs at `TimeBetStatistics=0.05`. Column 1 time, column 4 Ekin.
- **`ref/ekin_port_*.txt`, `GATE64_SIGMA=3.6814`** — means over known-good runs of this port,
  2026-09-30 / 10-01, after T45.

## Which log line is compared

The port prints Ekin twice per epoch and they are **not** interchangeable:

- `iter=N : time= … Etot= … Ekin= …` — printed inside `compute_energies()`, carrying
  `Ekin_c = Ekin/a²`. **This is the one comparable to GADGET's `energy.txt`, and the one parsed.**
- `[ENERGY-EXACT] …` — the same quantity in comoving units, a factor a² smaller.

## Updating the reference values

Only when the physics is *known* to have changed, never to make a failing gate pass. Record why in
the ticket that changed it, and re-measure the noise floor at the same time: a tolerance is only
valid against the scatter it was derived from.
