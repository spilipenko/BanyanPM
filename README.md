# BanyanPM

**A GADGET-2-compatible cosmological N-body code for AMD GPUs.**

BanyanPM runs collisionless cosmological simulations on a single AMD GPU using HIP/ROCm. It reads
GADGET-2 parameter files and format-1 snapshots and writes the same, so existing setups and
downstream tools — halo finders, power-spectrum codes — work unchanged.

It is built on the [Bonsai](https://github.com/treecode/Bonsai) GPU tree code, with the physics
reimplemented to follow [GADGET-2](https://wwwmpa.mpa-garching.mpg.de/gadget/): Barnes–Hut and
Springel opening criteria, spline softening, individual timesteps on a power-of-two ladder, and a
TreePM short/long-range split.

A banyan grows one crown on thousands of trunks from a single root system. `PM` is the particle
mesh.

> ### ⚠️ This code was written by an AI and has not been reviewed by a human
>
> Every line was written by **Claude (Opus, Anthropic)** working under the direction of
> [@spilipenko](https://github.com/spilipenko), through a long audit of the reference
> implementation. **No human has read the source.** The validation below is real and reproducible,
> but it is *behavioural* — it demonstrates that the code reproduces GADGET-2's results on the runs
> that were tested, not that the code is correct.
>
> Treat results as unverified. If you use this for science, check it independently first.

---

## Scope

Dark matter only. **There is no SPH, no gas, no cooling and no star formation**, and no MPI — one
GPU, one process. Gas-related parameter keys are parsed and ignored.

## Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPERIODIC=ON \
  -DGADGET_HIP_PMGRID=512 \
  -DMAC_SPRINGEL=ON
make -C build -j12 gadget_hip
```

`-DCMAKE_BUILD_TYPE=Release` is not optional in practice — omitting it produces a correct but far
slower binary, which has twice been mistaken for a physics regression. `MAC_SPRINGEL` must agree
with `TypeOfOpeningCriterion` in the parameter file; the binary refuses a mismatch rather than
silently substituting the criterion it was built with.

## Run

```bash
./build/gadget_hip --param run.param                                   # see examples/z0.param
./build/gadget_hip --param run.param --restart-file ckpt.bin           # with checkpointing
./build/gadget_hip --param run.param --restart-file ckpt.bin --resume  # continue
```

Writing a file named `stop` into `OutputDir` ends the run cleanly at the next step.

Output: `snapshot_NNN` (GADGET-2 format 1), `info.txt` (GADGET's per-step line), and `cpu.txt`
(cumulative per-phase timings in GADGET-2's own 18-column format, so it diffs directly against a
GADGET-2 run's file). Energies go to stdout; **no `energy.txt` is written**.

Full documentation of every build option, every `.param` key — including which are honoured and
which are parsed and ignored — and the differences from GADGET-2 is in the sections below and in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for the runtime switches.

### GPU resets on a machine with a display attached — fixed, but read this if you see it

On a GPU that also drives a desktop, runs used to die within a few steps with
`HW Exception by GPU node-1 ... reason: GPU Hang`, and `dmesg` showed repeated GPU resets. This was
**not** a fault in the physics — it was the display stack being starved.

The tree walk was issued as **a single kernel dispatch per step covering all active groups**, and at
512³ with the Springel criterion that dispatch ran for **~7 seconds**. While it occupied the GPU
the compositor could not complete its work, so the graphics ring hit `amdgpu.lockup_timeout`
(default **2000 ms**) and the driver reset the device. The simulation was collateral damage. The
giveaway in `dmesg` is a **graphics** ring timeout immediately before each reset:

```
amdgpu: ring gfx_0.0.0 timeout, signaled seq=31272, emitted seq=31274
```

It looked like a code bug because it was reproducible, always at the same step, and specific to the
Springel criterion — but only because that criterion produces the long dispatch. Geometric opening,
or a loose `ErrTolForceAcc`, shortens the dispatch enough to fit inside the watchdog and appears to
"work", which sent the diagnosis down a blind alley for a while.

**The walk is now split into chunks of active groups**, sized adaptively from the previous step's
measured walk time, so no single launch monopolises the device. The longest dispatch at 512³ fell
from ~7300 ms to ~540 ms. If you are on an older build, or still see resets, either workaround
below is sufficient:

```bash
# 1. run without a desktop -- nothing is submitting graphics work to starve
sudo systemctl isolate multi-user.target        # ... run ...
sudo systemctl isolate graphical.target

# 2. give the rings enough headroom (Fedora; merges with existing kernel args, reboot required)
sudo grubby --update-kernel=ALL --args="amdgpu.lockup_timeout=20000,20000,20000,20000"
```

Verified: with the timeout raised, 512³ Springel at `ErrTolForceAcc=0.002` runs with a desktop
session active and zero GPU resets. Expect the desktop to be sluggish during dense steps — the GPU
really is busy.

---

## Validation

Against a CPU GADGET-2 reference run on identical initial conditions: 512³ particles, 100 Mpc/h
box, ΛCDM (Ωm = 0.307, ΩΛ = 0.693, h = 0.678), z = 99 → 0, Springel opening criterion with
`ErrTolForceAcc = 0.002`, PMGRID 512.

**Energy.** Kinetic energy at all 20 statistics epochs, worst deviation **+0.145%** (at a = 0.26),
with no trend — +0.101% at a = 0.91 and +0.064% at a = 0.96.

**Halo mass function** at z = 0, both catalogues built by the same Rockstar binary with the same
configuration:

| | CPU GADGET-2 | BanyanPM |
|---|---|---|
| haloes | 309,610 | 306,957 (−0.86%) |
| N(> 10¹³·⁵) | 119 | 119 |
| N(> 10¹⁴) | 24 | 24 |
| N(> 10¹⁴·⁵) | 4 | 4 |
| mass-function ratio, median over 20 bins | — | 0.9984 |

**Individual haloes.** Matching the 100 most massive one-to-one: median mass difference **0.26%**,
median position offset **0.0089 Mpc/h** — about 1/22 of the mean interparticle spacing. 88 of 100
agree within 1%; the handful of percent-level outliers are merger-timing differences, which is
ordinary between two independent codes.

**Force accuracy.** 0.39% rms on the fine-particle force against the same reference.

**Halo mass function across redshift.** A second run of the same ICs on the integer-timeline build,
compared against the CPU reference at all seven epochs where the two output schedules coincide.
Rockstar catalogues on both sides, 20 bins from the 20-particle mass (1.27e10 M⊙/h) upward; haloes
are matched only when a counterpart agrees in **both** position and mass:

| z | N(>20 p) GPU/CPU | N(>100 p) GPU/CPU | matched | median offset | mass bias |
|---|---|---|---|---|---|
| 6.00 | 0.9989 | 1.0199 | 98.86% | 0.0144 | +0.79% |
| 4.39 | 0.9934 | 1.0060 | 98.64% | 0.0151 | +0.81% |
| 3.14 | 0.9918 | 1.0051 | 98.32% | 0.0156 | +0.64% |
| 2.19 | 0.9906 | 1.0008 | 98.21% | 0.0153 | +0.44% |
| 1.45 | 0.9902 | 0.9998 | 98.32% | 0.0154 | +0.30% |
| 0.89 | 0.9908 | 0.9990 | 98.37% | 0.0152 | +0.20% |
| 0.45 | 0.9907 | 0.9999 | 98.53% | 0.0152 | +0.23% |

![Halo mass function, CPU GADGET-2 vs BanyanPM at seven epochs](docs/massfunc-multi-epoch.png)

Offsets in Mpc/h. No mass bin deviates by more than 3σ Poisson at any epoch, and at z = 0.45 the
eight most massive bins reproduce the CPU counts *exactly* (300, 99, 43, 22, 8, 4, 3). The median
halo offset is flat at **15 kpc/h** across a factor of five in expansion factor — it does not grow,
even though individual particle trajectories diverge chaotically (rms 6e-2 Mpc/h between two runs
of the same binary). The residual ~0.9% deficit is confined to the 20–100 particle resolution
limit; above 100 particles the catalogues agree to 3 haloes in 58,240.

**Restart.** Stopping a run and resuming from a checkpoint reproduces the uninterrupted run to
within the PM-atomics noise floor — verified by comparing the stop/resume divergence against a
control of two *uninterrupted* runs of the same binary, which agree no better (median 6.98e-4 vs
6.71e-4 Mpc/h, rms 6.02e-2 vs 6.02e-2).

Note that this validates dark-matter-only, periodic, TreePM runs at one resolution. Nothing else
has been tested.

## Performance

Same problem, same machine class — 512³ to z = 0, AMD Strix Halo (gfx1151) against 16 CPU cores:

| | wall clock | per step |
|---|---|---|
| CPU GADGET-2, 16 cores | 48.3 h | 18.20 s |
| BanyanPM | **8.7 h** | **3.38 s** |
| | **5.6× faster** | |

Both runs were 9,200–9,600 steps. The gain is concentrated at late times, where clustering makes
the mesh assignment expensive; early steps are roughly at parity.

That wall clock was measured on the float-timeline build. The integer timeline has not been timed
over a full z = 0 run, so rather than extrapolate one, here is what *is* measured: per-step cost at
matched expansion factor (a = 0.733), where the current build sits at **3.52 s/step** against
**3.39–3.48 s/step** for the build the table above describes — parity, inside the spread between
two different seeds of the older build.

A per-phase breakdown of any run is in its `cpu.txt`. On a late step the cost is dominated by the
tree walk on dense steps (~47%) and by the tree rebuild otherwise (~40%) — the long-range force is
about 20%.

---

## Differences from GADGET-2

Read this before trusting a comparison. Items marked **open** are known gaps, not design choices.

- **Dark matter only**, single GPU, single process.
- **Monopole tree**, matching GADGET-2. Bonsai's quadrupole correction is available but off.
- **The long-range force is recomputed every step** and folded into the active particles'
  acceleration. GADGET evaluates PM only on PM steps and applies it as a separate kick over the PM
  interval to *every* particle. **Open** — a version that merely skipped PM recomputation cost 11%
  in kinetic energy and was withdrawn.
- **The tree is rebuilt every step.** GADGET rebuilds at `TreeDomainUpdateFrequency` and drifts node
  centres of mass in between. **Open**, and the largest remaining cost.
- **Integer timeline**, as in GADGET-2. Step boundaries are stored as integer ticks on a 2²⁸
  lattice spanning `log(a)` from `TimeBegin` to `TimeMax`, so a sync point is exact rather than the
  nearest representable float32, and the step ladder is a true power-of-two hierarchy. This
  replaced a float32 clock whose rounding produced phantom sub-ULP steps — on a zoom test the step
  count fell from 68,780 to ~2,710 against GADGET-2's 2,690. Restart files carry a version tag and
  a build refuses to read the older float layout rather than reinterpret it.
- **Not bit-reproducible.** PM mass assignment uses atomics; the floor is ~1e-8 relative. Establish
  that noise floor by running the same binary twice before attributing any difference to a change.
- **I/O**: GADGET-2 format 1 only; one file per snapshot. `energy.txt` and `timings.txt` are not
  written — energies go to stdout instead. `info.txt` and `cpu.txt` *are* written, both in
  GADGET's own format. Checkpointing uses its own format, not GADGET restart files.
- Unsupported parameter keys: `TreeDomainUpdateFrequency`, `TypeOfTimestepCriterion`,
  `OutputListOn`/`OutputListFilename`, `NumFilesPerSnapshot`, and all SPH keys.

---

## Licence and attribution

**GPL-3.0-or-later.** See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

- **[Bonsai](https://github.com/treecode/Bonsai)** (Bédorf, Gaburov & Portegies Zwart) is
  Apache-2.0 and is the code this is built on — tree construction, the warp-cooperative traversal,
  the device memory layer and the program skeleton. Apache-2.0 material may be combined into a
  GPLv3 work, which is why GPLv3 and not GPLv2.
- **[GADGET-2](https://wwwmpa.mpa-garching.mpg.de/gadget/)** (Volker Springel) is GPL v2 **or
  later** and is the reference implementation the physics follows and is validated against. Source
  locations are cited throughout the code comments.

**Volker Springel and the Bonsai authors are not involved in this project and do not endorse it.**
The name deliberately avoids both projects' names for that reason. "GADGET-2-compatible" is a
statement about file formats and reproduced results, nothing more.

## Authorship

Written by **Claude (Opus, Anthropic)**, directed by [@spilipenko](https://github.com/spilipenko).

The work was done as a contract-based audit: the port's behaviour was compared against GADGET-2
clause by clause, each discrepancy recorded as a numbered contract or ticket, and each fix required
a positive control — an instrument that cannot be made to fail has not been validated. Several
findings are recorded with the failed reasoning that preceded them, because the failure mode is
usually the reusable part.

See the disclaimer at the top: **no human has read this source code.**
