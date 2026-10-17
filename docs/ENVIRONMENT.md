# Runtime environment variables

Compile-time options are in [`COMPILE_OPTIONS.md`](COMPILE_OPTIONS.md); parameter-file keys are in
[`../examples/annotated.param`](../examples/annotated.param). This page covers the variables read
from the environment at run time.

Three things to know before the tables:

- **How a value is read is not uniform.** Some variables are *presence-gated*: they take effect as
  soon as they exist in the environment, whatever their value — `VAR=0` enables them just like
  `VAR=1`. Others are parsed with `atoi`/`atof`, so `VAR=0` genuinely means off. The *How it is
  read* column says which, for every variable. Do not assume `=0` disables something.
- **Diagnostic vs. result-changing.** Variables marked **diagnostic** do not alter the trajectory.
  They cost wall-clock time — some add device synchronisations and so perturb kernel overlap, some
  copy buffers back to the host every step — but the physics is unchanged. Variables marked
  **changes results** alter what the run computes.
- Everything is read once, on first use, and cached. Changing a variable mid-run is not possible,
  and these are read from the environment only — there is no parameter-file equivalent.

The code also contains internal bisection and instrumentation switches, named after the
investigation that introduced them. They are not part of the supported interface, are off (or
already at their intended setting) by default, and are not documented here.

---

## Long-range force cadence

These exist only in a build configured with `PERIODIC` and `GADGET_HIP_PMGRID` — without the
particle-mesh solver there is no long-range force to schedule.

| Variable | Default | How it is read | Effect |
|---|---|---|---|
| `GADGET_HIP_PM_CADENCE` | **on** | `atoi() != 0` | **Changes results.** Solve the particle mesh on GADGET's own PM cadence — the largest power-of-two tick count on the integer timeline not exceeding `dt_displacement` — rather than every step. The stored long-range force is then applied as a separate kick spanning the PM interval, midpoint to midpoint, to every particle exactly once per interval, which is how GADGET-2 applies it (`advance_and_find_timesteps`). Set `GADGET_HIP_PM_CADENCE=0` to recompute the mesh every step instead; that is the more accurate mode and costs roughly 14% of wall clock in PM. |
| `GADGET_HIP_PM_EVERY_STEP` | unset | **presence** | **Changes results.** Force the mesh to be recomputed every step. Equivalent to `GADGET_HIP_PM_CADENCE=0`. Presence-gated: `GADGET_HIP_PM_EVERY_STEP=0` still selects every-step. |
| `GADGET_HIP_PM_INTERVAL_SHIFT` | `0` | `atoi()` | **Changes results.** Shorten the PM interval by a factor `2^k`. Only meaningful while the cadence is active. The long-range kick is a midpoint rule, so its deviation from an every-step run falls roughly fourfold per halving of the interval; this is the knob for checking that convergence, and for buying accuracy back at a known cost. |

The cadence engages only when the run can support it: comoving integration must be on (the kick
needs the drift and kick factor tables) and particle IDs must be dense over `0 .. N-1` (the stored
force is kept in ID space so that it survives the per-step spatial re-sort). If either condition
fails, the run falls back to recomputing the mesh every step regardless of these variables. Which
mode is actually in effect is printed once, at the first PM step, on `stderr`.

## Short-range cutoff geometry

| Variable | Default | How it is read | Effect |
|---|---|---|---|
| `GADGET_HIP_T55_RCUT_MODE` | `1` | `atoi()` | **Changes results.** Geometry used to reject tree nodes beyond the short-range cutoff `Rcut`. `1` — per-axis gap against `Rcut`, which is GADGET's own test, and the default. `2` — per-axis gap against `6*Asmth`, the reach of the short-range force table itself; a guaranteed superset of everything that can contribute, and correspondingly more expensive in the walk. `0` — Euclidean gap against `Rcut`, a sphere inscribed in GADGET's box, which discards short-range contributions GADGET keeps. When set, the selected mode is echoed on `stderr`. |

## Memory and problem size

| Variable | Default | How it is read | Effect |
|---|---|---|---|
| `GADGET_HIP_UNIFIED_MEM` | `1` | `atoi() != 0` | Alias pinned host buffers as device memory instead of allocating a separate device copy, roughly halving the per-particle footprint. Read **only** in a binary configured with `-DGADGET_HIP_UNIFIED_MEM=ON`; in any other build it is ignored and the separate device copy is always made. Set `0` to turn the aliasing off at run time, without a rebuild. Correctness is unaffected either way; aliased buffers run at about 0.94–0.99 of device-memory bandwidth depending on the access pattern, so write-heavy kernels pay a few percent. |
| `GADGET_HIP_UNIFIED_ARRAYS` | all bits set | `strtoll()`, base-prefixed | Bitmask selecting which of the aliasable buffers actually alias, when `GADGET_HIP_UNIFIED_MEM` is on. The default enables every one. Its use is to narrow the aliasing to a subset — either to recover a run if one buffer behaves badly on unfamiliar hardware, or to attribute a cost or a result difference to a specific buffer. The bit assignments are the `AliasId` enum in `include/my_cuda_rt.h`. |
| `GADGET_HIP_TREE_ALLOC_FACTOR` | unset | `atof()`, used if `> 0` | Override the tree allocation factor — the headroom multiplier on the node arrays. When unset, the parameter file's `TreeAllocFactor` is used if positive, otherwise the built-in default of `0.8`. Any effective value below `0.05` is clamped up to `0.05`. This is the only way to set the factor for runs started without a parameter file (`--plummer`, `--direct` and the other built-in generators). |

## Tree-walk launch tuning

Neither chunking nor the launch overrides changes the set of interactions evaluated. Chunking
changes only the order in which groups are dispatched, so results move by the same floating-point
summation order that the walk's atomics already introduce.

| Variable | Default | How it is read | Effect |
|---|---|---|---|
| `GADGET_HIP_WALK_CHUNKS` | unset (adaptive) | `atoi()` | Number of dispatches the tree walk is split into. Unset or negative sizes the split adaptively from the previous step's measured walk time; `0` issues the walk as a single dispatch; a positive value forces exactly that many chunks. Splitting the walk bounds how long any one kernel launch occupies the GPU, which matters where a long-running kernel can trip a device watchdog. The first walk of a run has no measurement to work from and starts at 16 chunks. |
| `GADGET_HIP_WALK_CHUNK_MS` | `300.0` | `atof()` | Target wall time per chunk, in milliseconds, for the adaptive sizing above. Ignored when `GADGET_HIP_WALK_CHUNKS` forces a count. |
| `GADGET_HIP_LAUNCH_THREADS` | `0` | `atoi()`, used if `> 0` | Override the thread-block size of the tree-walk kernel. `0` keeps the compiled-in geometry. **The kernel's warp-cooperative reductions assume a full 32-lane warp and are not templated on lane count**, so values that do not provide one are not safe. |
| `GADGET_HIP_LAUNCH_BLOCKS` | `0` | `atoi()`, used if `> 0` | Override the grid size of the tree-walk kernel. `0` keeps the compiled-in value. |

Setting either launch override prints the resulting geometry, alongside the defaults it replaced,
on `stderr`.

## Fallbacks and compatibility

Not needed in normal use. Each exists because the behaviour it selects is occasionally the one you
want — either to work around a device-side problem, or to reproduce output that another tool
expects.

| Variable | Default | How it is read | Effect |
|---|---|---|---|
| `GADGET_HIP_PM_CIC` | `1` | `atoi()` | Thread-to-particle mapping in the particle-mesh mass assignment. `1`, the default, is transposed so that Peano-Hilbert-sorted neighbours do not all target the same few grid cells; `0` is the direct mapping, where a whole wavefront collides on eight addresses and the atomics serialise. Both sum the same set of contributions, but in a different order, so results differ in the last bits — not bit-identical. There is no reason to set `0` except to reproduce output from a run that did. |
| `GADGET_HIP_NO_BOX_WRAP` | unset | **presence** | **Changes results, and breaks downstream tools.** Stops mapping particle coordinates back into the periodic box each step, so positions accumulate their unwrapped trajectory. Snapshots then carry coordinates outside `[0, BoxSize)`, and any analysis assuming periodic coordinates — halo finders included — gives wrong answers without complaining. Read only in a `PERIODIC` build. |
| `GADGET_HIP_T35_NO_DEVICE_DT`, `GADGET_HIP_T35_NO_DEVICE_SOFT` | unset | **presence** | Route the per-particle timestep-globals reduction, and the softening lookup, through the host instead of the device kernel. Both are substantially slower — the first copies the particle arrays to the host every step. They exist as a fallback should the device path stall on a particular driver or device, and as a cross-check that the two paths agree. |

## Logging, timing and diagnostics

All **diagnostic**: none of these changes the trajectory. Several add device synchronisations or
host copies and so cost wall-clock time; where that cost is large it is noted.

| Variable | Default | How it is read | Effect |
|---|---|---|---|
| `GADGET_HIP_DEBUG` | unset | **presence** | Verbose developer logging: device pointer ranges, buffer sizes, group-max traces, windowed buffer dumps. Identical to the `--debug` flag. Voluminous — at large particle counts this is the majority of all log output. |
| `GADGET_HIP_PHASE_TIME` | unset | **presence** | Per-step phase timers on `stderr` (`predict`, `sort_bodies`, `build_tree`, `approximate_gravity`, `PM + long-range`, `correct`, `STEP TOTAL`), plus sub-phase timers inside the PM solve. Both ends of every phase are device-synced, so the numbers measure work rather than launch latency — which also means enabling them removes the overlap between phases and lengthens the step. |
| `GADGET_HIP_PROPS_TIME` | unset | **presence** | Wall time of the tree-properties stage, per step, with the stream synced on both sides. |
| `GADGET_HIP_MEMREPORT` | `0` | `atoi() != 0` | Per-buffer device memory report at each allocation stage. Each row states whether the host side is pinned and whether the buffer has a separate device copy at all, so the total reflects the real footprint rather than counting a shared allocation twice. |
| `GADGET_HIP_NO_CPU_LOG` | unset | **presence** | Stop writing `cpu.txt`, and with it the per-phase device synchronisations needed to produce it. `cpu.txt` is written by default, but only for runs started with `--param`, since it needs an `OutputDir`. Use this to recover the overlap those synchronisations cost. |
| `GADGET_HIP_INVARIANTS` | unset | **presence** | Runtime invariant checks, one line per check on `stderr`: the tree's leaves exactly tile `[0, N)`, `t_current` is exactly the minimum over every particle's next-step time, per-particle force softening is strictly positive, and buffers expected to be duplicates of one another are. Also prints `t_current` at full precision (`%.9g`), which the ordinary step line's six significant figures can hide — useful when a step looks as though it is not advancing. Costs several device-to-host copies per step. |
| `GADGET_HIP_ENERGY_EVERY_STEP` | off with `--param`, on without | `atoi() != 0` | Evaluate the global energies every step instead of at the parameter file's `TimeBetStatistics` cadence. GADGET uses the cadence, and every-step evaluation is expensive at large `N`: it forces a fresh long-range potential solve on every step. Runs started without `--param` have no cadence to follow and so evaluate every step by default. |
| `GADGET_HIP_CELL_WATER` | unset | **presence** | Report the tree walk's per-step cell-list high-water mark against its capacity, plus forced-descent counts, big-stack retries, and the pure/mixed/coarse split of zoom groups. The frontier figure is what says how close the walk's fixed-size buffer is to overflowing. A big-stack retry serialises a group through a GPU-wide spin lock and is never normal, so retries are reported whether or not this is set; the variable makes the line print every step. |
| `GADGET_HIP_NACTIVE_DUMP` | unset | **presence** | Print the genuine active-particle count per step, read back from the tree walk's own output mask rather than from a host-side counter. Costs one device-to-host copy and an `O(N)` host pass per step. |
