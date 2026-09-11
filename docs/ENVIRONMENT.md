# Runtime environment variables

The port carries ~90 environment switches. They fall into two groups, and the distinction is the
important part:

- **Behaviour switches** — documented below. Each has a default that is the intended production
  behaviour; the variable exists so the *other* setting can be measured against it on the same
  binary. That is how every performance and physics claim in this project was established: same
  binary, same state, one variable changed.
- **Audit probes** — named after the ticket or contract that introduced them (`GADGET_HIP_T<nn>_*`,
  `GADGET_HIP_C<A-D><nn>_*`). Off by default, usually printing statistics or enabling a one-off
  trace. They are development instruments. The authoritative description of each is the ticket it
  is named for.

Unless stated otherwise, a variable is enabled by being set to any value (`=1` by convention) and
disabled by being unset.

---

## Behaviour switches

### Logging and profiling

| Variable | Default | Effect |
|---|---|---|
| `GADGET_HIP_DEBUG` | off | Per-step developer probes: device pointer ranges, buffer sizes, group-max traces, windowed buffer dumps. Same as `--debug`. In a 512³ run these were ~60% of all log lines. |
| `GADGET_HIP_PHASE_TIME` | off | Per-step phase timers (`predict`, `sort_bodies`, `build_tree`, `approximate_gravity`, `PM + long-range`, `correct`, `STEP TOTAL`) plus sub-phase timers inside the PM solve. Both sides device-synced, so they time work rather than kernel-launch latency — which also means enabling them perturbs overlap. |
| `GADGET_HIP_PROPS_TIME` | off | Timers for the tree-properties stage. |
| `GADGET_HIP_MEMREPORT` | off | Per-buffer device memory report. |
| `GADGET_HIP_NO_CPU_LOG` | off | Disable `cpu.txt` and, with it, the per-phase device syncs that produce it. Also the A/B control for measuring what those syncs cost. |
| `GADGET_HIP_INVARIANTS` | off | Runtime invariant checks, including full-precision `t_current` printing (`%.9g`). Useful when a step looks like it is not advancing: the `iter=` line's six significant figures can hide a step that is genuinely tiny. |

### Physics and performance

| Variable | Default | Effect |
|---|---|---|
| `GADGET_HIP_PM_CIC` | `1` | `1` = transposed thread→particle mapping in the PM mass assignment. `0` = the legacy mapping, in which Peano-Hilbert-sorted particles put a whole wavefront on the same eight grid addresses and the atomics serialise. The default is **34× faster** at z=0 and sums the same contributions. |
| `GADGET_HIP_NO_BOX_WRAP` | off | Disables mapping particles back into the periodic box each step. Set it only to reproduce the pre-fix behaviour: with it set, coordinates accumulate their unwrapped trajectory and any downstream tool that assumes periodic coordinates gets wrong answers. |
| `GADGET_HIP_ENERGY_EVERY_STEP` | off with `--param` | Compute global energies every step instead of at `TimeBetStatistics` cadence. Gadget uses the cadence; every-step evaluation cost 9.5 s of a 14.9 s step at 512³. Runs without `--param` use every-step, since there is no cadence to follow. |
| `GADGET_HIP_PM_CADENCE` | off | Recompute the PM mesh on its own cadence instead of every step. **Leave off.** As implemented, the long-range force is folded into the active particles' acceleration, so with a cadence inactive particles never receive the long-range impulse; this cost 11% in kinetic energy by z=0. |
| `GADGET_HIP_PM_EVERY_STEP` | on | Forces PM every step (the default when the cadence is off). |
| `GADGET_HIP_PM_NO_IDSPACE` | off | Disables the id-space fast path for storing PM forces across steps. |
| `GADGET_HIP_T35_NO_DEVICE_SOFT` | off | Recompute per-particle softening on the host instead of the device. The device path was verified bit-exact over 60 steps; the host path costs 1.78 s/step against 0.012 s at 512³. |
| `GADGET_HIP_T35_NO_DEVICE_DT` | off | Reduce timestep globals on the host instead of the device. The host path adds a 4.8 GB device→host copy every step. |
| `GADGET_HIP_T35_DT_CROSSCHECK` | off | Run both timestep reductions and compare. |
| `GADGET_HIP_CD13` | on | The bin-growth synchronisation rule: a particle may lengthen its timestep only when its end tick is divisible by the new step. Disabling it is the A/B control for that rule. |
| `GADGET_HIP_CD04_LADDER` | on | Anchor the timestep ladder to the run's timeline span. |
| `GADGET_HIP_TREE_ALLOC_FACTOR` | — | Override the tree allocation factor. |
| `GADGET_HIP_LAUNCH_BLOCKS`, `GADGET_HIP_LAUNCH_THREADS` | — | Override kernel launch geometry. |

---

## Audit probes

Off by default. Each is tied to the ticket or contract in its name; that ticket is the accurate
description of what it does and what it was used to establish.

```
GADGET_HIP_CA01_AUDIT            GADGET_HIP_CA01_CC_SKIP        GADGET_HIP_CA01_CELL_CENTER
GADGET_HIP_CA07                  GADGET_HIP_CB19_NO_RCUT_REJECT GADGET_HIP_CD01_STATS
GADGET_HIP_CD01_TRACE            GADGET_HIP_CD02_KICK_TEND      GADGET_HIP_CD03B_FIX_INDEX
GADGET_HIP_CD03_STRICT_ACTIVE    GADGET_HIP_CD08_STATS          GADGET_HIP_CD13_STATS
GADGET_HIP_COMOVING_DE_CHECK     GADGET_HIP_COMPACT_GROUPS      GADGET_HIP_DUMP_ACC
GADGET_HIP_DUMP_ACC_ITER         GADGET_HIP_DUMP_ACC_WARM       GADGET_HIP_ENERGY_EXTRAP
GADGET_HIP_EPOT_EXTRAP_CHECK     GADGET_HIP_EPOT_TOPN           GADGET_HIP_GROUP_CORNER_DUMP
GADGET_HIP_GROUP_CORNER_STATS    GADGET_HIP_KICK_EXTRAP_CHECK   GADGET_HIP_NACTIVE_DUMP
GADGET_HIP_NTAB                  GADGET_HIP_POISON              GADGET_HIP_SPIKE
GADGET_HIP_STAGE_TRACE           GADGET_HIP_T10_GROUP_SHIFT     GADGET_HIP_T11_PROX_CENTER
GADGET_HIP_T1_AOLD_REDUCE        GADGET_HIP_T24_ACC1_TOPN       GADGET_HIP_T24_NODE_SCAN
GADGET_HIP_T24_NODE_WINDOW       GADGET_HIP_T24_WINDOW          GADGET_HIP_T26_BJ_DIST
GADGET_HIP_T26_BJ_DIST_BOXSIZE   GADGET_HIP_T26_BJ_DIST_EVERY   GADGET_HIP_T27_DISABLE_BOX_FLOOR
GADGET_HIP_T27_SOFT_COEF         GADGET_HIP_T28_BOXSIZE         GADGET_HIP_T28_DISABLE
GADGET_HIP_T28_NODESOFT          GADGET_HIP_T28_STATS           GADGET_HIP_T28_TRACE
GADGET_HIP_T31_BISECT            GADGET_HIP_T31_FORCE_GEO       GADGET_HIP_T31_SKIP_GMA
GADGET_HIP_T31_STATS             GADGET_HIP_T31_TIME_WALK       GADGET_HIP_T4_COUNT
GADGET_HIP_T4_FIX_G              GADGET_HIP_T6_DRIFT_TP         GADGET_HIP_T7_INJECT
GADGET_HIP_T9_CELLSIZE_BJ        GADGET_HIP_T9_CELLSIZE_GEO     GADGET_HIP_T9_NO_S
GADGET_HIP_TRACE_DUP             GADGET_HIP_TRACE_Y             GADGET_HIP_TRACE_Y_DUMP
GADGET_HIP_TRACE_Y_GROUP         GADGET_HIP_TRACE_Y_VERBOSE     GADGET_HIP_TRAJ_DUMP
GADGET_HIP_TRAJ_IDS              GADGET_HIP_TRAJ_UNSORTED_DUMP  GADGET_HIP_RESTART_MAGIC
GADGET_HIP_RESTART_VERSION
```

Several of these select between a legacy behaviour and a corrected one and are **on** by default
(`GADGET_HIP_T9_CELLSIZE_GEO`, `GADGET_HIP_T9_CELLSIZE_BJ`, `GADGET_HIP_T9_NO_S`,
`GADGET_HIP_T11_PROX_CENTER`, `GADGET_HIP_CA01_CELL_CENTER`, `GADGET_HIP_CD02_KICK_TEND`,
`GADGET_HIP_CD03_STRICT_ACTIVE`, `GADGET_HIP_CD03B_FIX_INDEX`, `GADGET_HIP_ENERGY_EXTRAP`,
`GADGET_HIP_CD04_LADDER`, `GADGET_HIP_T6_DRIFT_TP`). The startup banner prints the active setting
for each, so the log records which physics the run actually used — check that banner before
comparing two runs.
