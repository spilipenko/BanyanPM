# Compile-time options

Several things GADGET-2 puts in a `Makefile` are compile-time choices here too, for the same
reason: `PMGRID`, `PERIODIC`, `UNEQUALSOFTENINGS`, `PLACEHIGHRESREGION` and `LONGIDS` are not
parameter-file values in GADGET-2 either. This page lists every one of them, what the guarded code
actually does, and what happens when you leave one out.

Runtime switches are in [`ENVIRONMENT.md`](ENVIRONMENT.md). Options are set on the `cmake` line;
a handful of constants are header edits instead, and those are kept in their own section below so
there is no confusion about which is which.

---

## The binary tells you how it was built

Every run prints a `[BUILD]` block on `stderr` before it reads the parameter file, listing every
option below with its state — and, for the ones that are `OFF`, what the binary will therefore
refuse. An `[ENV]` block follows with every `GADGET_HIP_*` variable set in the environment. If you
are unsure what a build is, run it with a nonexistent parameter file and read the block; it prints
before anything can fail.

A debug build says so there too, which is worth checking before reporting a performance problem.

## Read this first: four options whose omission is not a build error

Every option in this document defaults to **OFF**, and the two most important of them —
`PERIODIC` and `MAC_SPRINGEL` — are not even namespaced, so a stray build directory is easy to
mistake for a configured one. A binary missing one of these four **compiles cleanly, links, starts,
and then refuses the parameter file** — or, in one case, refuses the initial conditions. The refusal
is loud and names the option, but it arrives after the build, which is the wrong time to find out.

| Omitted | What the binary rejects | Message |
|---|---|---|
| `PERIODIC` | `PeriodicBoundariesOn 1` in the parameter file | `PeriodicBoundariesOn=1 does not match this build's periodicity mode (built WITHOUT PERIODIC)` (`src/gadget_params.cpp:245-251`) |
| `MAC_SPRINGEL` | `TypeOfOpeningCriterion 1` | `FATAL (C-B-03): the parameter file requests TypeOfOpeningCriterion = 1 … but this binary was built WITHOUT the relative MAC` (`src/main.cpp:865-879`) |
| `UNEQUAL_SOFTENINGS` | a parameter file giving **more than one distinct non-zero** `Softening*` value | `FATAL (T28): this parameter file specifies N DIFFERENT non-zero softenings … built WITHOUT mixed-softening support` (`src/main.cpp:893-918`) |
| `GADGET_HIP_LONGIDS` | an initial-conditions file whose `ID` block is 8 bytes per particle | `ID block size mismatch` (`src/gadget_snapshot.cpp:666-667`), fatal at IC load (`src/main.cpp:1618`) |

Two of these refuse **in both directions**, which is worth knowing before you reuse a build
directory:

- `PERIODIC` is matched exactly. A build **with** `PERIODIC` refuses `PeriodicBoundariesOn 0`, and a
  build **without** it refuses `PeriodicBoundariesOn 1`. There is no "periodic build running a
  non-periodic problem" mode.
- `GADGET_HIP_LONGIDS` is likewise an exact match against the file: a `LONGIDS` binary rejects a
  4-byte-ID snapshot, because the ID block length is checked against `N × 8`, not sniffed.

Two asymmetries to be aware of, because they are the cases that do **not** stop:

- **`MAC_SPRINGEL` is only checked one way.** `TypeOfOpeningCriterion 1` without the build is
  refused; `TypeOfOpeningCriterion 0` *with* the build is **not**. The opening criterion is selected
  at compile time only (`CUDAkernels/dev_approximate_gravity_warp_new.cu:1807-1846`), so a
  `MAC_SPRINGEL` binary evaluates the relative criterion whatever the parameter file asks for. The
  startup banner prints which one is compiled in — `[INIT] MAC: Springel(2005)` or
  `[INIT] MAC: improved-BH (theta)` (`src/main.cpp:1469-1473`). Check that line before comparing two
  runs.
- **`GADGET_HIP_HIGHRES`** is checked against the command line, not the parameter file.
  `--zoom-mask` is parsed by every build, since the option table is not compiled conditionally, but
  the zoom mesh only exists with this option. Passing `--zoom-mask` to a binary built without it is
  **refused at startup**, because the alternative is a run that quietly gives every high-resolution
  particle nothing but the coarse mesh's long-range force. `--zoom-enlarge` and `--zoom-recenter`
  are inert without a mask even in a zoom build, so those warn rather than refuse.

`tests/gate64/gate64.sh` checks the configuration of the binary it is given before running any
physics, by reading the startup banner, which catches a missing `PERIODIC` or `PMGRID`.

---

## Canonical builds

### Standard periodic cosmological box

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DPERIODIC=ON \
      -DMAC_SPRINGEL=ON \
      -DGADGET_HIP_PMGRID=512
make -C build -j12 gadget_hip
```

`PERIODIC` requires `GADGET_HIP_PMGRID` and is refused at configure time without it. `MAC_SPRINGEL`
is in the list because `TypeOfOpeningCriterion 1` is what GADGET-2's own example parameter files
use; drop it only if your parameter file says `TypeOfOpeningCriterion 0`.

### Zoom (`PLACEHIGHRESREGION`)

```bash
cmake -S . -B build-zoom -DCMAKE_BUILD_TYPE=Release \
      -DPERIODIC=ON \
      -DMAC_SPRINGEL=ON \
      -DUNEQUAL_SOFTENINGS=ON \
      -DGADGET_HIP_HIGHRES=ON \
      -DGADGET_HIP_PMGRID=256
make -C build-zoom -j12 gadget_hip
```

`UNEQUAL_SOFTENINGS` belongs in every zoom build: a zoom has per-type softenings by construction,
and a build without it refuses such a parameter file. `GADGET_HIP_HIGHRES` requires
`GADGET_HIP_PMGRID`.

### Large runs, on an APU

```bash
cmake -S . -B build-big -DCMAKE_BUILD_TYPE=Release \
      -DPERIODIC=ON \
      -DMAC_SPRINGEL=ON \
      -DGADGET_HIP_PMGRID=512 \
      -DGADGET_HIP_UNIFIED_MEM=ON
make -C build-big -j12 gadget_hip
```

`GADGET_HIP_UNIFIED_MEM` takes the resident footprint from 440 to 333 bytes per particle — at
512³, 55.0 GiB down to 41.7 GiB. It is only a win where host and device memory are the same
silicon; see its entry below for the cost.
It can be turned back off at run time with `GADGET_HIP_UNIFIED_MEM=0`, so this build is a superset
of the one above rather than a different configuration.

### Non-periodic (isolated) runs

`GADGET_HIP_PMGRID` **without** `PERIODIC` compiles the non-periodic long-range solver instead of
the periodic one — GADGET-2's own `pm_nonperiodic` mode, for an isolated system with vacuum
boundaries rather than a cosmological box. The tree runs without minimum-image wrapping and the
mesh FFTs a doubled `2 × PMGRID` grid, zero-padded so the inherently periodic transform cannot wrap
real structure back onto itself. The Green's function is a cached real-space kernel rather than an
analytic k-space formula, and `Asmth` and `Rcut` follow from the mesh the same way:
`Asmth = 1.25 · meshSize / (2·PMGRID)`, `Rcut = 4.5 · Asmth`, echoed as an `[ISOLATED]` line at
startup.

```bash
cmake -S . -B build-isolated -DCMAKE_BUILD_TYPE=Release \
  -DGADGET_HIP_PMGRID=256 -DMAC_SPRINGEL=ON          # note: no PERIODIC
make -C build-isolated -j12 gadget_hip
./build-isolated/gadget_hip --param run.param --meshsize 1000.0
```

Three things this mode requires that the periodic one does not:

- **`--meshsize` is mandatory** and the run refuses to start without it. It is the physical size of
  the *doubled* grid, so the usable region is `meshSize/2` on a side; the outer half is the padding.
- **Particles must lie within `[0, meshSize/2)` on every axis.** To guarantee that, every loaded
  coordinate is shifted by `+meshSize/4` at startup, which centres an origin-centred distribution
  in the occupied half with equal padding on both sides.
- **The shift is undone when snapshots are written**, so output stays in the initial conditions'
  frame — no correction is needed on your side. It is recorded in a sidecar next to the checkpoint
  file, so a resumed run writes in the same frame as the run it continues. Unlike the periodic
  zoom's recentring, coordinates are *not* then wrapped into `[0, BoxSize)`: an isolated system has
  no periodic images, and wrapping would move particles to the far side of a box that does not
  exist.
- The parameter file's `PeriodicBoundariesOn` must be `0`, since periodicity is matched against the
  build in both directions.

### Toolchain and target

| Variable | Default | Notes |
|---|---|---|
| `CMAKE_BUILD_TYPE` | empty | Set `Release`. A default-configured build is correct and much slower. |
| `CMAKE_HIP_ARCHITECTURES` | `gfx1151` (`CMakeLists.txt:63-65`) | The GPU target. **Set this for any other device** — the default is a single architecture, not a fat binary. |
| `CMAKE_CXX_COMPILER` | searched | ROCm's own `clang++` is used as the host compiler, not stock GCC. The search covers `$ROCM_PATH`, `/usr/lib64/rocm`, `/opt/rocm` and `/opt/rocm-*` (newest first), under `llvm/bin` and `lib/llvm/bin`, and configuration fails with the list of places it looked if none has one (`CMakeLists.txt:14-52`). Set `ROCM_PATH` to point the search, or this variable to override it outright. |

Dependencies: HIP and OpenMP always; **rocFFT only when `GADGET_HIP_PMGRID` is set**; MPI only for
`GADGET_HIP_BUILD_BONSAI_TOOLS`. A non-PM build does not need rocFFT installed.

---

## CMake options

`ON`/`OFF` options are booleans; the four `GADGET_HIP_N*`/`PMGRID` entries take a value and are
empty by default.

### Physics and numerics

| Option | Macro defined | Default | What it does |
|---|---|---|---|
| `PERIODIC` | `PERIODIC` | `OFF` | Minimum-image periodic tree walk plus the TreePM short-range/long-range split. `Asmth = 1.25 × BoxSize / PMGRID` and `Rcut = 4.5 × Asmth`, from GADGET-2's own compile-time `ASMTH`/`RCUT` constants, and not separately tunable (`src/main.cpp:1935-1947`). Requires `GADGET_HIP_PMGRID`: in GADGET-2 periodicity and "use PM" are independent, but only the TreePM combination is implemented here, so `PERIODIC` alone — which would drop the long-range force with nothing computing it — is refused at configure time (`CMakeLists.txt:254-259`). Exactly matched against `PeriodicBoundariesOn`. |
| `MAC_SPRINGEL` | **`_MAC_SPRINGEL_`** | `OFF` | Selects Springel (2005)'s relative-acceleration multipole acceptance criterion, driven by `ErrTolForceAcc`, in place of the improved Barnes–Hut opening angle driven by `ErrTolTheta`. GADGET-2's two-phase bootstrap is reproduced: the first force evaluation of a run has no `OldAcc` yet and uses the geometric criterion, switching to the relative one permanently thereafter (`CUDAkernels/dev_approximate_gravity_warp_new.cu:1807-1842`). Required for `TypeOfOpeningCriterion 1`. |
| `UNEQUAL_SOFTENINGS` | **`UNEQUALSOFTENINGS`** | `OFF` | GADGET-2's mixed-softening machinery: a per-node softening summary (`nodeSoftInfo`, one float per node, encoding the node's maximum `ForceSoftening` in its magnitude and "members differ" in its sign) plus the forced descent that opens a node whose members have different softenings when the target is inside the softening radius (`CUDAkernels/dev_approximate_gravity_warp_new.cu:1848-1880`). A run whose populated types all share one softening pays nothing for leaving it off: the buffer is allocated at one element, the reduction is not compiled, and the walk carries no extra load or branch (`src/build.cpp:386`, `411-416`). Needed by every zoom and every multi-species run. |
| `GADGET_HIP_PMGRID` | **`PMGRID`** | empty | Particle-mesh cells per side, GADGET-2's `PMGRID`. **Empty means the PM solver is not compiled in at all** — there is deliberately no default mesh size, since a plausible-but-wrong default here would be far more consequential than a tuning default. Setting it pulls in rocFFT and the PM orchestration sources. With `PERIODIC` the mesh is periodic; without it, the non-periodic isolated solver is compiled instead, which FFTs a doubled `2 × PMGRID` grid for zero padding and requires `--meshsize` at startup (`src/main.cpp:959-966`, `1950-1964`). |
| `GADGET_HIP_HIGHRES` | `GADGET_HIP_HIGHRES` | `OFF` | `PLACEHIGHRESREGION`: a second, non-periodic mesh refining the high-res region, on top of the periodic one, as GADGET-2 does it, with the tree-side per-type `Rcut`/`Asmth` split that goes with it. Requires `GADGET_HIP_PMGRID`; refused together with `GADGET_HIP_QUADRUPOLE` (`CMakeLists.txt:272-280`). Works with or without `PERIODIC` — both are real GADGET-2 modes. The fine mesh is a doubled FFT like any isolated solve, so `GADGET_HIP_PMGRID=512` means a 1024³ fine transform. Without this option `--zoom-mask` is refused at startup rather than ignored. |
| `GADGET_HIP_QUADRUPOLE` | `GADGET_HIP_QUADRUPOLE` → `_QUADRUPOLE_` | `OFF` | Adds quadrupole moment corrections to the tree force. Off by default to match GADGET-2, which is monopole-only by deliberate design. **Unusable in any TreePM build:** combined with `PMGRID` it is a hard `#error`, because there is no reference for how the short-range `erfc` suppression should apply to a quadrupole term and minimum-image wrapping was never added to the quadrupole accumulator (`CUDAkernels/dev_approximate_gravity_warp_new.cu:149-156`). Since `PERIODIC` implies `PMGRID`, this leaves non-periodic pure-tree runs as the only configuration that can use it. |

### Capability and limits

| Option | Macro defined | Default | What it does |
|---|---|---|---|
| `GADGET_HIP_LONGIDS` | `GADGET_HIP_LONGIDS` | `OFF` | 8-byte instead of 4-byte particle IDs in the snapshot `ID` block, matching GADGET-2's own `LONGIDS` flag. The width is **never autodetected** — on read it is asserted against the block length, on write it is simply used. Set it whenever the IDs in play exceed 2³² or your ICs were written by a `LONGIDS` GADGET-2. |
| `GADGET_HIP_UNIFIED_MEM` | `GADGET_HIP_UNIFIED_MEM` | `OFF` | Takes a pinned buffer's device pointer from `hipHostGetDevicePointer` instead of allocating a second copy, so seventeen large arrays exist once rather than as a host/device pair and host-to-device transfers become synchronisation points. Footprint goes from 440 to 333 bytes per particle — 55.0 GiB to 41.7 GiB at 512³ — which is what raises the largest problem fitting in 128 GB to roughly 770³. The trade: writes through the alias are slower (measured at 0.990 of device-memory bandwidth for reads, 0.971 read-modify-write, 0.935 streaming writes), worth about 6% per step early in a run and about 11% by z ≈ 1 as the write-heavy phases take a larger share. Results are unaffected, including the timestep sequence. **OFF by default because a discrete GPU genuinely needs the copy**; it is only a win on an APU. Runtime overrides `GADGET_HIP_UNIFIED_MEM=0` and `GADGET_HIP_UNIFIED_ARRAYS=<bitmask>` exist on a binary built with it, so there is no need for a second build directory to A/B it. If the runtime declines the mapping the code falls back to a separate device allocation with a warning rather than failing. |
| `GADGET_HIP_NLEAF` | **`NLEAF`** | empty → header's `16` | Octree leaf size. **This is the option that sets the particle ceiling.** A leaf's particle count minus one is packed into the top `log2(NLEAF)` bits of a 32-bit word and the index of its first body into the remaining `LEAFBIT = 32 − log2(NLEAF)`, so the run is capped at 2^`LEAFBIT` particles — 268,435,456 at the default 16. Exceeding it is a refusal at allocation time naming the field and the limit, not a truncated index (`src/build.cpp:66-81`). Smaller `NLEAF` buys index bits at an accuracy and performance cost. Legal values are 1, 2, 8, 16, 32, 64, 128; anything else is a `#error`. `NCRIT >= NLEAF` is enforced at compile time. |
| `GADGET_HIP_NCRIT` | **`NCRIT`** | empty → header's `32` | Tree-walk group size (the i-particle group). 32 is deliberate rather than inherited: it matches the wave32 width exactly, so the walk's `NI = (nb_i <= WARP_SIZE) ? 1 : 2` dispatch always takes the single-particle-per-lane path. Measured on gfx1151, `NCRIT=32` is about 15% faster than 64 (260.8 µs against 306.0 µs mean `dev_approximate_gravity` duration over a 21-dispatch sample) with no accuracy penalty against direct summation, the cause being 96 against 120 VGPRs per wave. Legal values 1, 2, 8, 16, 32, 64, 128. Must be `>= NLEAF`, so lowering `NCRIT` below 16 also needs `GADGET_HIP_NLEAF` lowered. |
| `GADGET_HIP_NTHREAD` | **`NTHREAD`** | empty → header's `128` | Threads per block in the warp-cooperative tree-walk kernel, i.e. how many warps cooperate per block. Legal values 8, 16, 32, 64, 96, 128, 256. This also scales the tree-walk stack reservation — see `LMEM_STACK_SIZE` below — so it has a memory consequence as well as an occupancy one. |
| `GADGET_HIP_DETERMINISTIC_CIC` | `GADGET_HIP_DETERMINISTIC_CIC` | `OFF` | Makes the run **bit-reproducible**. PM mass assignment normally uses `atomicAdd` on `float`; float addition is not associative and the launch order varies, so the density grid differs in its last bits between two runs of the same binary, the FFT spreads that across the box and the trajectory diverges chaotically — 0.42% in kinetic energy by z = 0 at 64³, and ±40 in step count. With this option the scatter accumulates into fixed-point unsigned 64-bit integers, which *are* associative, and the result no longer depends on order. **Not a production configuration.** It costs an extra grid-sized `uint64` buffer — about 1.1 GB at `PMGRID=512` — and 64-bit atomics are slower than 32-bit. Its use is the honest A/B: build two binaries with it, change one thing, and any difference in the output is that change rather than the noise floor, which otherwise swallows anything below about 0.5% at 64³. |

### Build layout

| Option | Default | What it does |
|---|---|---|
| `GADGET_HIP_BUILD_BONSAI_TOOLS` | `OFF` | Builds three legacy Bonsai-era utilities (`bonsai_io`, `bonsai_driver`, `bonsai_clrshm`). They are the only code in the tree that includes `<mpi.h>`, which is why MPI is a hard dependency once this is on, and they have nothing to do with the simulation. Defines no macro; it gates targets only. |

---

## Header constants a user might change

These are plain `#define`s, not CMake options: changing one means **editing the header and
rebuilding**. They are here because each sets a real limit, and a run that hits one needs to know
which number moved it. Everything in this section lives in `include/node_specs.h` unless noted.

| Constant | Value | Effect of changing it |
|---|---|---|
| `LMEM_STACK_SIZE` | `8192` (line 95) | Per-thread storage in the tree walk's cell-list ring buffer; the per-warp capacity is `LMEM_STACK_SIZE × 32` = 262,144 entries. **Must be a power of two** — there is a `#error` on the derived `CELL_LIST_MEM_PER_WARP` that checks it. This is the buffer the walk's live frontier has to fit in: when a group's frontier exceeds it the walk retries that group on a 256×-larger stack, serialised through a GPU-wide spin lock, and if that also fails the step is refused rather than integrated with incomplete forces (`src/gpu_iterate.cpp:4843-4853`). It is also the floor on the single largest allocation in the run: `2 × (LMEM_STACK_SIZE × NTHREAD + LMEM_EXTRA_SIZE) × nBlocks` words, 5.010 GiB at any particle count below about 1.39 × 10⁸ (`src/build.cpp:150-152`, `219`). Raise it if `GADGET_HIP_CELL_WATER` shows the high-water mark approaching capacity; lower it only if that headroom is provably unnecessary at your clustering. |
| `LMEM_EXTRA_SIZE` | `2048` (line 96) | Per-block slack in the same reservation. |
| `MAXLEVELS` | `30` (line 139) | Maximum octree depth, the length of the per-level index arrays, **and the Morton key resolution**: the domain factor is `BoxSize / 2^MAXLEVELS`, so two particles closer together than that are indistinguishable to the key. A build that reaches the limit stops with `FATAL: tree depth reached MAXLEVELS`, reporting the box size and the key resolution (`src/build.cpp:620-640`). The cause is coincident particles, not a large box: more than `NLEAF` particles sharing one key produce a node at every level, and unlike GADGET-2 this code does not randomise the subnode of coincident particles, so it has no escape. Check the ICs for duplicates before raising this. |
| `NLEAF`, `NCRIT`, `NTHREAD` | `16`, `32`, `128` (lines 157-194) | The defaults the CMake options above override. Each is wrapped in `#ifndef`, so the CMake option and a header edit are the same mechanism; prefer the option, which keeps two differently-sized build directories side by side. |
| `ERR_TOL_FORCE_ACC` | `0.005f` (line 75) | Default `ErrTolForceAcc` for the Springel MAC when no parameter file supplies one. `0.005` paired with `ErrTolTheta = 0.5` is what GADGET-2's own example parameter files use. A `--param` run takes the file's value, so this only affects runs driven from the command line. |
| `BONSAI_DENSITY` | `0` (line 338) | The SPH-style density and neighbour-count subsystem inherited from Bonsai. **It has no effect on gravity here** — force softening comes from a separate per-particle buffer, and the density chain feeds only itself. At `0` its buffer is allocated at one element and the whole chain is eliminated from the inner force loop, saving a `float2` per particle — 1 GB at 512³. Set it to `1` only if you want those fields for their own sake. |
| `START_LEVEL_MIN_NODES` | `16` (line 142) | Minimum node count a level must reach before leaves are created. A tree-build tuning constant. |
| `ENABLE_LOG` | `1` (`include/log.h:2`) | Compiles the `LOG`/`LOGF` macros out entirely when `0`. The macros are already gated on a runtime flag, so this buys little beyond code size. |

Two constants in the same header are present but inert, and should not be mistaken for switches:
`INDSOFT` (individual softening with a cubic spline kernel, line 59) and `USE_THRUST` (use Thrust's
radix sort in place of the built-in one, line 29) are both **commented out**, and
`DO_BLOCK_TIMESTEP_EXCHANGE_MPI` (line 26) only ever mattered under MPI.

---

## Macros with no CMake option

These gate real code but have no `option()` behind them. Nothing on a `cmake` line sets them.

| Macro | How it is defined | What it means |
|---|---|---|
| `USE_MPI` | **never defined, anywhere** | Guards roughly 53 blocks, including the entire body of `src/parallel.cpp`. Multi-rank is not supported: there is no MPI collective on any GADGET-specific quantity, the PM solver refuses more than one process, and the core library links with zero MPI references. Adding `-DUSE_MPI` restores the code but not working multi-rank operation. |
| `IMPBH` | `include/node_specs.h:58`, unconditionally | Selects the improved Barnes–Hut opening test, which is always used for the group's AABB distance test. It is not a toggle in practice: with `MAC_SPRINGEL` off it is the whole criterion, and with `MAC_SPRINGEL` on it is still what the bootstrap step and the group-distance test use. |
| `DO_BLOCK_TIMESTEP` | `include/node_specs.h:21`, unconditionally | Individual timesteps on a power-of-two ladder, which is the integrator this code implements. Not intended to be removed. |
| `_MPIMT`, `USE_CUB`, `__HIP_PLATFORM_AMD__=1` | `add_definitions`, `CMakeLists.txt:96` | Always on. `__HIP_PLATFORM_AMD__` is needed globally rather than only on GPU sources because every file in the tree includes `hip/hip_runtime.h`. |
| `_QUADRUPOLE_` | `#define`d by `GADGET_HIP_QUADRUPOLE`, `CUDAkernels/dev_approximate_gravity_warp_new.cu:145-147` | The macro the kernel actually tests. Do not define it directly; use the CMake option, which also gets you the configure-time compatibility checks. |

A number of further macros are inherited from Bonsai and are either never defined (`USE_OPENGL`,
`USE_DUST`, `GALACTICS`, `SHMODE` outside one local block, `EXACT_KEY`, `USE_AVX`), defined
unconditionally in a header (`USE_CUDA`, `PARALLELSORT`, `USE_GROUP_TREE`, `USE_ALL2ALLV`), or
reachable only through the MPI paths that are compiled out (`USE_HASH_TABLE_DOMAIN_DECOMP`,
`NUMBER_OF_FULL_EXCHANGE`, `MULTI_GPU_MEM_INCREASE`). None is a supported build switch.

---

## Combinations refused before the binary exists

Unlike the runtime refusals at the top of this page, these stop at configure or compile time.

| Combination | Where it stops | Why |
|---|---|---|
| `PERIODIC` without `GADGET_HIP_PMGRID` | configure, `CMakeLists.txt:255-259` | Only the TreePM combination is implemented; `PERIODIC` alone would drop all long-range force with nothing computing it. |
| `GADGET_HIP_HIGHRES` without `GADGET_HIP_PMGRID` | configure, `CMakeLists.txt:273-276` | The zoom is a refinement of the PM solve, meaningless with no PM. |
| `GADGET_HIP_HIGHRES` with `GADGET_HIP_QUADRUPOLE` | configure, `CMakeLists.txt:277-280` | The per-type `Rcut`/`Asmth` split extends the monopole path only, and there is no GADGET-2 reference for the combination. |
| `GADGET_HIP_PMGRID` with `GADGET_HIP_QUADRUPOLE` | compile, `CUDAkernels/dev_approximate_gravity_warp_new.cu:149-156` | No reference for how the TreePM short-range `erfc` suppression applies to a quadrupole term, and the quadrupole accumulator has no minimum-image wrapping. Building it would give unsuppressed, non-minimum-image forces with no error. |
| `NCRIT < NLEAF` | compile, `include/node_specs.h:319-321` | A group cannot be smaller than the leaves it is built from. |
| `NLEAF`, `NCRIT` or `NTHREAD` outside its legal set | compile, `include/node_specs.h` | Each has an explicit bit layout per value; there is no generic case. |
| `LMEM_STACK_SIZE` not a power of two | compile, `include/node_specs.h:98-100` | The cell list is a ring buffer addressed by masking. |
