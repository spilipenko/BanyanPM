#include "hip/hip_runtime.h"
#include "bonsai.h"
// #include "support_kernels.cu0
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "../profiling/bonsai_timing.h"
PROF_MODULE(dev_approximate_gravity);


#include "node_specs.h"

#ifdef WIN32
#define M_PI        3.14159265358979323846264338328
#endif

#define WARP_SIZE2 5
#define WARP_SIZE  32

#if NCRIT > 4*WARP_SIZE
#error "NCRIT in include/node_specs.h must be <= WARP_SIZE"
#endif


#define laneId (threadIdx.x & (WARP_SIZE - 1))
#define warpId (threadIdx.x >> WARP_SIZE2)

#define BTEST(x) (-(int)(x))

#define FULL_MASK 0xffffffffffffffffULL

// Transverse-force bug trace (user request 2026-08-30): isolates whether the spurious y/z force
// found in the Zel'dovich symmetry test (translational invariance in y/z demands zero net
// transverse force; the port shows a real, structured, non-random one ~80x larger than Gadget-2's
// comparable tree-approximation noise, already proven via GADGET_HIP_DUMP_ACC to originate in the
// tree walk, not PM) comes from the monopole/APPROX path (group-shared MAC opening a node that a
// true per-particle MAC would have split) or the DIRECT leaf-particle path (which should be
// numerically exact regardless of grouping). Accumulates the y-contribution to one hardcoded
// target particle's acceleration separately per path; host code reads these back once per step.
// Target position matches particle id=1 from the earlier GADGET_HIP_DUMP_ACC trace
// (pos=(450.08,500,500) at iter 0 on zeldovich_ic_32b) -- adjust and rebuild to trace a different
// particle.
__device__ double g_traceApproxY = 0.0;
__device__ double g_traceDirectY = 0.0;
__device__ int    g_traceApproxN = 0;
__device__ int    g_traceDirectN = 0;
__device__ int    g_traceEnabled = 0;
__device__ int    g_traceVerbose = 0;
__device__ float  g_traceX = 450.08f, g_traceY = 500.0f, g_traceZ = 500.0f;

static __device__ __forceinline__ bool isTraceTarget(const float4 p)
{
  return g_traceEnabled &&
         fabsf(p.x - g_traceX) < 1.0f && fabsf(p.y - g_traceY) < 1.0f && fabsf(p.z - g_traceZ) < 1.0f;
}

// Safe per-source record buffer (user request 2026-08-30, second attempt): the first attempt used
// a conditional printf() inside the __shfl_sync-based warp loops in directAcc/approxAcc, which is
// NOT safe -- these functions assume full-warp lockstep for their shuffles, and divergent printf
// calls can corrupt that, which is exactly what happened (a cross-check against the independent
// bulk position dump showed the printf trace reporting two different position values for the same
// ptclIdx, which is impossible for a stable read-only array -- the printf itself was the bug, not
// the port). This version only ever does a plain atomicAdd-reserved global-memory write (no
// shuffles, no cross-thread assumptions), which is safe under warp divergence.
struct TraceRecord { int ptclIdx; int isDirect; float px, py, pz, mass, drx, dry, drz, deltaY; };
#define TRACE_BUF_SIZE 4096
__device__ TraceRecord g_traceBuf[TRACE_BUF_SIZE];
__device__ int g_traceBufCount = 0;

static __device__ __forceinline__ void traceRecord(int ptclIdx, int isDirect, float3 jpos, float jmass,
                                                     float drx, float dry, float drz, float deltaY)
{
  const int slot = atomicAdd(&g_traceBufCount, 1);
  if (slot < TRACE_BUF_SIZE)
  {
    g_traceBuf[slot].ptclIdx = ptclIdx;
    g_traceBuf[slot].isDirect = isDirect;
    g_traceBuf[slot].px = jpos.x; g_traceBuf[slot].py = jpos.y; g_traceBuf[slot].pz = jpos.z;
    g_traceBuf[slot].mass = jmass;
    g_traceBuf[slot].drx = drx; g_traceBuf[slot].dry = dry; g_traceBuf[slot].drz = drz;
    g_traceBuf[slot].deltaY = deltaY;
  }
}

// Host wrappers -- hipMemcpyToSymbol/FromSymbol need same-translation-unit visibility of the
// __device__ symbols above (pm.h's own convention note), so gpu_iterate.cpp calls these plain
// functions instead of touching g_trace* directly.
void traceY_reset()
{
  int one = 1;
  int verbose = getenv("GADGET_HIP_TRACE_Y_VERBOSE") ? 1 : 0;
  double zeroD = 0.0;
  int zeroI = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_traceEnabled), &one, sizeof(int));
  hipMemcpyToSymbol(HIP_SYMBOL(g_traceVerbose), &verbose, sizeof(int));
  hipMemcpyToSymbol(HIP_SYMBOL(g_traceApproxY), &zeroD, sizeof(double));
  hipMemcpyToSymbol(HIP_SYMBOL(g_traceDirectY), &zeroD, sizeof(double));
  hipMemcpyToSymbol(HIP_SYMBOL(g_traceApproxN), &zeroI, sizeof(int));
  hipMemcpyToSymbol(HIP_SYMBOL(g_traceDirectN), &zeroI, sizeof(int));
  hipMemcpyToSymbol(HIP_SYMBOL(g_traceBufCount), &zeroI, sizeof(int));
}

void traceY_dump(const char *path)
{
  int count = 0;
  hipMemcpyFromSymbol(&count, HIP_SYMBOL(g_traceBufCount), sizeof(int));
  const int n = count < TRACE_BUF_SIZE ? count : TRACE_BUF_SIZE;
  std::vector<TraceRecord> recs(n);
  if (n > 0)
    hipMemcpyFromSymbol(recs.data(), HIP_SYMBOL(g_traceBuf), n * sizeof(TraceRecord));
  FILE *f = fopen(path, "w");
  if (f)
  {
    fprintf(f, "# total_recorded=%d buf_capacity=%d (truncated=%d)\n", count, TRACE_BUF_SIZE, count > TRACE_BUF_SIZE);
    fprintf(f, "# ptclIdx isDirect px py pz mass drx dry drz deltaY\n");
    for (int i = 0; i < n; i++)
    {
      const TraceRecord &r = recs[i];
      fprintf(f, "%d %d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
              r.ptclIdx, r.isDirect, r.px, r.py, r.pz, r.mass, r.drx, r.dry, r.drz, r.deltaY);
    }
    fclose(f);
  }
}

void traceY_read(double &approxY, double &directY, int &approxN, int &directN)
{
  hipMemcpyFromSymbol(&approxY, HIP_SYMBOL(g_traceApproxY), sizeof(double));
  hipMemcpyFromSymbol(&directY, HIP_SYMBOL(g_traceDirectY), sizeof(double));
  hipMemcpyFromSymbol(&approxN, HIP_SYMBOL(g_traceApproxN), sizeof(int));
  hipMemcpyFromSymbol(&directN, HIP_SYMBOL(g_traceDirectN), sizeof(int));
}


// Gadget-2 is monopole-only by deliberate design, not simplification (GADGET2_NOTES.md: avoids a
// time-asymmetry/secular-integration-error problem under dynamic tree updates between rebuilds).
// This was Bonsai's own inherited default (hardcoded `#if 1` here, not a build flag at all) --
// found 2026-08-28 while auditing this file for periodic-tree work, and left ON, unnoticed,
// through all of Phase 3's validation (Springel MAC, N^2 comparisons, NCRIT experiments all ran
// with quadrupole active). Default flipped OFF per explicit user decision ("we will not use
// quadrupole") -- kept available via -DGADGET_HIP_QUADRUPOLE=ON for anyone wanting to experiment
// with it later, matching this project's own convention of keeping prior defaults reachable
// (e.g. NCRIT=64 via build-ncrit64/) rather than deleting the code path.
#ifdef GADGET_HIP_QUADRUPOLE
#define _QUADRUPOLE_
#endif

#if defined(PMGRID) && defined(_QUADRUPOLE_)
#error "GADGET_HIP_PMGRID + GADGET_HIP_QUADRUPOLE together is not implemented: Gadget-2 has no \
quadrupole moments to reference for how its TreePM short-range erfc suppression (shortrange_table) \
should apply to a quadrupole force term, and periodic minimum-image wrapping has NOT been added to \
the quadrupole add_acc() below -- silently building this combination would give wrong (unsuppressed, \
non-minimum-image) forces without any error. Build without one of the two flags, or implement \
this combination deliberately (derive the quadrupole short-range split) before removing this guard."
#endif

// Phase 4 (PLAN.md): TreePM short-range/long-range split, matching Gadget-2's own
// force_treeevaluate_shortrange exactly (forcetree.c:1420-1729, GADGET2_NOTES.md). **Gated behind
// PMGRID alone, not PERIODIC** (LOG.md §34) -- this is Gadget-2's own structure, confirmed by
// reading forcetree.c directly: force_treeevaluate_shortrange is `#ifdef PMGRID`-only, with
// `#ifdef PERIODIC` used *inside* it only for the minimum-image wrap. Gadget-2 has no mode where
// PM runs without a paired tree short-range correction (confirmed with the user directly) -- it's
// always Tree-only or Tree+PM, never PM-only -- so a non-periodic PMGRID build needs this same
// erfc suppression just as much as a periodic one, using Asmth/Rcut derived from
// TotalMeshSize/GRID instead of BoxSize/PMGRID (pm_nonperiodic.c:125-126 vs pm_periodic.c:60-61,
// same formula shape). Verified by rebuilding every existing build dir after this change and
// re-running their existing tests, not just assumed safe.
//
// __constant__ memory (not extra kernel parameters) for the cutoff constants and the shortrange
// table: these are set once per run (matching Gadget-2's own All.Rcut/All.Asmth globals), and
// threading them through every already-templated function signature in this file
// (approximate_gravity, add_acc, split_node_grav_*, directAcc, approxAcc) would be far more
// invasive than one host-side hipMemcpyToSymbol call at startup.
#ifdef PMGRID
#define GADGET_HIP_NTAB 1000
__constant__ float  g_pm_rcut2;
__constant__ float  g_pm_asmthfac;
__constant__ float  g_pm_shortrange_table[GADGET_HIP_NTAB];
// Gadget-2 uses a SEPARATE table for the potential's own erfc suppression, not the force table
// above (forcetree.c:2676: shortrange_table_potential[i] = erfc(u), vs. the force table's
// erfc(u) + 2u/sqrt(pi)*exp(-u^2)) -- the extra term in the force table comes from differentiating
// erfc(alpha r)/r via the product rule, so it's specific to the force, not the potential itself.
// Found 2026-08-28 while wiring PM's own potential into acc.w: this port's add_acc() had been
// reusing the FORCE table for the potential term too (a real bug, though one that only affects
// the .w/energy-diagnostic accounting, never the actual dynamics, since integration only reads
// .xyz) -- fixed here with its own correctly-tabulated array.
__constant__ float  g_pm_shortrange_table_potential[GADGET_HIP_NTAB];

// Shared table-construction logic for both the periodic and isolated setup entry points below --
// only Rcut/Asmth differ between them (derived from BoxSize/PMGRID vs. TotalMeshSize/GRID), the
// erfc table formula itself is identical, matching how Gadget-2 shares force_treeevaluate_
// shortrange verbatim between both cases and only varies All.Rcut[0]/All.Asmth[0]'s derivation.
static void pm_upload_shortrange_tables(float rcut, float asmth)
{
  const float rcut2    = rcut * rcut;
  const float asmthfac = 0.5f / asmth * (GADGET_HIP_NTAB / 3.0f);

  // Gadget-2's own shortrange_table formula, verbatim (forcetree.c:2670-2677, GADGET2_NOTES.md):
  // u ranges over [0,3) at NTAB points; table[i] = erfc(u) + 2u/sqrt(pi)*exp(-u^2) for force,
  // plain erfc(u) for potential (forcetree.c:2676 -- a distinct table, not a simplification of
  // the force one).
  float h_table[GADGET_HIP_NTAB];
  float h_table_pot[GADGET_HIP_NTAB];
  for (int i = 0; i < GADGET_HIP_NTAB; i++)
  {
    const double u = 3.0 / GADGET_HIP_NTAB * (i + 0.5);
    h_table[i]     = (float)(std::erfc(u) + 2.0 * u / std::sqrt(M_PI) * std::exp(-u * u));
    h_table_pot[i] = (float)std::erfc(u);
  }

  hipMemcpyToSymbol(g_pm_rcut2, &rcut2, sizeof(float));
  hipMemcpyToSymbol(g_pm_asmthfac, &asmthfac, sizeof(float));
  hipMemcpyToSymbol(g_pm_shortrange_table, h_table, sizeof(h_table));
  hipMemcpyToSymbol(g_pm_shortrange_table_potential, h_table_pot, sizeof(h_table_pot));
}

#ifdef GADGET_HIP_HIGHRES
// Phase 5 ticket 07 (PLAN.md): the zoom (PLACEHIGHRESREGION) fine grid's own short-range cutoff
// pair, selected per TARGET particle inside add_acc() below via the runtime zoom bitmask --
// mirrors Gadget-2's own force_treeevaluate_shortrange() exactly (verified against
// Gadget-2.0.7/Gadget2/timestep.c:623-628's identical "Asmth[0] normally, Asmth[1] for
// high-res-typed" selection, the tree-side counterpart of which lives in forcetree.c using
// the same All.Asmth[0]/[1] globals). The shortrange_table[]/shortrange_table_potential[] erfc
// lookup tables above are shared UNCHANGED between both grids -- their VALUES depend only on the
// dimensionless u=r*asmthfac (index i in [0,NTAB) maps to u in [0,3)), never on Rcut/Asmth
// themselves; only the two scalar SCALE factors (asmthfac, rcut2) differ per grid, exactly like
// g_pm_rcut2/g_pm_asmthfac above.
__constant__ float        g_pm_rcut2_1;
__constant__ float        g_pm_asmthfac_1;
__constant__ unsigned int g_pm_zoom_mask;

// Called whenever the zoom region is (re)computed (octree::recomputeZoomRegion(), gpu_iterate.cpp)
// -- Rcut[1]/Asmth[1] can change reactively during a run (a high-res particle leaving the current
// region triggers a recompute), unlike Rcut[0]/Asmth[0] which are fixed for the whole run (set
// once via pm_periodic_setup_gravity_kernel/pm_isolated_setup_gravity_kernel above).
// `g_pm_zoom_mask` defaults to 0 (zero-initialized __constant__ memory) until this is ever called
// -- with mask=0 the `(1u<<type) & mask` test below is always false, so every target
// unconditionally uses grid 0's Rcut/Asmth, exactly matching a non-zoom run. Re-uploaded here
// alongside rcut1/asmth1 for simplicity even though it doesn't itself change reactively --
// negligible cost, avoids a second upload entry point.
void pm_zoom_upload_rcut_asmth(float rcut1, float asmth1, unsigned int zoomMask)
{
  const float rcut2_1    = rcut1 * rcut1;
  const float asmthfac_1 = 0.5f / asmth1 * (GADGET_HIP_NTAB / 3.0f);
  hipMemcpyToSymbol(g_pm_rcut2_1, &rcut2_1, sizeof(float));
  hipMemcpyToSymbol(g_pm_asmthfac_1, &asmthfac_1, sizeof(float));
  hipMemcpyToSymbol(g_pm_zoom_mask, &zoomMask, sizeof(unsigned int));
}
#endif
#endif

// Ticket T27 (gadget-audit map): softening-based bJ floor, modeled on Gadget-2's own real
// force_treebuild_single mechanism (forcetree.c: `if(nfreep->len < 1.0e-3*epsilon) randomize...`)
// -- a node whose size has shrunk pathologically small relative to the local softening scale is,
// per Gadget-2's own reasoning, "well below gravitational softening length-scale anyway" so
// further geometric precision there can't affect the (already-softened) force result. Unlike
// Ticket 24's `0.35*g_periodic_boxSize` floor, this is available regardless of periodicity (no
// box size needed) and scales with the physical softening scale rather than an arbitrary box
// fraction. g_t27_soft_floor_coef == 0.0f (the default) means "floor disabled" -- production
// behavior is then governed entirely by whatever box-size floor is compiled in, unchanged. Do NOT
// assume Gadget-2's literal 1.0e-3 coefficient transfers -- see t27_set_soft_floor_coef's own
// comment for why this needs empirical calibration on the port's own geometry.
__constant__ float g_t27_soft_floor_coef   = 0.0f;
__constant__ int   g_t27_disable_box_floor = 0;
__constant__ int   g_t31_force_geo = 0;   // T31 diagnostic: force the geometric MAC branch

// T28/C-B-15 instrumentation: how often the mixed-softening forced descent actually fires, and how
// often the "no contributor" sentinel is hit (Gadget's endrun(987) case). A rule never observed to
// fire is indistinguishable from one that cannot fire (PLAN.md rule 8).
// A/B control: GADGET_HIP_T28_DISABLE=1 turns OFF both mixed-softening rules (C-C-08's
// max(h_target, h_node) and C-B-15's forced descent) in the SAME binary, so the comparison against
// Gadget-2 cannot be confounded by a rebuild. Default 0 = rules active.
__device__ int g_t28_disable = 0;
__device__ unsigned int g_cb15_forced = 0;
__device__ unsigned int g_cb15_undefined = 0;

void t28_set_disable(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t28_disable), &on, sizeof(on));
}

void cb15_reset_counters()
{
  unsigned int z = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_cb15_forced), &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_cb15_undefined), &z, sizeof(z));
}

void cb15_read_counters(unsigned int *forced, unsigned int *undefinedNode)
{
  hipMemcpyFromSymbol(forced, HIP_SYMBOL(g_cb15_forced), sizeof(*forced));
  hipMemcpyFromSymbol(undefinedNode, HIP_SYMBOL(g_cb15_undefined), sizeof(*undefinedNode));
}

// Walk stack-overflow counter. treewalk() returns a 0xFFFFFFFF sentinel when a group's cell list
// outgrows CELL_LIST_MEM_PER_WARP, and approximate_gravity_main turns that into `success = false`.
// The ONLY consumer of that flag is a retry loop under `#ifdef SHMODE` -- and SHMODE is not defined
// anywhere in this project, so the flag was silently discarded and the affected group's particles
// were left with NO force for that step (the acc write sits after the early return). Count them so
// the host can refuse rather than integrate a silently incomplete force.
__device__ unsigned int g_walk_bailouts = 0;

// T49 item A: how many times the walk tripped the cell-list stack guard ONLY because the guard now
// accounts for nextLevelCellCounter. Nonzero means a pre-fix build was wrapping the ring buffer at
// those steps and walking cell indices from the wrong level; zero means the defect stayed dormant.
__device__ unsigned int g_cell_guard_window = 0;

// Phase 5: ported from the phase4-node-maintenance branch, where it was built for T48. The HIGH-WATER
// MARK of cell-list occupancy -- how close the widest group's frontier came to capacity. The bailout
// counter says IF a group overflowed; this says how much margin there was when it did not, which is
// the difference between "the buffer is sized too tightly for this problem" and "this configuration
// opens pathologically more nodes". `_old` keeps the pre-T49-A expression so the two are comparable.
__device__ unsigned int g_cell_high_water     = 0;
__device__ unsigned int g_cell_high_water_old = 0;

// Phase 5 / T49 item B: the big-stack RETRY. A group whose cell list overflows its warp's share is
// re-walked on the whole second half of MEM_BUF with SHIFT=8 (capacity x256), serialised by a
// SPIN LOCK -- one group at a time across the entire GPU, with every other block busy-waiting. The
// path has always been live (SHMODE is undefined project-wide), but its outcome was checked with
// assert(), which NDEBUG deletes from a Release build, while g_walk_bailouts was incremented BEFORE
// it ran. These separate "needed a retry" from "the retry also failed" -- only the latter is a lost
// force, and the former is what makes a step take minutes instead of seconds.
__device__ unsigned int g_walk_retries    = 0;
__device__ unsigned int g_walk_retry_fail  = 0;

// T50 section 2: how many group-walks pruned at Rcut[1] because every one of the group's targets is
// high-res, vs. how many had to fall back to max(Rcut[0],Rcut[1]) because the group is mixed (or the
// zoom is inactive). The optimisation rests on the claim that a zoom's groups are almost never mixed
// -- groups are spatially compact and the high-res species occupies a small sub-volume -- so count it
// instead of believing it.
__device__ unsigned int g_zoom_pure_groups  = 0;   // every target high-res  -> prunes at Rcut[1]
__device__ unsigned int g_zoom_mixed_groups = 0;   // some high-res, some not -> max(Rcut[0],Rcut[1])
// The first version had two buckets, and \"mixed\" then also counted every group holding NO high-res
// particle at all -- which is not mixed, it is correctly using Rcut[0]. With this IC's ~23.1M
// low-resolution particles that was ~0.7M of the 1.1M groups reported as mixed, so the
// genuinely-mixed fraction (the zoom region's boundary shell) read about 3x too high.
__device__ unsigned int g_zoom_coarse_groups = 0;  // no high-res target at all -> Rcut[0], correct

// A/B switch for the pruning change itself (GADGET_HIP_T50_NO_PURE_PRUNE=1): forces every group back
// onto max(Rcut[0],Rcut[1]), the pre-T50 behaviour. Without this the claim "the per-group prune is
// what makes a small PMGRID runnable" would rest on a note about a DIFFERENT configuration rather
// than on an A/B of the same binary.
__constant__ int g_t50_no_pure_prune = 0;

void t50_set_no_pure_prune(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t50_no_pure_prune), &on, sizeof(on));
}

// T55: which GEOMETRY reject_node_rcut() uses. Default 1 (per-axis at Rcut), see that function.
//   0 = Euclidean gap vs Rcut            -- the pre-T55 behaviour, kept for A/B
//   1 = per-axis gap vs Rcut             -- Gadget's geometry (forcetree.c:1583-1605)
//   2 = per-axis gap vs 6*Asmth          -- the erfc table's own reach, a guaranteed superset
__constant__ int g_t55_rcut_mode = 1;

void t55_set_rcut_mode(int mode)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t55_rcut_mode), &mode, sizeof(mode));
}

void walk_reset_bailouts()
{
  unsigned int z = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_walk_bailouts), &z, sizeof(z));
}

void walk_reset_guard_window()
{
  unsigned int z = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_cell_guard_window), &z, sizeof(z));
}

unsigned int walk_read_guard_window()
{
  unsigned int v = 0;
  hipMemcpyFromSymbol(&v, HIP_SYMBOL(g_cell_guard_window), sizeof(v));
  return v;
}

void walk_reset_high_water()
{
  unsigned int z = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_cell_high_water),     &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_cell_high_water_old), &z, sizeof(z));
}

void walk_read_high_water(unsigned int *now, unsigned int *oldExpr)
{
  hipMemcpyFromSymbol(now,     HIP_SYMBOL(g_cell_high_water),     sizeof(*now));
  hipMemcpyFromSymbol(oldExpr, HIP_SYMBOL(g_cell_high_water_old), sizeof(*oldExpr));
}

unsigned int walk_cell_list_capacity() { return (unsigned int) CELL_LIST_MEM_PER_WARP; }

void walk_reset_retries()
{
  unsigned int z = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_walk_retries),    &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_walk_retry_fail), &z, sizeof(z));
}

void walk_read_retries(unsigned int *retries, unsigned int *failed)
{
  hipMemcpyFromSymbol(retries, HIP_SYMBOL(g_walk_retries),    sizeof(*retries));
  hipMemcpyFromSymbol(failed,  HIP_SYMBOL(g_walk_retry_fail), sizeof(*failed));
}

void walk_reset_zoom_groups()
{
  unsigned int z = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_zoom_pure_groups),   &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_zoom_mixed_groups),  &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_zoom_coarse_groups), &z, sizeof(z));
}

void walk_read_zoom_groups(unsigned int *pure, unsigned int *mixed, unsigned int *coarse)
{
  hipMemcpyFromSymbol(pure,   HIP_SYMBOL(g_zoom_pure_groups),   sizeof(*pure));
  hipMemcpyFromSymbol(mixed,  HIP_SYMBOL(g_zoom_mixed_groups),  sizeof(*mixed));
  hipMemcpyFromSymbol(coarse, HIP_SYMBOL(g_zoom_coarse_groups), sizeof(*coarse));
}

unsigned int walk_read_bailouts()
{
  unsigned int v = 0;
  hipMemcpyFromSymbol(&v, HIP_SYMBOL(g_walk_bailouts), sizeof(v));
  return v;
}

// Host-side one-time setup, called unconditionally from main.cpp (not gated by PERIODIC/PMGRID --
// softening-based flooring makes sense for any build). Coefficient and box-floor-disable flag are
// both env-var overridable so a bisection-style calibration sweep (matching Ticket 24's own
// bJ-floor bisection methodology) doesn't require a rebuild per data point:
//   GADGET_HIP_T27_SOFT_COEF=<float>       -- bJ floor = coef * groupMaxSoftening (0 = disabled)
//   GADGET_HIP_T27_DISABLE_BOX_FLOOR=1     -- zero out Ticket 24's 0.35*boxSize floor (PERIODIC
//                                              builds only; lets the softening floor be tested
//                                              alone, "replace" mode, vs. combined with the box
//                                              floor via max(), "complement" mode, the default).
void t27_set_soft_floor_coef(float defaultCoef)
{
  float coef = defaultCoef;
  const char *e = getenv("GADGET_HIP_T27_SOFT_COEF");
  if (e) coef = atof(e);
  // Ticket 24's 0.35*boxSize floor is DISABLED BY DEFAULT as of the T4 resolution: it was
  // compensating for the G-convention unit error in `aold`, which is now fixed at source. Both it
  // and Ticket 27's K*softening floor are invented constants with no Gadget-2 counterpart.
  // GADGET_HIP_T27_DISABLE_BOX_FLOOR=0 restores the floor for A/B work.
  int disableBox = 1;
  const char *d = getenv("GADGET_HIP_T27_DISABLE_BOX_FLOOR");
  if (d) disableBox = atoi(d);
  hipMemcpyToSymbol(HIP_SYMBOL(g_t27_soft_floor_coef), &coef, sizeof(float));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t27_disable_box_floor), &disableBox, sizeof(int));
  fprintf(stderr, "[T27-SOFT-FLOOR] coef=%.6g disableBoxFloor=%d\n", coef, disableBox);
}

// T31 diagnostic: force the geometric MAC branch regardless of groupMaxAcc.
void t31_set_force_geo(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t31_force_geo), &on, sizeof(int));
}

// ---------------------------------------------------------------------------
// Ticket T4 (contract C-B-02): does the Springel MAC's PRIMARY relative test ever actually fire?
// Gadget-2 forms `aold` from the un-G'd acceleration (gravtree.c:317, computed BEFORE the
// `GravAccel *= All.G` at :325-328), so at forcetree.c:1279 both `mass` and `aold` are G-free.
// The port scales the whole accumulated acceleration buffer by G (gpu_iterate.cpp:947-949) before
// it becomes acc0, so the `groupMaxAcc` feeding `aold` below carries a factor of G that `mJ` (a
// bare summed multipole mass) does not. At this project's own cosmological G=43007.1 that inflates
// the per-node relative error budget from ErrTolForceAcc=0.005 to ~215 and shrinks the opening
// radius by G^(1/4)~14.4x. Prediction under the bug: primary-test opens ~= 0, with essentially
// every split coming from the proximity box or T28's bJ<=0 rule instead.
//
// Deliberately a plain per-lane atomic, NOT a __ballot_sync-based warp aggregation: this kernel's
// warp-intrinsic semantics under hipify are themselves an open audit item, and perturbing the
// thing being measured would be far worse than the diagnostic being slow. It is only ever enabled
// for a short iteration window (default 1-3, see GADGET_HIP_T4_COUNT in gpu_iterate.cpp).
// Zero cost when g_t4_count == 0.
// T9 (contract C-B-06): use the fixed octree cell side as bJ instead of the AABB extent.
__constant__ int g_t9_cellsize_bj = 0;

void t9_set_cellsize_bj(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t9_cellsize_bj), &on, sizeof(int));
  fprintf(stderr, "[T9-CELLSIZE-BJ] Springel MAC node size = %s\n",
          on ? "fixed octree cell (Gadget-2 len)" : "AABB extent (legacy)");
}

__device__   unsigned long long g_t4_openPrimary = 0;  // primary relative test fired
__device__   unsigned long long g_t4_openProx    = 0;  // proximity/safety box fired
__device__   unsigned long long g_t4_openBJzero  = 0;  // T28's bJ<=0 single-particle force-open
__device__   unsigned long long g_t4_reject      = 0;  // node approximated (not split)
__constant__ int g_t4_count = 0;

static __device__ __forceinline__ void t4_tally(unsigned long long *counter)
{
  if (g_t4_count) atomicAdd(counter, 1ULL);
}

void t4_counters_set(int enabled)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t4_count), &enabled, sizeof(int));
}

void t4_counters_reset()
{
  unsigned long long z = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_t4_openPrimary), &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t4_openProx),    &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t4_openBJzero),  &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t4_reject),      &z, sizeof(z));
}

void t4_counters_read(unsigned long long &primary, unsigned long long &prox,
                      unsigned long long &bjzero,  unsigned long long &reject)
{
  hipMemcpyFromSymbol(&primary, HIP_SYMBOL(g_t4_openPrimary), sizeof(primary));
  hipMemcpyFromSymbol(&prox,    HIP_SYMBOL(g_t4_openProx),    sizeof(prox));
  hipMemcpyFromSymbol(&bjzero,  HIP_SYMBOL(g_t4_openBJzero),  sizeof(bjzero));
  hipMemcpyFromSymbol(&reject,  HIP_SYMBOL(g_t4_reject),      sizeof(reject));
}
// ---------------------------------------------------------------------------

#ifdef PERIODIC
__constant__ float  g_periodic_boxSize;
__constant__ float  g_periodic_boxHalf;

// Gadget-2's own NEAREST() macro (forcetree.c:43), verbatim: minimum-image wrap of one
// coordinate difference into (-boxHalf, boxHalf]. Periodic-only -- an isolated (non-periodic)
// system has no wraparound at all, matching Gadget-2's own force_treeevaluate_shortrange, which
// only calls NEAREST() inside its own #ifdef PERIODIC block.
static __device__ __forceinline__ float pm_nearest(float x)
{
  if (x > g_periodic_boxHalf) return x - g_periodic_boxSize;
  if (x < -g_periodic_boxHalf) return x + g_periodic_boxSize;
  return x;
}

// Host-side one-time setup for the PERIODIC case (declared in pm.h, called once from main.cpp
// before any gravity kernel launch). Defined in this .cu file, not pm_*.cu/.cpp, because
// hipMemcpyToSymbol needs the __constant__ symbols visible in the same translation unit.
// C-C-03: the direct-gravity kernel lives in another translation unit and so cannot read the
// __constant__ box size. Cache it here as the setup runs, and expose it, rather than plumbing the
// value separately from main.cpp and risking the two disagreeing.
static float s_periodic_boxSize_host = 0.0f;
float pm_get_periodic_boxsize() { return s_periodic_boxSize_host; }

void pm_periodic_setup_gravity_kernel(float boxSize, float rcut, float asmth)
{
  s_periodic_boxSize_host = boxSize;
  const float boxHalf = 0.5f * boxSize;
  hipMemcpyToSymbol(g_periodic_boxSize, &boxSize, sizeof(float));
  hipMemcpyToSymbol(g_periodic_boxHalf, &boxHalf, sizeof(float));
  pm_upload_shortrange_tables(rcut, asmth);
}
#endif

#if defined(PMGRID) && !defined(PERIODIC)
// Host-side one-time setup for the isolated (non-periodic) TreePM case (LOG.md §34) -- the
// tree-side counterpart to pm_solver_create_isolated's own PM setup. No boxSize/boxHalf upload
// (no wrap needed), just the shared erfc tables, with Rcut/Asmth derived from the isolated PM
// solver's own TotalMeshSize/GRID convention (pm_nonperiodic.c:125-126) instead of periodic's
// BoxSize/PMGRID.
void pm_isolated_setup_gravity_kernel(float rcut, float asmth)
{
  pm_upload_shortrange_tables(rcut, asmth);
}
#endif

/***********************************/
/***** DENSITY   ******************/

static __device__ __forceinline__ void computeDensityAndNgb(
    const float r2, const float hinv2, const float mass,
    float &density, float &nb)
{
#if 0  /* full kernel for reference */
  const float hinv = 1.0f/h;
  const float hinv2 = hinv*hinv;
  const float hinv3 = hinv*hinv2;
  const float C     = 3465.0f/(512.0f*M_PI)*hinv3;
  const float q2    = r2*hinv2;
  const float rho   = fmaxf(0.0f, 1.0f - q2);
  nb      += ceilf(rho);
  const float rho2 = rho*rho;
  density += C * rho2*rho2;
#else
  const float rho   = fmaxf(0.0f, 1.0f - r2*hinv2);   /* fma, fmax */
  const float rho2  = rho*rho;                        /* fmul */
  density += rho2*rho2;                               /* fma */
  nb      += ceilf(rho2);                             /* fadd, ceil */

  /*2x fma, 1x fmul, 1x fadd, 1x ceil, 1x fmax */
  /* total: 6 flops or 8 flops with ceil&fmax */
#endif
}

#if 0
static __device__ __forceinline__ float adjustH(const float h_old, const float nnb)
{
	const float nbDesired 	= 42;
	const float f      	= 0.5f * (1.0f + cbrtf(nbDesired / nnb));
	const float fScale 	= max(min(f, 1.2), 0.8);
	return (h_old*fScale);
}
#endif





/************************************/
/*********   PREFIX SUM   ***********/
/************************************/

// Ported off raw NVIDIA PTX (shfl.sync.up.b32, %lanemask_lt/%lanemask_le special
// registers) to portable HIP __shfl_up(); the PTX clamp-at-boundary-0 semantics is exactly
// what __shfl_up already does (returns the caller's own value when the source lane would be
// out of range), so the predicate the PTX computed is reproduced with a plain lane-index/
// distance compare instead of an inline-asm predicate register.
static __device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
{
  uint result = __shfl_up(partial, up_offset);
  const unsigned lane = threadIdx.x & (warpSize - 1);
  if (lane >= up_offset) result += partial;
  return result;
}

static __device__ __forceinline__ int lanemask_lt()
{
  const unsigned lane = threadIdx.x & (warpSize - 1);
  return (int)((1u << lane) - 1u);
}

static __device__ __forceinline__ int lanemask_le()
{
  const unsigned lane = threadIdx.x & (warpSize - 1);
  return (int)((2u << lane) - 1u);
}

static __device__ __forceinline__ int ShflSegScanStepB(
            int partial,
            uint distance,
            uint up_offset)
{
  int r0 = __shfl_up(partial, up_offset);
  if (up_offset <= distance) partial = r0 + partial;
  return partial;
}

  template<const int SIZE2>
static __device__ __forceinline__ int inclusive_segscan_warp_step(int value, const int distance)
{
  for (int i = 0; i < SIZE2; i++)
    value = ShflSegScanStepB(value, distance, 1<<i);
  return value;
}

  template <const int levels>
static __device__ __forceinline__ uint inclusive_scan_warp(const int sum)
{
  uint mysum = sum;
#pragma unroll
  for(int i = 0; i < levels; ++i)
    mysum = shfl_scan_add_step(mysum, 1 << i);
  return mysum;
}

/*********************/

static __device__ __forceinline__ int2 warpIntExclusiveScan(const int value)
{
  const int sum = inclusive_scan_warp<WARP_SIZE2>(value);
  return make_int2(sum-value, __shfl_sync(FULL_MASK, sum, WARP_SIZE-1, WARP_SIZE));
}

static __device__ __forceinline__ int2 warpBinExclusiveScan(const bool p)
{
  const unsigned int b = __ballot_sync(FULL_MASK, p);
  return make_int2(__popc(b & lanemask_lt()), __popc(b));
}


static __device__ __forceinline__ int2 inclusive_segscan_warp(
    const int packed_value, const int carryValue)
{
  const int  flag = packed_value < 0;
  const int  mask = -flag;
  const int value = (~mask & packed_value) + (mask & (-1-packed_value));

  const int flags = __ballot_sync(FULL_MASK, flag);

  const int dist_block = __clz(__brev(flags));

  const int distance = __clz(flags & lanemask_le()) + laneId - 31;
  const int val = inclusive_segscan_warp_step<WARP_SIZE2>(value, min(distance, laneId));
  return make_int2(val + (carryValue & (-(laneId < dist_block))), __shfl_sync(FULL_MASK, val, WARP_SIZE-1, WARP_SIZE));
}

/**** binary scans ****/


#if 0
static __device__ int warp_exclusive_scan(const bool p, int &psum)
{
  const unsigned int b = __ballot(p);
  psum = __popc(b & lanemask_lt());
  return __popc(b);
}
static __device__ int warp_exclusive_scan(const bool p)
{
  const int b = __ballot(p);
  return __popc(b & lanemask_lt());
}
#endif


/**************************************/
/*************** Tree walk ************/
/**************************************/

  template<int SHIFT>
__forceinline__ static __device__ int ringAddr(const int i)
{
  return (i & ((CELL_LIST_MEM_PER_WARP<<SHIFT) - 1));
}


/*********** Forces *************/

// Phase 5 ticket 03 (PLAN.md): Gadget-2's own cubic-spline (Monaghan) softening kernel, exact
// formula -- NOT an approximation -- replacing this port's prior flat-Plummer `1/sqrt(r2+eps2)`
// softening entirely (PHASE5_ZOOM_COMOVING_SPEC.md Sec 4.1, forcetree.c:1356-1371 force,
// forcetree.c:2263-2280 potential). `h` is the already-resolved softening LENGTH for this
// specific interaction (not squared, unlike the old `eps2`) -- callers are responsible for
// picking the right `h` (target-only for an approximated/monopole node, since a node has no
// single "type" of its own; max(target,source) `ForceSoftening` for an actual leaf-particle
// source, matching Gadget-2's own `UNEQUALSOFTENINGS` behavior, made unconditional in this port
// per PLAN.md Phase 5.5's own decision -- see directAcc() below). `h<=0` is a sentinel meaning
// "softening disabled" (used only by synthetic diagnostics that want a pure point-mass
// reference): the `r>=h` branch below is taken unconditionally whenever `h<=0`, since `r>=0>=h`
// always holds then, so no division by a non-positive `h` ever occurs.
// Phase 5 ticket 07 (PLAN.md): `type` is the TARGET's own Gadget-2 particle type (0-5), used ONLY
// to select which (Rcut,Asmth) pair the PMGRID short-range suppression below uses -- unlike the
// softening length `h` above, this is never combined with the source's own type (a node has no
// single "type" anyway, and Gadget-2's own selection is target-only, timestep.c:623-628 / the
// force_treeevaluate_shortrange counterpart it mirrors).
static __device__ __forceinline__ float4 add_acc(
    float4 acc,  const float4 pos,
    const float massj, const float3 posj,
    const float h,
    const int type,
    float2 &density)
{
#ifdef PERIODIC
  const float3 dr = make_float3(pm_nearest(posj.x - pos.x), pm_nearest(posj.y - pos.y),
                                 pm_nearest(posj.z - pos.z));
#else
  const float3 dr = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);
#endif

  const float r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  const float r  = sqrtf(r2);

  // mrinv: subtracted from acc.w (potential); mrinv3: multiplies dr to form the force vector.
  // Named to match this file's own pre-existing convention (mrinv/mrinv3), even though neither is
  // a literal "mass*rinv" anymore outside the r>=h branch.
  float mrinv, mrinv3;

  if (r >= h)
  {
    // Exact Newtonian, no softening -- also the branch always taken when h<=0 (see above).
    const float rinv  = (r2 > 0.0f) ? rsqrtf(r2) : 0.0f;
    const float rinv3 = rinv*rinv*rinv;
    mrinv  = massj * rinv;
    mrinv3 = massj * rinv3;
  }
  else
  {
    const float h_inv  = 1.0f / h;
    const float h3_inv = h_inv * h_inv * h_inv;
    const float u      = r * h_inv;
    float wp;
    if (u < 0.5f)
    {
      mrinv3 = massj * h3_inv * (10.666666667f + u*u*(32.0f*u - 38.4f));
      wp     = -2.8f + u*u*(5.333333333f + u*u*(6.4f*u - 9.6f));
    }
    else
    {
      mrinv3 = massj * h3_inv * (21.333333333f - 48.0f*u + 38.4f*u*u -
                                  10.666666667f*u*u*u - 0.066666667f/(u*u*u));
      wp     = -3.2f + 0.066666667f/u +
               u*u*(10.666666667f + u*(-16.0f + u*(9.6f - 2.133333333f*u)));
    }
    // acc.w -= mrinv  ==>  acc.w += massj*h_inv*wp, matching Gadget-2's `pot += mass*h_inv*wp`.
    mrinv = -massj * h_inv * wp;
  }

#ifdef PMGRID
  // Gadget-2's own TreePM short-range/long-range split (forcetree.c:1718-1729,
  // GADGET2_NOTES.md): suppress the tree's contribution via a tabulated complementary error
  // function of the UNSOFTENED distance so it falls to zero right where PM takes over, instead
  // of double-counting. Interactions beyond the table's range (r >= 3*Asmth) contribute nothing
  // at all, matching Gadget-2's own `if(tabindex < NTAB)` guard exactly -- not an approximation.
  // Gated by PMGRID alone (LOG.md §34), not PERIODIC -- Gadget-2 never runs PM without a paired
  // tree short-range correction, periodic or not (confirmed with the user directly). This
  // multiplicative suppression composes correctly with the spline softening above exactly as it
  // did with the old Plummer formula: it's a separate correction for r itself (the short/long
  // split), orthogonal to how the near-field softening kernel shapes force/potential vs. r
  // (confirmed against Gadget-2's own force_treeevaluate_shortrange(), which applies the spline
  // kernel first and this same erfc suppression immediately after, forcetree.c:1710-1723).
#ifdef GADGET_HIP_HIGHRES
  // Phase 5 ticket 07: high-res-typed targets use the fine grid's own (Rcut[1],Asmth[1]) pair
  // instead of the coarse grid's (Rcut[0],Asmth[0]) -- see g_pm_zoom_mask's own doc comment above.
  const bool  useFine  = (bool)((1u << type) & g_pm_zoom_mask);
  const float asmthfac = useFine ? g_pm_asmthfac_1 : g_pm_asmthfac;
#else
  const float asmthfac = g_pm_asmthfac;
#endif
  const int tabindex = (int)(asmthfac * r);
  if (tabindex >= GADGET_HIP_NTAB)
    return acc;
  // Two different tables, matching Gadget-2 exactly (forcetree.c:2670-2677): the force uses
  // erfc(u)+2u/sqrt(pi)*exp(-u^2) (mrinv3, the potential's own table would be wrong here since
  // the extra term comes from differentiating erfc(alpha r)/r), the potential uses plain erfc(u)
  // (mrinv) -- using the force table for both (as this file did until this fix) silently
  // mis-suppresses the tree's own potential/.w term; the force/.xyz term was always correct.
  mrinv  *= g_pm_shortrange_table_potential[tabindex];
  mrinv3 *= g_pm_shortrange_table[tabindex];
#endif

  acc.w -= mrinv;
  acc.x += mrinv3 * dr.x;
  acc.y += mrinv3 * dr.y;
  acc.z += mrinv3 * dr.z;

#if BONSAI_DENSITY
  computeDensityAndNgb(r2,pos.w,massj,density.x,density.y);
#endif

  return acc;
}
// Phase 5 ticket 03 (PLAN.md): leaf-particle interactions -- the source `ptclIdx` is a real
// particle with its own resolved Gadget-2 `ForceSoftening`, so the softening length used here is
// `max(h_i[k], sourceForceSoftening)`, matching Gadget-2's own `UNEQUALSOFTENINGS` behavior
// (forcetree.c:1512,1530-1532: `h = ForceSoftening[ptype]; if(h < ForceSoftening[P[no].Type]) h =
// ForceSoftening[P[no].Type];`) -- made UNCONDITIONAL in this port (PLAN.md Phase 5.5's own
// decision: Gadget-2 gates this behind a legacy-performance macro with no correctness upside to
// leaving it off, and this port has no MPI import/export path where the original motivation even
// applies).
template<int NI, bool FULL>
static __device__ __forceinline__ void directAcc(
    float4 acc_i[NI],
    const float4 pos_i[NI],
    const int ptclIdx,
    const float h_i[NI],
    const int type_i[NI],
    float2 density_i[NI],
    const real4  *body_pos,
    const float  *body_forceSoftening)
{
  const float4 M0    = (FULL || ptclIdx >= 0) ? body_pos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const float  M0sft = (FULL || ptclIdx >= 0) ? body_forceSoftening[ptclIdx] : 0.0f;

//#pragma unroll
  for (int j = 0; j < WARP_SIZE; j++)
  {
    const float4 jM0 = make_float4(__shfl_sync(FULL_MASK, M0.x, j), __shfl_sync(FULL_MASK, M0.y, j),
                                   __shfl_sync(FULL_MASK, M0.z, j), __shfl_sync(FULL_MASK, M0.w,j));
    const float  jSoft = __shfl_sync(FULL_MASK, M0sft, j);
    const int    jPtclIdx = __shfl_sync(FULL_MASK, ptclIdx, j);
    const float  jmass = jM0.w;
    const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
    for (int k = 0; k < NI; k++)
    {
      const float traceBeforeY = acc_i[k].y;
      acc_i[k] = add_acc(acc_i[k], pos_i[k], jmass, jpos, fmaxf(h_i[k], jSoft), type_i[k], density_i[k]);
      if (isTraceTarget(pos_i[k]))
      {
        const float deltaY = acc_i[k].y - traceBeforeY;
        atomicAdd(&g_traceDirectY, (double)deltaY);
        atomicAdd(&g_traceDirectN, 1);
        if (g_traceVerbose)
        {
#ifdef PERIODIC
          const float drx = pm_nearest(jpos.x - pos_i[k].x);
          const float dry = pm_nearest(jpos.y - pos_i[k].y);
          const float drz = pm_nearest(jpos.z - pos_i[k].z);
#else
          const float drx = jpos.x - pos_i[k].x, dry = jpos.y - pos_i[k].y, drz = jpos.z - pos_i[k].z;
#endif
          traceRecord(jPtclIdx, /*isDirect=*/1, jpos, jmass, drx, dry, drz, deltaY);
        }
      }
    }
  }
}


#ifdef _QUADRUPOLE_

static __device__ __forceinline__ float4 add_acc(
    float4 acc,
    const float4 pos,
    const float mass, const float3 com,
    const float4 Q0,  const float4 Q1, float eps2,
    float2 &density)
{
#if 1
  const float3 dr = make_float3(pos.x - com.x, pos.y - com.y, pos.z - com.z);
  const float  r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;

  const float rinv  = rsqrtf(r2);
  const float rinv2 = rinv *rinv;
  const float mrinv  =  mass*rinv;
  const float mrinv3 = rinv2*mrinv;
  const float mrinv5 = rinv2*mrinv3;
  const float mrinv7 = rinv2*mrinv5;   // 16

  float  D0  =  mrinv;
  float  D1  = -mrinv3;
  float  D2  =  mrinv5*(  3.0f);
  float  D3  =  mrinv7*(-15.0f); // 3

  const float q11 = Q0.x;
  const float q22 = Q0.y;
  const float q33 = Q0.z;
  const float q12 = Q1.x;
  const float q13 = Q1.y;
  const float q23 = Q1.z;

  const float  q  = q11 + q22 + q33;
  const float3 qR = make_float3(
      q11*dr.x + q12*dr.y + q13*dr.z,
      q12*dr.x + q22*dr.y + q23*dr.z,
      q13*dr.x + q23*dr.y + q33*dr.z);
  const float qRR = qR.x*dr.x + qR.y*dr.y + qR.z*dr.z;  // 22

  acc.w  -= D0 + 0.5f*(D1*q + D2*qRR);
  float C = D1 + 0.5f*(D2*q + D3*qRR);
  acc.x  += C*dr.x + D2*qR.x;
  acc.y  += C*dr.y + D2*qR.y;
  acc.z  += C*dr.z + D2*qR.z;               // 23

// total: 16 + 3 + 22 + 23 = 64 flops

  return acc;
#endif
}

// Phase 5 ticket 03 (PLAN.md): quadrupole node interactions keep their OWN pre-existing flat
// squared-softening ("eps2") behavior unchanged internally -- deliberately NOT re-derived for
// Gadget-2's spline kernel. Gadget-2 itself has no quadrupole option at all (monopole-only by
// design, GADGET2_NOTES.md), the user has separately decided against using this port's own
// optional quadrupole correction, and re-deriving a quadrupole force/potential expansion around a
// spline-softened (rather than Plummer-softened) kernel is a nontrivial physics derivation with
// no reference to check it against -- out of scope here, same "no Gadget-2 reference for this
// combination" reasoning already applied elsewhere in this project (e.g. PERIODIC+QUADRUPOLE
// refused at configure time). This overload still takes `h_i[NI]` (not a scalar) purely so its
// call site in approximate_gravity() below stays uniform between the quadrupole and
// non-quadrupole builds -- it locally squares the target's own resolved h back into an `eps2`
// before calling the unchanged quadrupole add_acc() above, i.e. target-only Plummer softening at
// the node level, matching this same file's own non-quadrupole approxAcc() below in spirit
// (target-only, no per-node "type" to combine with).
// `type_i[NI]` is unused here (kept only so this overload's call site in approximate_gravity()
// stays uniform between quadrupole and non-quadrupole builds, exactly like `h_i[NI]`'s own
// existing doc comment above explains) -- the quadrupole add_acc() has no PMGRID/erfc suppression
// at all (PMGRID+QUADRUPOLE is refused at compile time), so there is no Rcut/Asmth selection to
// make here.
template<int NI, bool FULL>
static __device__ __forceinline__ void approxAcc(
    float4 acc_i[NI],
    const float4 pos_i[NI],
    float2 dens_i[NI],
    const int cellIdx,
    const float h_i[NI],
    const int type_i[NI],
    const real4 *multipole_data)
{
  (void) type_i;
  const int cellAddr = cellIdx + cellIdx + cellIdx;
  float4 M0, Q0, Q1;
  if (FULL || cellIdx >= 0)
  {
    M0 = multipole_data[cellAddr];
    Q0 = multipole_data[cellAddr + 1];
    Q1 = multipole_data[cellAddr + 2];
  }
  else
    M0 = Q0 = Q1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  for (int j = 0; j < WARP_SIZE; j++)
  {
    const float4 jM0 = make_float4(__shfl_sync(FULL_MASK, M0.x, j), __shfl_sync(FULL_MASK, M0.y, j),
                                   __shfl_sync(FULL_MASK, M0.z, j), __shfl_sync(FULL_MASK, M0.w,j));
    const float4 jQ0 = make_float4(__shfl_sync(FULL_MASK, Q0.x, j), __shfl_sync(FULL_MASK, Q0.y, j),
                                   __shfl_sync(FULL_MASK, Q0.z, j), 0.0f);
    const float4 jQ1 = make_float4(__shfl_sync(FULL_MASK, Q1.x, j), __shfl_sync(FULL_MASK, Q1.y, j),
                                   __shfl_sync(FULL_MASK, Q1.z, j), 0.0f);
    const float  jmass = jM0.w;
    const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
      for (int k = 0; k < NI; k++)
        acc_i[k] = add_acc(acc_i[k], pos_i[k], jmass, jpos, jQ0, jQ1, h_i[k]*h_i[k], dens_i[k]);

  }
}

#else

// Monopole-only counterpart of the above -- found missing entirely 2026-08-28 when
// GADGET_HIP_QUADRUPOLE was added and defaulted OFF: Bonsai's inherited kernel had no
// approxAcc() at all for the non-quadrupole case, meaning `_QUADRUPOLE_` was never actually an
// optional toggle in practice despite looking like one (the `#if 1` compiled it unconditionally,
// so this gap was never exercised). Reuses the monopole add_acc() already defined above (the same
// one directAcc() uses for leaf particles), reading only mass+COM (M0) from multipole_data and
// skipping the quadrupole tensor reads (Q0/Q1) entirely -- this is Gadget-2's own default
// behavior (GADGET2_NOTES.md: monopole-only by deliberate design), not a reduced-accuracy
// workaround.
// Phase 5 ticket 03 (PLAN.md): approximated/monopole-node interactions use the TARGET's own
// resolved ForceSoftening only (`h_i[k]`) -- a node has no single "type" of its own to combine
// with via max(), unlike directAcc()'s real leaf-particle source above. This matches Gadget-2's
// own behavior for the common case: the opening criterion already keeps approximated nodes well
// outside softening range in practice (r>=h there almost always), and Gadget-2's own node-opening
// safety check for the rare close/mixed-softening case (`maxsoft`-triggered forced descent,
// forcetree.c:1333-1341) is a documented, deliberate simplification NOT implemented in this port
// -- see PLAN.md Phase 5.5 / ticket 03's closing notes.
template<int NI, bool FULL>
static __device__ __forceinline__ void approxAcc(
    float4 acc_i[NI],
    const float4 pos_i[NI],
    float2 dens_i[NI],
    const int cellIdx,
    const float h_i[NI],
    const int type_i[NI],
    const real4 *multipole_data,
    const float *nodeSoftInfo)     // T28/C-C-08 (nullptr when UNEQUALSOFTENINGS is off)
{
  const int cellAddr = cellIdx + cellIdx + cellIdx;
  float4 M0;
  if (FULL || cellIdx >= 0)
    M0 = multipole_data[cellAddr];
  else
    M0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  // T28/C-C-08: an approximated node stands for members that may have a LARGER softening than the
  // target, and Gadget-2 softens such an interaction with max(h_target, node max) -- forcetree.c:
  // `if(h < All.ForceSoftening[maxsofttype]) h = All.ForceSoftening[maxsofttype];`. The port used
  // the target's h alone, which coincides with Gadget only while every type shares one softening.
  // One extra float load per cell, and only when the feature is compiled in.
  float nSoft = 0.0f;
#ifdef UNEQUALSOFTENINGS
  if (nodeSoftInfo && !g_t28_disable && (FULL || cellIdx >= 0)) nSoft = fabsf(nodeSoftInfo[cellIdx]);
#endif

  for (int j = 0; j < WARP_SIZE; j++)
  {
    const float4 jM0 = make_float4(__shfl_sync(FULL_MASK, M0.x, j), __shfl_sync(FULL_MASK, M0.y, j),
                                   __shfl_sync(FULL_MASK, M0.z, j), __shfl_sync(FULL_MASK, M0.w,j));
    const int    jCellIdx = __shfl_sync(FULL_MASK, cellIdx, j);
    const float  jmass = jM0.w;
    const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
    const float  jNodeSoft = __shfl_sync(FULL_MASK, nSoft, j);   // T28/C-C-08
#pragma unroll
    for (int k = 0; k < NI; k++)
    {
      const float traceBeforeY = acc_i[k].y;
      acc_i[k] = add_acc(acc_i[k], pos_i[k], jmass, jpos,
                         fmaxf(h_i[k], jNodeSoft),   // T28/C-C-08: max(target, node max)
                         type_i[k], dens_i[k]);
      if (isTraceTarget(pos_i[k]))
      {
        const float deltaY = acc_i[k].y - traceBeforeY;
        atomicAdd(&g_traceApproxY, (double)deltaY);
        atomicAdd(&g_traceApproxN, 1);
        if (g_traceVerbose)
        {
#ifdef PERIODIC
          const float drx = pm_nearest(jpos.x - pos_i[k].x);
          const float dry = pm_nearest(jpos.y - pos_i[k].y);
          const float drz = pm_nearest(jpos.z - pos_i[k].z);
#else
          const float drx = jpos.x - pos_i[k].x, dry = jpos.y - pos_i[k].y, drz = jpos.z - pos_i[k].z;
#endif
          traceRecord(jCellIdx, /*isDirect=*/0, jpos, jmass, drx, dry, drz, deltaY);
        }
      }
    }
  }
}

#endif

// Ticket 20 (tickets/T20-test-summation-order-sensitivity.md): test-only, opt-in double-precision
// force accumulator. Exact duplicates of add_acc()/directAcc()/approxAcc() (the monopole,
// non-quadrupole variants above -- the only ones this ticket's build compiles,
// GADGET_HIP_QUADRUPOLE stays OFF), with ONLY the accumulator's type changed from float4 to
// double4. Per-interaction terms (r, r2, u, mrinv, mrinv3, wp...) are still computed in float32
// exactly as in the original -- this deliberately isolates precision/order-of-summation of the
// RUNNING TOTAL, matching this ticket's method (test double-precision ACCUMULATION, not a
// broader double-precision force law). Verified by inspection before writing this: the
// __shfl_sync calls below broadcast the SOURCE particle's mass/position (jM0/jSoft) across the
// warp lanes -- never the accumulator itself, which is purely thread-local for the entire tree
// walk in the original code too -- so no double/double4 __shfl_sync handling is needed anywhere
// here. Only referenced by approximate_gravity()/treewalk() below when
// GADGET_HIP_T20_DOUBLE_FORCE_ACCUM is defined (a separate, dedicated CMake option/build,
// default OFF); the persistent bodies_acc1 buffer (`acc_out` in treewalk()) stays float4
// unconditionally -- the cast down to float happens only at treewalk()'s final write, not here.
#ifdef GADGET_HIP_T20_DOUBLE_FORCE_ACCUM
static __device__ __forceinline__ double4 add_acc_d(
    double4 acc, const float4 pos,
    const float massj, const float3 posj,
    const float h,
    const int type,
    float2 &density)
{
#ifdef PERIODIC
  const float3 dr = make_float3(pm_nearest(posj.x - pos.x), pm_nearest(posj.y - pos.y),
                                 pm_nearest(posj.z - pos.z));
#else
  const float3 dr = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);
#endif

  const float r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  const float r  = sqrtf(r2);

  float mrinv, mrinv3;

  if (r >= h)
  {
    const float rinv  = (r2 > 0.0f) ? rsqrtf(r2) : 0.0f;
    const float rinv3 = rinv*rinv*rinv;
    mrinv  = massj * rinv;
    mrinv3 = massj * rinv3;
  }
  else
  {
    const float h_inv  = 1.0f / h;
    const float h3_inv = h_inv * h_inv * h_inv;
    const float u      = r * h_inv;
    float wp;
    if (u < 0.5f)
    {
      mrinv3 = massj * h3_inv * (10.666666667f + u*u*(32.0f*u - 38.4f));
      wp     = -2.8f + u*u*(5.333333333f + u*u*(6.4f*u - 9.6f));
    }
    else
    {
      mrinv3 = massj * h3_inv * (21.333333333f - 48.0f*u + 38.4f*u*u -
                                  10.666666667f*u*u*u - 0.066666667f/(u*u*u));
      wp     = -3.2f + 0.066666667f/u +
               u*u*(10.666666667f + u*(-16.0f + u*(9.6f - 2.133333333f*u)));
    }
    mrinv = -massj * h_inv * wp;
  }

#ifdef PMGRID
#ifdef GADGET_HIP_HIGHRES
  const bool  useFine  = (bool)((1u << type) & g_pm_zoom_mask);
  const float asmthfac = useFine ? g_pm_asmthfac_1 : g_pm_asmthfac;
#else
  const float asmthfac = g_pm_asmthfac;
#endif
  const int tabindex = (int)(asmthfac * r);
  if (tabindex >= GADGET_HIP_NTAB)
    return acc;
  mrinv  *= g_pm_shortrange_table_potential[tabindex];
  mrinv3 *= g_pm_shortrange_table[tabindex];
#endif

  acc.w -= (double)mrinv;
  acc.x += (double)(mrinv3 * dr.x);
  acc.y += (double)(mrinv3 * dr.y);
  acc.z += (double)(mrinv3 * dr.z);

#if BONSAI_DENSITY
  computeDensityAndNgb(r2,pos.w,massj,density.x,density.y);
#endif

  return acc;
}

template<int NI, bool FULL>
static __device__ __forceinline__ void directAcc_d(
    double4 acc_i[NI],
    const float4 pos_i[NI],
    const int ptclIdx,
    const float h_i[NI],
    const int type_i[NI],
    float2 density_i[NI],
    const real4  *body_pos,
    const float  *body_forceSoftening)
{
  const float4 M0    = (FULL || ptclIdx >= 0) ? body_pos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const float  M0sft = (FULL || ptclIdx >= 0) ? body_forceSoftening[ptclIdx] : 0.0f;

  for (int j = 0; j < WARP_SIZE; j++)
  {
    const float4 jM0 = make_float4(__shfl_sync(FULL_MASK, M0.x, j), __shfl_sync(FULL_MASK, M0.y, j),
                                   __shfl_sync(FULL_MASK, M0.z, j), __shfl_sync(FULL_MASK, M0.w,j));
    const float  jSoft = __shfl_sync(FULL_MASK, M0sft, j);
    const float  jmass = jM0.w;
    const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
    for (int k = 0; k < NI; k++)
      acc_i[k] = add_acc_d(acc_i[k], pos_i[k], jmass, jpos, fmaxf(h_i[k], jSoft), type_i[k], density_i[k]);
  }
}

template<int NI, bool FULL>
static __device__ __forceinline__ void approxAcc_d(
    double4 acc_i[NI],
    const float4 pos_i[NI],
    float2 dens_i[NI],
    const int cellIdx,
    const float h_i[NI],
    const int type_i[NI],
    const real4 *multipole_data)
{
  const int cellAddr = cellIdx + cellIdx + cellIdx;
  float4 M0;
  if (FULL || cellIdx >= 0)
    M0 = multipole_data[cellAddr];
  else
    M0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  for (int j = 0; j < WARP_SIZE; j++)
  {
    const float4 jM0 = make_float4(__shfl_sync(FULL_MASK, M0.x, j), __shfl_sync(FULL_MASK, M0.y, j),
                                   __shfl_sync(FULL_MASK, M0.z, j), __shfl_sync(FULL_MASK, M0.w,j));
    const float  jmass = jM0.w;
    const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
    for (int k = 0; k < NI; k++)
      acc_i[k] = add_acc_d(acc_i[k], pos_i[k], jmass, jpos, h_i[k], type_i[k], dens_i[k]);
  }
}

// Type/function-name switch used by approximate_gravity()/treewalk() below: ACC_VEC_T is the
// accumulator type threaded through the whole tree walk, directAccT/approxAccT are whichever
// (original float4, or this ticket's double4) leaf/node accumulation function is actually called.
#define ACC_VEC_T double4
#define directAccT directAcc_d
#define approxAccT approxAcc_d
#else
#define ACC_VEC_T float4
#define directAccT directAcc
#define approxAccT approxAcc
#endif // GADGET_HIP_T20_DOUBLE_FORCE_ACCUM


/*******************************/
/****** Opening criterion ******/
/*******************************/

//Improved Barnes Hut criterium
#ifdef UNEQUALSOFTENINGS
// T28/C-B-15: the group-to-node minimum-image gap, factored out of split_node_grav_impbh so the
// mixed-softening forced descent can test the SAME distance the acceptance criterion uses.
// Gadget-2's rule is per-particle (`r2 < h*h`, forcetree.c:1667); this port decides per GROUP
// (C-A-12), and for any member i of the group `ds <= r_i`, so `ds2 < h*h` descends at least as
// often as a per-particle test would. Conservative in the safe direction.
static __device__ __forceinline__ float group_node_gap2(
    const float4 nodeCOM, const float4 groupCenter, const float4 groupSize)
{
#ifdef PERIODIC
  float3 dr = make_float3(
      fabsf(pm_nearest(groupCenter.x - nodeCOM.x)) - (groupSize.x),
      fabsf(pm_nearest(groupCenter.y - nodeCOM.y)) - (groupSize.y),
      fabsf(pm_nearest(groupCenter.z - nodeCOM.z)) - (groupSize.z));
#else
  float3 dr = make_float3(
      fabsf(groupCenter.x - nodeCOM.x) - (groupSize.x),
      fabsf(groupCenter.y - nodeCOM.y) - (groupSize.y),
      fabsf(groupCenter.z - nodeCOM.z) - (groupSize.z));
#endif
  dr.x += fabsf(dr.x); dr.x *= 0.5f;
  dr.y += fabsf(dr.y); dr.y *= 0.5f;
  dr.z += fabsf(dr.z); dr.z *= 0.5f;
  return dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
}
#endif

static __device__ bool split_node_grav_impbh(
    const float4 nodeCOM,
    const float4 groupCenter,
    const float4 groupSize)
{
  //Compute the distance between the group and the cell
#ifdef PERIODIC
  // Minimum-image group-to-node distance (Gadget-2's NEAREST(), forcetree.c:43/1502-1504) --
  // applied to the raw difference before the group's own half-extent is subtracted, matching
  // where Gadget-2 itself applies NEAREST() (to the raw coordinate difference, before any other
  // use of it).
  float3 dr = make_float3(
      fabsf(pm_nearest(groupCenter.x - nodeCOM.x)) - (groupSize.x),
      fabsf(pm_nearest(groupCenter.y - nodeCOM.y)) - (groupSize.y),
      fabsf(pm_nearest(groupCenter.z - nodeCOM.z)) - (groupSize.z)
      );
#else
  float3 dr = make_float3(
      fabsf(groupCenter.x - nodeCOM.x) - (groupSize.x),
      fabsf(groupCenter.y - nodeCOM.y) - (groupSize.y),
      fabsf(groupCenter.z - nodeCOM.z) - (groupSize.z)
      );
#endif

  dr.x += fabsf(dr.x); dr.x *= 0.5f;
  dr.y += fabsf(dr.y); dr.y *= 0.5f;
  dr.z += fabsf(dr.z); dr.z *= 0.5f;

  //Distance squared, no need to do sqrt since opening criteria has been squared
  const float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

  return (ds2 <= fabsf(nodeCOM.w));
}

//Minimum distance
__device__ bool split_node_grav_md(
    const float4 nodeCenter,
    const float4 nodeSize,
    const float4 groupCenter,
    const float4 groupSize)
{
  //Compute the distance between the group and the cell
  float3 dr = {fabs(groupCenter.x - nodeCenter.x) - (groupSize.x + nodeSize.x),
    fabs(groupCenter.y - nodeCenter.y) - (groupSize.y + nodeSize.y),
    fabs(groupCenter.z - nodeCenter.z) - (groupSize.z + nodeSize.z)};

  dr.x += fabs(dr.x); dr.x *= 0.5f;
  dr.y += fabs(dr.y); dr.y *= 0.5f;
  dr.z += fabs(dr.z); dr.z *= 0.5f;

  //Distance squared, no need to do sqrt since opening criteria has been squared
  float ds2    = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

  return (ds2 <= fabs(nodeCenter.w));
}

//Springel (2005) / Gadget-2 relative acceleration opening criterion (AMD_PORT_PLAN.md Phase 3):
//open node J if  d_iJ < (G*m_J*b_J^2 / (Delta_acc*|a_i_old|))^(1/4)  for the *nearest* point of
//the group's bounding box, mirroring split_node_grav_impbh's group-vs-node distance geometry so
//the walk stays one shared decision per (node, group) pair. groupMaxAcc is a single conservative
//per-group |a_old| (the max over the group's members, computed host-side in
//octree::computeGroupMaxAccel) rather than a true per-particle value, since the whole group shares
//one walk decision -- this can only make the criterion *more* conservative (opens more, never
//less) for individual members than a true per-particle MAC would, the same trade-off the group's
//AABB itself already makes for the geometric distance test. Assumes G=1 (Bonsai's native unit
//system, pre-Phase-5); revisit when Phase 5 plumbs real physical units.
//
//b_J: Bonsai's tree nodes are NOT cubes -- unlike Gadget-2's strict octree (every node at a given
//level is an exact cube, so `nop->len` is one unambiguous scalar), Bonsai's AABBs
//(compute_propertiesD.cu) are fitted tightly to each node's actual particle extent per axis, so
//there's no single "the length" the same way. Rather than invent an unrelated proxy (an earlier
//version of this function used the AABB half-diagonal, `sqrt(x^2+y^2+z^2)`, which has no
//counterpart in Gadget-2 at all), use the exact quantity Bonsai's OWN property kernel already
//computes for its own BH criterion's `l` (compute_propertiesD.cu: `l = 2*fmaxf(boxSize.x,
//fmaxf(boxSize.y, boxSize.z))`, i.e. twice the longest half-axis -- the natural generalization of
//"cube side length" to a non-cubic box, degenerating to the exact cube side length when the box
//happens to be cubic). That kernel doesn't store `l` separately (only the theta-folded product
//derived from it), but it costs nothing to recompute here from nodeSize (== boxSizeInfo), which is
//already an input. This removes one real, needless source of mismatch from the reference
//implementation; it is not expected to fully explain the calibration gap measured in
//AMD_PORT_PLAN.md Phase 3 §11 on its own (back-of-envelope, half-diagonal vs. 2*max-axis differ by
//well under 2x for a typical box, not the ~10x gap measured) -- the more likely remaining
//contributor is that Bonsai's AABBs are tightly fitted to particles while Gadget-2's octree nodes
//are fixed-size grid cubes regardless of how sparsely populated they are, a structural geometry
//difference rather than a scalar-choice one. Re-measure with the warm-start harness after this
//change before assuming any further recalibration is still needed.
//
//Comparison form matches Gadget-2's own forcetree.c exactly (`mass * len*len > r2*r2 * aold`,
//aold = ErrTolForceAcc * OldAcc): a cross-multiplication, not the algebraically-equivalent
//sqrt(...)-and-divide form the (1/4)-power formula above suggests -- no division, no sqrt, and
//critically no epsilon guard needed for groupMaxAcc == 0 (iter 0 / static ICs, before any
//bodies_acc0 exists): the condition degrades to `mJ*bJ2 > 0`, true for any real node, which is
//*exactly* Gadget-2's own accepted behavior on its first force evaluation too (P[i].OldAcc = 0
//in init.c) -- confirmed by reading forcetree.c, not assumed. Don't "fix" this into a
//near-exact-on-iter-0 special case; it already matches the reference implementation.
//
//2026-08-28 update: added Gadget-2's mandatory second (proximity/safety-box) criterion, found
//missing during a deep-read pass of Gadget-2's source (see the function body below and
//GADGET2_NOTES.md in the gadget/ project root for the full writeup) -- the relative criterion
//above was never meant to be used alone.
// T11 (contract C-B-05): per-axis gap between a group's AABB and a node reference point, with
// the same minimum-image treatment and max(.,0) clamp the Springel MAC already used. Factored out
// so the primary test and the proximity test can use DIFFERENT node reference points, which is what
// Gadget-2 does: the relative test measures to the node's centre of MASS (`nop->u.d.s`,
// forcetree.c:1211-1213) while the mandatory proximity/safety box measures to the node's GEOMETRIC
// CENTRE (`nop->center`, forcetree.c:1288-1298). The port used the COM for both.
//
// This is NOT the group-vs-particle approximation (that is contract C-B-09, inherent to the
// warp-cooperative design): the group AABB stays the target reference in both branches here. Only
// the NODE reference point changes, and the node's geometric centre is already available at the
// call site as `cellPos`.
static __device__ __forceinline__ float3 group_node_gap(
    const float4 groupCenter, const float4 groupSize, float nx, float ny, float nz)
{
#ifdef PERIODIC
  float3 dr = make_float3(fabsf(pm_nearest(groupCenter.x - nx)) - groupSize.x,
                          fabsf(pm_nearest(groupCenter.y - ny)) - groupSize.y,
                          fabsf(pm_nearest(groupCenter.z - nz)) - groupSize.z);
#else
  float3 dr = make_float3(fabsf(groupCenter.x - nx) - groupSize.x,
                          fabsf(groupCenter.y - ny) - groupSize.y,
                          fabsf(groupCenter.z - nz) - groupSize.z);
#endif
  dr.x += fabsf(dr.x); dr.x *= 0.5f;
  dr.y += fabsf(dr.y); dr.y *= 0.5f;
  dr.z += fabsf(dr.z); dr.z *= 0.5f;
  return dr;
}

__constant__ int g_t11_prox_center = 0;

void t11_set_prox_center(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t11_prox_center), &on, sizeof(int));
  fprintf(stderr, "[T11-PROX-CENTER] Springel proximity box referenced to %s\n",
          on ? "node GEOMETRIC CENTRE (Gadget-2)" : "node centre of mass (legacy)");
}

// C-A-01 part (b): which "geometric centre" the proximity box uses -- the node's FIXED octree-cell
// centre (Gadget-2's `nop->center`, forcetree.c:190-206) or the particle-fitted AABB midpoint.
// T11 fixed COM -> AABB midpoint; this fixes AABB midpoint -> cube centre, which is the point
// Gadget-2 actually uses. The two differ by up to half a cell for a clustered node -- the same
// failure the size term had, from the same cause.
__constant__ int g_ca01_cell_center = 0;

void ca01_set_cell_center(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_ca01_cell_center), &on, sizeof(int));
  fprintf(stderr, "[CA01-CELL-CENTRE] proximity box centred on %s\n",
          on ? "the FIXED octree-cell centre (Gadget-2 nop->center)"
             : "the particle-fitted AABB midpoint (legacy)");
}

static __device__ bool split_node_grav_springel(
    const float4 nodeCOM,      //.xyz = center of mass, .w = mass (real, not yet MAC-encoded)
    const float4 nodeCenter,   //.xyz = node AABB midpoint (boxCenterInfo), T11/C-B-05
    const float4 nodeSize,     //.xyz = AABB half-extents
    const float  nodeCellSize, //T26: fixed octree-cell side length for this node (cellSizeInfo)
    const float4 nodeCellCentre, //C-A-01(b): fixed octree-cell centre (cellCenterInfo)
    const float4 groupCenter,
    const float4 groupSize,
    const float  groupMaxAcc,
    const float  errTolForceAcc,
    const float  groupMaxSoftening = 0.0f)  //T27: per-group max Gadget-2 ForceSoftening
{
  // Primary relative test: measured to the node's centre of mass, as Gadget-2 does.
  const float3 dr  = group_node_gap(groupCenter, groupSize, nodeCOM.x, nodeCOM.y, nodeCOM.z);
  const float  ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  // T11/C-B-05 + C-A-01(b): the proximity safety box is measured to the node's GEOMETRIC centre,
  // and that centre is the FIXED octree-cell centre (Gadget-2's `nop->center`), not the
  // particle-fitted AABB midpoint. Three settings, so each step is separately testable:
  //   g_t11_prox_center=0                        -> centre of mass (the original Bonsai behaviour)
  //   g_t11_prox_center=1, g_ca01_cell_center=0  -> AABB midpoint (T11's partial fix)
  //   g_t11_prox_center=1, g_ca01_cell_center=1  -> octree cell centre (Gadget-2)
  //   g_ca01_cell_center=2                     -> cell centre displaced by one whole cell in x.
  //     Not a physics option: it is the positive control for this switch. "Changing the centre
  //     changes nothing" is only informative if a centre change CAN change something, and =2
  //     displaces it far enough that the proximity census must move. Without it, a flag that
  //     never reached the kernel would look exactly like a true null.
  float4 proxCentre = g_ca01_cell_center ? nodeCellCentre : nodeCenter;
  if (g_ca01_cell_center == 2) proxCentre.x += nodeCellSize;
  const float3 drProx = g_t11_prox_center
      ? group_node_gap(groupCenter, groupSize, proxCentre.x, proxCentre.y, proxCentre.z)
      : dr;

  const float mJ  = fabsf(nodeCOM.w);
  // T26 attempted structural fix -- IMPLEMENTED, BUILT, AND TESTED, BUT REGRESSES THE T24 REPRO;
  // NOT ACTIVE. T24 root-caused (by direct bisection, not inference) that the AABB-doubled bJ below
  // (`2*max(nodeSize.x,y,z)`) can shrink pathologically for a tightly-clustered, high-mass node as
  // the Zel'dovich test's caustic forms (t->1), letting the walk wrongly approximate a node for a
  // target in a genuine, confirmed extreme close encounter. T26 attempted the structural fix this
  // function's own header comment predicted back in the original Phase 3 session: replace bJ with
  // the node's FIXED octree-cell side length (`cellSizeInfo`, computed once at tree-build time in
  // compute_propertiesD.cu's compute_scaling as domain_fac*2^(MAXLEVELS-level), decoded from each
  // node's own build-time subdivision depth already stored in node_bodies[idx].x -- the literal
  // structural analogue of Gadget-2's own fixed-size `len`, not a proxy for it). The plumbing (this
  // parameter, cellSizeInfo itself, and the full call-chain threading it down to this function) is
  // real, builds cleanly, and was verified NOT to be the source of any bug via a direct control
  // experiment (see below). But using it as bJ (`const float bJ = nodeCellSize;`) reproduces T24's
  // exact catastrophic blowup on the real N=32768/PMGRID64 Zel'dovich repro (Epot swinging across
  // 15+ orders of magnitude starting ~iter 235, same signature T24 fixed) -- confirmed via a direct
  // A/B, not assumed: reverting ONLY this one line back to the T24 floor formula while leaving every
  // other T26 plumbing change in place (new cellSizeInfo buffer/kernel arg/host set_args call, all
  // still compiled in and executing) reproduces T24's own verified-clean trajectory to 7 significant
  // figures. This isolates the regression to the fixed-cell VALUE itself for at least one real node
  // in the encounter, not to any wiring mistake. A representative sample of the globally-smallest-
  // AABB internal nodes at iter 228-242 (T26-NODE-SCAN-SMALL diagnostic, gpu_iterate.cpp) showed
  // cellSizeInfo consistently LARGER than the AABB-doubled bJ there (~400 vs ~1011, the expected
  // direction, which should be the SAFER direction per this function's own accept/reject logic) --
  // meaning the actual pathological node in the encounter was NOT among that generic sample (its own
  // AABB apparently isn't the single smallest in the tree, only anomalously small relative to how
  // close the encounter's target sits to it), so this fix's exact failure mode -- what cellSizeInfo
  // evaluates to for the SPECIFIC node T24-ACC1-TOPN already knows to implicate, and why that value
  // is unsafe if it's isn't simply too small -- is not yet directly observed. Leading unconfirmed
  // hypothesis for a follow-up session: Bonsai's octree is not guaranteed perfectly balanced/uniform
  // in the way this derivation assumes for every node (particularly for internally-linked/merged
  // nodes formed by link_tree from children created at different levels, vs. leaves created directly
  // by cl_build_nodes's own per-level loop) -- node_bodies[idx].x's level field may not carry the
  // same fixed-cell-size meaning uniformly across the whole tree the way it does for the leaf-level
  // nodes sampled here, an untested assumption this fix's derivation silently depends on. Per this
  // ticket's own explicit fallback instruction ("if the structural fix isn't safely implementable in
  // one session... stop and hand off a precisely-scoped follow-up rather than forcing a partial
  // change"), T24's original, independently-reverified floor remains the ACTIVE formula below; the
  // fixed-cell attempt and its plumbing are kept in place (cellSizeInfo still computed and available
  // every iteration, zero runtime cost when unused here beyond the one extra global memory fetch) for
  // a future session to root-cause precisely why it fails before retrying it as bJ's source.
  // T9 (contract C-B-06): same substitution as the geometric path, on the criterion production
  // actually ships with. T26 tried this and saw a regression -- but T26 ran with the G-convention
  // bug (T4/C-B-02) making the primary relative test nearly inert, so that negative result does not
  // transfer and is being re-tested here. Gated by GADGET_HIP_T9_CELLSIZE_BJ.
  float bJ = g_t9_cellsize_bj ? nodeCellSize
                              : 2.0f * fmaxf(nodeSize.x, fmaxf(nodeSize.y, nodeSize.z));

  // T28 (Reconsider Ticket 24's root cause): the REAL defect behind the T24/T26/T27 blowup,
  // measured directly (GADGET_HIP_T28_TRACE, gpu_iterate.cpp) on the exact iter-235 repro, not
  // inferred. The pathological source node was traced precisely: level=6 (a completely ordinary
  // subdivision depth -- cellSizeInfo for this level is ~507, well ABOVE softening, so the tree is
  // NOT recursing pathologically deep here; the whole "compact node -> excess depth" framing behind
  // T24/T26/T27 does not apply to this node at all). It is a genuine SINGLE-PARTICLE leaf (nchild=1,
  // multipole mass exactly one particle's mass -- not a duplicate-position artifact, not an
  // index-desync: mass/COM/member-count are all mutually consistent). A single point has, by
  // construction, an AABB half-extent of exactly zero in Bonsai's particle-fitted AABB scheme, so
  // bJ==0.0f EXACTLY for every such leaf (810 of ~8700 nodes in this one group's own walk at iter
  // 235 alone -- single-particle leaves are common, unremarkable tree structure, not a pathology).
  // With bJ==0, BOTH of this function's own tests are structurally defeated regardless of target
  // distance: the primary relative-error test (`mJ*bJ2 > ds2^2*aold`) has bJ2==0, so its LHS is 0,
  // never greater than a non-negative RHS unless ds2 is ALSO exactly 0; the proximity safety test
  // below (`dr < 0.6*bJ`) needs `dr < 0`, impossible after the dr.{x,y,z}=max(dr,0) clamp above. So
  // a bJ==0 node can NEVER be split by this function, no matter how close (even ds2==0, i.e. the
  // particle sits inside the target group's own bounding box) the target actually is -- exactly the
  // classic Salmon & Warren (1994) point-mass singularity Gadget-2's own close-encounter safety net
  // exists to prevent. Gadget-2 itself has an explicit rule for this (forcetree.c, paraphrased in
  // GADGET2_NOTES.md): a node with exactly one particle is always force-opened, never approximated,
  // "because sometimes (mass*pos)*(1.0/mass) != pos even in full double precision" -- and the PORT
  // ALREADY HAS this identical rule, verbatim, for its OTHER (geometric, split_node_grav_impbh) MAC
  // path: compute_propertiesD.cu's compute_scaling sets `cellOp = 10e10` (force-open) whenever
  // `nchild == 1`. That rule was simply never carried over to this (Springel/_MAC_SPRINGEL_) MAC
  // path when it was added -- cellOp is a completely different quantity (boxCenterInfo.w) that this
  // function never reads at all. This is the missing mechanism, not a depth/subdivision issue: fix
  // it here, directly, mirroring Gadget-2's own real rule and the port's own pre-existing sibling
  // precedent, at zero cost to every non-degenerate node (this is a structural completeness fix, not
  // a tuned magnitude constant -- it changes the accept/reject decision for ONLY the small, exact
  // set of genuinely zero-extent nodes, never nudging bJ's actual numeric value for anything else).
  if (bJ <= 0.0f)
  {
    t4_tally(&g_t4_openBJzero);
    return true;
  }

#ifdef PERIODIC
  if (!g_t27_disable_box_floor)
    bJ = fmaxf(bJ, 0.35f * g_periodic_boxSize);
#endif
  // Ticket T27: softening-based floor, complementing (default) or replacing (via
  // GADGET_HIP_T27_DISABLE_BOX_FLOOR=1, PERIODIC builds only) Ticket 24's box-size floor above --
  // see t27_set_soft_floor_coef's comment for the mechanism this mirrors and why the coefficient
  // is calibrated empirically rather than copied from Gadget-2's own 1.0e-3 (that threshold
  // targets near-DEGENERATE particle positions in a true per-particle octree; this floor guards a
  // per-GROUP AABB in a different tree/MAC geometry entirely -- same physical motivation, not the
  // same numeric regime).
  if (g_t27_soft_floor_coef > 0.0f)
    bJ = fmaxf(bJ, g_t27_soft_floor_coef * groupMaxSoftening);
  (void)nodeCellSize; // computed and threaded through for future work (see comment above); not used as bJ yet
  const float bJ2 = bJ * bJ;
  const float aold = errTolForceAcc * groupMaxAcc;

  if(mJ * bJ2 > ds2 * ds2 * aold)
  {
    t4_tally(&g_t4_openPrimary);
    return true;
  }

  //Gadget-2's mandatory second criterion (forcetree.c:1286-1298, GADGET2_NOTES.md), found missing
  //here during the 2026-08-28 Gadget-2 deep-read pass: the relative criterion alone can accept a
  //node the target actually falls inside (Salmon & Warren 1994's close-encounter pathology), so
  //Gadget-2 additionally force-opens any node whose particle lies within a box 1.2x the node's
  //size (0.6*len per axis, on ALL three axes) around its center, regardless of what the relative
  //test said. Group-AABB generalization of that per-particle test: `dr.{x,y,z}` above is already
  //the per-axis gap between the node center and the GROUP's own AABB (zero if they overlap on
  //that axis), so reuse it directly -- this reduces to Gadget-2's exact per-particle test when
  //groupSize==0, and is conservative (opens more, never less) for a multi-particle group, the same
  //trade-off the group's AABB already makes for the primary distance test above.
  const float proximity = 0.6f * bJ;
  const bool  proxOpen  = (drProx.x < proximity && drProx.y < proximity && drProx.z < proximity);
  t4_tally(proxOpen ? &g_t4_openProx : &g_t4_reject);
  return proxOpen;
}

#ifdef PMGRID
// Rcut node-level early-rejection (PLAN.md's deferred Phase 4 performance item, LOG.md §31).
// Gated by PMGRID alone, not PERIODIC (LOG.md §34) -- matches add_acc's own erfc suppression
// above, which now applies to non-periodic TreePM too. Correctness never depended on this --
// add_acc's own per-interaction `tabindex >= NTAB` guard already drops any contribution beyond
// the erfc table's range unconditionally -- this exists purely to avoid walking into (or
// evaluating the monopole/quadrupole of) subtrees that are GUARANTEED to contribute nothing,
// instead of doing that work only to have add_acc discard it per-particle.
//
// Uses the node's own full geometric AABB (nodeCenter/nodeSize -- boxCenterInfo/boxSizeInfo, the
// same quantities split_node_grav_md/springel already use for a node's physical extent), not just
// its center of mass: this is the critical property that makes it safe to never descend into a
// rejected node at all. The minimum possible distance from ANY point in the group's own AABB to
// ANY point in the node's AABB is what's compared against Rcut -- since every descendant's own
// AABB is strictly contained within its parent's, if the PARENT's minimum-possible-distance
// already exceeds Rcut, every descendant's does too, with no exceptions to check for.
static __device__ __forceinline__ bool reject_node_rcut(
    const float4 nodeCenter, const float4 nodeSize,
    const float4 groupCenter, const float4 groupSize,
    const float groupRcut2)
{
  // Same minimum-image treatment as split_node_grav_impbh/springel above (periodic only -- an
  // isolated system has no wraparound), but subtracting BOTH half-extents (group's and node's
  // own, matching split_node_grav_md's un-periodic version) since this is a true AABB-vs-AABB
  // gap, not a point-vs-AABB one.
#ifdef PERIODIC
  float3 dr = make_float3(
      fabsf(pm_nearest(groupCenter.x - nodeCenter.x)) - (groupSize.x + nodeSize.x),
      fabsf(pm_nearest(groupCenter.y - nodeCenter.y)) - (groupSize.y + nodeSize.y),
      fabsf(pm_nearest(groupCenter.z - nodeCenter.z)) - (groupSize.z + nodeSize.z)
      );
#else
  float3 dr = make_float3(
      fabsf(groupCenter.x - nodeCenter.x) - (groupSize.x + nodeSize.x),
      fabsf(groupCenter.y - nodeCenter.y) - (groupSize.y + nodeSize.y),
      fabsf(groupCenter.z - nodeCenter.z) - (groupSize.z + nodeSize.z)
      );
#endif
  dr.x = fmaxf(dr.x, 0.0f);
  dr.y = fmaxf(dr.y, 0.0f);
  dr.z = fmaxf(dr.z, 0.0f);
  // T55. This used to be the EUCLIDEAN gap, `dr.x^2+dr.y^2+dr.z^2 > Rcut^2` -- a sphere. Gadget's
  // own test is PER AXIS (forcetree.c:1583-1605): it rejects only when some single axis exceeds
  // `rcut + 0.5*nop->len`, which is a BOX, and a box reaches `sqrt(3)*Rcut` along the diagonal
  // where the sphere stops at Rcut.
  //
  // The difference is not cosmetic, because NEITHER code applies a hard per-particle `r > Rcut`
  // cut: both rely on `tabindex < NTAB`, which cuts at 6*Asmth = (4/3)*Rcut. So the pruning
  // geometry alone decides whether a pair in (Rcut, 6*Asmth) is evaluated, and those pairs are not
  // negligible for sparse heavy particles -- measured at 2.8% of the total force for the low-res
  // species of the zoom-ics set (T54).
  //
  // Measured consequence of the sphere: for ~1% of low-res particles this port DROPPED a
  // short-range pair Gadget keeps (123 cases against 44 where it kept an extra one), landing 10x
  // further from a direct pair sum than Gadget (2.6e-2 of the total force against Gadget's
  // 2.6e-3). The disputed pairs were the 12 FACE DIAGONALS of the low-res lattice at
  // dmean*sqrt(2) = 2.994, which is exactly the region a sphere cut excludes and a box cut keeps.
  //
  // Dropping is not symmetric with keeping, which is what makes this a real defect rather than a
  // different amount of work. The tree supplies w(u)*m/r^2 and the PM grid independently supplies
  // (1-w(u))*m/r^2 for EVERY pair at EVERY distance, so evaluating an extra far pair is free -- the
  // grid already supplied its complement and w(u) makes the term tiny -- while omitting a pair
  // whose w(u) is still finite loses that force outright, because nothing notices the tree's
  // omission. So the prune must err permissive, and the per-axis test is the permissive one.
  //
  // This port's box is additionally inflated by the GROUP's half-extent (the `groupSize` subtracted
  // above), because it prunes for a whole warp-group of targets at once rather than per particle.
  // That makes mode 1 a strict superset of Gadget's own node list, never a subset -- which is the
  // property that matters.
  const float dmax = fmaxf(dr.x, fmaxf(dr.y, dr.z));
  const float ds2  = (g_t55_rcut_mode == 0)
                   ? (dr.x*dr.x + dr.y*dr.y + dr.z*dr.z)   // sphere (pre-T55)
                   : (dmax * dmax);                        // box, i.e. Chebyshev
  // T50 section 2: `groupRcut2` is chosen ONCE PER GROUP by the caller -- Rcut[1]^2 when every one
  // of the group's targets is high-res, otherwise max(Rcut[0],Rcut[1])^2. The old code used the max
  // unconditionally because a group *can* in principle mix coarse- and fine-grid targets, and
  // rejecting at the smaller radius would then drop a node still in range for a coarse target. But
  // since Rcut[1] << Rcut[0] always, "the max" is just Rcut[0], so a pure high-res group got no
  // pruning benefit from the zoom at all -- which is what makes a small PMGRID (where Rcut[0] is
  // large) exhaust the walk's cell list. Pruning a pure group at its own Rcut[1] is exactly what
  // Gadget-2 does for high-res particles: its own walk is limited to Rcut, and the erfc table
  // extends to 6*Asmth > Rcut = 4.5*Asmth, so the truncation residual is Gadget's own choice, not a
  // new approximation introduced here.
  // Mode 2 widens the bound from Rcut to the erfc table's own reach, 6*Asmth. Rcut = 4.5*Asmth, so
  // that is exactly (4/3)*Rcut and (16/9)*Rcut^2 -- no new constant has to be uploaded, and it holds
  // for the fine grid too since Rcut[1]/Asmth[1] is the same ratio. Beyond 6*Asmth the per-particle
  // `tabindex >= NTAB` test discards the interaction anyway, so mode 2 cannot miss anything that
  // could contribute; it only costs a wider frontier.
  const float bound2 = (g_t55_rcut_mode == 2) ? groupRcut2 * (16.0f/9.0f) : groupRcut2;
  return ds2 > bound2;
}
#endif


#define TEXTURES

template<int SHIFT, int BLOCKDIM2, int NI, bool INTCOUNT>
static __device__
uint2 approximate_gravity(
    ACC_VEC_T acc_i[NI],
    const float4 _pos_i[NI],
    const float4 groupPos,
    const float h_i[NI],
    const int type_i[NI],
    const uint2 top_cells,
    int *shmem,
    int *cellList,
    const float4 groupSize,
    float2 dens_i[NI],
    const real4 *body_pos,
    const float *body_forceSoftening,
    const real4 *multipole_data,
    const float4  *boxSizeInfo,
    const float4  *boxCenterInfo,
    const float   groupMaxAcc = 0.0f,
    const float   errTolForceAcc = 0.0f,
    const float   *cellSizeInfo = NULL,  //T26: per-node fixed octree-cell size, parallel to boxSizeInfo
    const float   *nodeSoftInfo = NULL,  //T28/C-A-04: per-node (maxSoft, mixed) summary
    const float   groupMaxSoftening = 0.0f,  //T27: per-group max Gadget-2 ForceSoftening
    const float4  *cellCenterInfo = NULL)  //C-A-01(b): per-node fixed octree-cell centre
{
  const int laneIdx = threadIdx.x & (WARP_SIZE-1);

  /* this helps to unload register pressure */
  float4 pos_i[NI];
#pragma unroll 1
  for (int i = 0; i < NI; i++)
    pos_i[i] = _pos_i[i];

  uint2 interactionCounters = make_uint2(0,0); /* # of approximate and exact force evaluations */

#pragma unroll 1
  for (int i = 0; i < NI; i++)
    dens_i[i] = make_float2(0,0);


  volatile int *tmpList = shmem;

  int approxCellIdx, directPtclIdx;

  int directCounter = 0;
  int approxCounter = 0;

  for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE)
    if (root_cell + laneIdx < top_cells.y)
      cellList[ringAddr<SHIFT>(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;

  int nCells = top_cells.y - top_cells.x;

#ifdef PMGRID
  // T50 section 2: the Rcut^2 this group's node rejection will use, decided ONCE here rather than
  // per cell. A group is "pure" when every target held by every lane of this warp is high-res; then
  // every target uses Asmth[1]/Rcut[1] in add_acc, so the whole group may prune at Rcut[1].
  //
  // The ballot is phrased as "did ANY lane see a non-high-res target", which makes it independent of
  // the wavefront width -- `__popc(ballot) == WARP_SIZE` would silently break on a wave64 target,
  // and FULL_MASK here is 64-bit while WARP_SIZE is 32.
  //
  // Every lane holds a valid target: body_i[0] = body_addr + laneId % nb_i wraps, so short groups
  // duplicate members rather than leaving lanes idle. There is no inactive-lane case to mask off.
#ifdef GADGET_HIP_HIGHRES
  float groupRcut2;
  {
    bool myFine = true;
    for (int k = 0; k < NI; k++)
      myFine = myFine && ((1u << type_i[k]) & g_pm_zoom_mask) != 0u;
    const bool anyCoarse = g_t50_no_pure_prune
                         || (__ballot_sync(FULL_MASK, !myFine) != 0ull);
    // Separates \"mixed\" from \"entirely low-res\": the latter is not a missed pruning opportunity.
    const bool anyFine = (__ballot_sync(FULL_MASK, myFine) != 0ull);
    // mask == 0 (zoom inactive) makes every lane "coarse", so this lands on g_pm_rcut2 via the max
    // below -- g_pm_rcut2_1 is zero-initialised __constant__ memory until the zoom uploads it.
    groupRcut2 = anyCoarse ? fmaxf(g_pm_rcut2, g_pm_rcut2_1) : g_pm_rcut2_1;
    if (laneIdx == 0)
    {
      if (!anyCoarse)   atomicAdd(&g_zoom_pure_groups,   1u);
      else if (anyFine) atomicAdd(&g_zoom_mixed_groups,  1u);
      else              atomicAdd(&g_zoom_coarse_groups, 1u);
    }
  }
#else
  const float groupRcut2 = g_pm_rcut2;
#endif
#endif

  int cellListBlock        = 0;
  int nextLevelCellCounter = 0;

  unsigned int cellListOffset = 0;

  /* process level with n_cells */
#if 1
  while (nCells > 0)
  {
    /* extract cell index from the current level cell list */
    const int cellListIdx = cellListBlock + laneIdx;
    const bool useCell    = cellListIdx < nCells;
    const int cellIdx     = cellList[ringAddr<SHIFT>(cellListOffset + cellListIdx)];
    cellListBlock += min(WARP_SIZE, nCells - cellListBlock);

    /* read from gmem cell's info */
    // Multiply with useCell  to prevent out of bound access, out of bound becomes idx '0'.
    // Previous texture lookup would allow out of bound look up as it would bound it to texture size.
    const float4 cellSize = boxSizeInfo[cellIdx*useCell];
    const float4 cellPos  = boxCenterInfo[cellIdx*useCell];
#if 1
    const float4 cellCOM = multipole_data[useCell*(cellIdx+cellIdx+cellIdx)];

    /* check if cell opening condition is satisfied */
// T31 BISECT: -DGADGET_HIP_T31_BISECT compiles the _MAC_SPRINGEL_ build with this block replaced
// by the geometric one, making the two kernels textually identical while the macro stays defined
// everywhere else. If the 512^3 stall survives that, the cause is outside these lines.
#if defined(_MAC_SPRINGEL_) && !defined(GADGET_HIP_T31_BISECT)
    // Ticket 23: replicate Gadget-2's real two-phase bootstrap (gravtree.c:317-322) -- the very
    // first force evaluation of a run has no OldAcc yet (P[i].OldAcc==0, init.c) and Gadget-2
    // itself uses the plain geometric/ErrTolTheta criterion for that one evaluation only, then
    // permanently switches to the relative (Springel) criterion once OldAcc is populated. Before
    // this fix, this kernel used split_node_grav_springel() unconditionally, every iteration,
    // including the first -- with groupMaxAcc==0.0f (bodies_acc0 is zero-initialized at tree build,
    // build.cpp's ccalloc, and not yet written by any force evaluation), aold=errTolForceAcc*0=0,
    // so the primary test `mJ*bJ2 > ds2*ds2*aold` degraded to `mJ*bJ2 > 0` -- true for any node with
    // positive mass and nonzero size, i.e. it always opened/split every node, silently turning the
    // first force evaluation of every _MAC_SPRINGEL_ run into a near-exact direct summation instead
    // of Gadget-2's real geometric approximation. groupMaxAcc==0.0f is a reliable, precise proxy for
    // "this group has never had a real per-step force recorded yet": bodies_acc0 is only ever
    // overwritten with the just-computed, physically real force (correct_particles(), timestep.cu),
    // and a genuine gravitational |acc| landing on exactly 0.0f post-bootstrap is astronomically
    // unlikely, so this cannot misfire past the true bootstrap step. Found and fixed by Ticket 23
    // (this code path had never been exercised at runtime before -- MAC_SPRINGEL defaults OFF).
    const float4 cellCOM1 = make_float4(cellCOM.x, cellCOM.y, cellCOM.z, cellPos.w);
    // T26: nodeCellSize falls back to the old AABB-doubled proxy (2*max(nodeSize.x,y,z)) whenever
    // cellSizeInfo isn't provided (cellSizeInfo == NULL, e.g. the LET remote-walk kernel, which
    // never actually reaches split_node_grav_springel anyway since groupMaxAcc stays 0 there --
    // see this function's own header default-argument comment) so this stays safe even if that
    // ever changes.
    const float nodeCellSize = cellSizeInfo
      ? cellSizeInfo[cellIdx*useCell]
      : 2.0f*fmaxf(cellSize.x, fmaxf(cellSize.y, cellSize.z));
    // C-A-01(b): same NULL fallback discipline as nodeCellSize above -- without the buffer the
    // proximity box falls back to the AABB midpoint it used before, never to garbage.
    const float4 nodeCellCentre = cellCenterInfo ? cellCenterInfo[cellIdx*useCell] : cellPos;
    // T31 discriminator: force the geometric branch regardless of groupMaxAcc. If the 512^3 stall
    // survives this, the springel MAC's own logic is not being executed and the cause is codegen
    // or data outside this decision; if it disappears, the kernel is reaching the springel branch
    // even though the host-side groupMaxAccInfo is all zeros.
    bool splitCell = (!g_t31_force_geo && groupMaxAcc > 0.0f)
      ? split_node_grav_springel(cellCOM, cellPos, cellSize, nodeCellSize, nodeCellCentre, groupPos, groupSize, groupMaxAcc, errTolForceAcc, groupMaxSoftening)
      : split_node_grav_impbh(cellCOM1, groupPos, groupSize);
#else
    const float4 cellCOM1 = make_float4(cellCOM.x, cellCOM.y, cellCOM.z, cellPos.w);
    bool splitCell = split_node_grav_impbh(cellCOM1, groupPos, groupSize);
#endif

#ifdef UNEQUALSOFTENINGS
    // T28/C-B-15 + C-C-08: Gadget-2's mixed-softening forced descent
    // (forcetree.c:1654-1677, the live PMGRID walk):
    //
    //   maxsofttype = (bitflags >> 2) & 7;
    //   if (maxsofttype == 7) { if (mass > 0) endrun(987); ...skip node... }
    //   else if (h < ForceSoftening[maxsofttype]) {
    //          h = ForceSoftening[maxsofttype];
    //          if (r2 < h*h && ((bitflags >> 5) & 1))  { descend; }   /* node mixes softenings */
    //   }
    //
    // Applied on WHICHEVER branch produced splitCell above -- the shipping build has
    // _MAC_SPRINGEL_ off, so this must not live inside the Springel arm.
    // `nodeSoftInfo` encodes both facts: |v| = node max ForceSoftening, v < 0 = members differ,
    // v == 0 = no contributor (Gadget's maxsofttype == 7).
    if (nodeSoftInfo && !g_t28_disable && useCell && !splitCell)
    {
      const float nsRaw = nodeSoftInfo[cellIdx*useCell];
      if (nsRaw == 0.0f)
      {
        // Gadget calls endrun(987) here when such a node carries mass. A massless node contributes
        // nothing either way, so descending is both safe and the conservative reading.
        if (cellCOM.w > 0.0f) { splitCell = true; atomicAdd(&g_cb15_undefined, 1u); }
      }
      else
      {
        const float nodeMaxSoft = fabsf(nsRaw);
        const bool  nodeMixed   = (nsRaw < 0.0f);
        // h that WOULD be used for this node if it were approximated (this is C-C-08's h).
        const float hUsed = fmaxf(h_i[0], nodeMaxSoft);
        if (nodeMixed && group_node_gap2(cellCOM1, groupPos, groupSize) < hUsed*hUsed)
        {
          splitCell = true;                       // forced descent
          atomicAdd(&g_cb15_forced, 1u);
        }
      }
    }
#endif
#else /*added by egaburov, see compute_propertiesD.cu for matching code */
    bool splitCell = split_node_grav_impbh(cellPos, groupPos, groupSize);
#endif

    /* compute first child, either a cell if node or a particle if leaf */
    const int cellData = __float_as_int(cellSize.w);
    const int firstChild =  cellData & 0x0FFFFFFF;
    const int nChildren  = (cellData & 0xF0000000) >> 28;

    if(cellData == 0xFFFFFFFF)
      splitCell = false;

    /**********************************************/
    /* split cells that satisfy opening condition */
    /**********************************************/

    const bool isNode = cellPos.w > 0.0f;

#ifdef PMGRID
    // Rcut early-rejection (LOG.md §31, extended to non-periodic TreePM in §34): a genuine third
    // outcome alongside split/approx below -- computed once per cell and applied to all three
    // branches, so a rejected cell is walked into exactly zero times (not split, not approx'd,
    // not directAcc'd), not just cheaply skipped.
    // C-B-19 probe: the Rcut rejection must be strictly CONSERVATIVE -- it may only discard a
    // subtree every one of whose particles is beyond Rcut of every group member. If that holds,
    // disabling the rejection must leave the forces BIT-IDENTICAL (it only removes an
    // optimisation, never an interaction). Any difference is a silently dropped in-range
    // interaction. Compile with -DGADGET_HIP_CB19_NO_RCUT_REJECT to run the B side of that A/B.
#ifdef GADGET_HIP_CB19_NO_RCUT_REJECT
    const bool rejectCell = false;
#else
    const bool rejectCell = reject_node_rcut(cellPos, cellSize, groupPos, groupSize, groupRcut2);
#endif
#else
    const bool rejectCell = false;
#endif

    {
      bool splitNode  = isNode && splitCell && !rejectCell && useCell;

      /* use exclusive scan to compute scatter addresses for each of the child cells */
      const int2 childScatter = warpIntExclusiveScan(nChildren & (-splitNode));

      /* make sure we still have available stack space */
      //
      // T49 item A. `cellList` is a RING buffer of CAP = CELL_LIST_MEM_PER_WARP<<SHIFT entries,
      // addressed by ringAddr() = absolute & (CAP-1). Four absolute indices bound the data that is
      // still going to be READ, in increasing order:
      //
      //   cellListOffset + cellListBlock                  oldest live entry -- the first cell of
      //                                                   THIS level no warp iteration has consumed
      //                                                   yet. cellListBlock was advanced at the
      //                                                   top of the loop and this iteration's 32
      //                                                   cells are already in registers, so
      //                                                   everything below this index is dead.
      //   cellListOffset + nCells                         end of this level = start of the next
      //   ... + nextLevelCellCounter                      end of the children ALREADY scattered
      //   ... + childScatter.y                            end of what this iteration will write
      //
      // so the live span that must not be overwritten is
      //
      //   (nCells - cellListBlock) + nextLevelCellCounter + childScatter.y
      //
      // The original guard omitted `nextLevelCellCounter`: zero at the start of a level, the whole
      // next level by its end. Whenever
      //
      //   CAP - nextLevelCellCounter < (old expression) <= CAP
      //
      // it passed while the true span exceeded CAP, and the scatter below wrapped onto the oldest
      // UNREAD cells of the current level, after which the walk read cell indices from the wrong
      // level. Every term is warp-uniform (childScatter.y is the warp total of the exclusive scan),
      // so the early return stays warp-uniform as it was.
      //
      // On the severity, because the first write-up of this overstated it: a REPORTED overflow is
      // harmless (the group is re-walked on the big stack and acc_out is written only after the
      // early return, so the corrupted attempt is discarded), and with branching that follows the
      // tree the omission is self-limiting -- the old expression peaks at the START of a level
      // while the true span peaks at its END, where it equals the next level's width, which the old
      // guard then checks. Host replays of this index algebra find corrupted reads but no silent
      // ones under that rule. Let the per-chunk child count be arbitrary, as a garbage-driven
      // traversal makes it once the first corrupted read lands, and silent corruption is abundant.
      // See tickets/T49 and analysis/ringsim*.cpp in the audit repo.
      const int live    = (nCells - cellListBlock) + nextLevelCellCounter + childScatter.y;
      const int liveOld = (nCells - cellListBlock) + childScatter.y;
      if (laneIdx == 0 && live    > 0) atomicMax(&g_cell_high_water,     (unsigned int) live);
      if (laneIdx == 0 && liveOld > 0) atomicMax(&g_cell_high_water_old, (unsigned int) liveOld);
      if (live > (CELL_LIST_MEM_PER_WARP<<SHIFT))
      {
        // Did the defect ever actually fire, or was the frontier always far enough below CAP that
        // the missing term could not matter? Counts only what the OLD guard would have let through,
        // kept separate from g_walk_bailouts so "the fix changed an outcome" is a measurement.
        if (laneIdx == 0 && liveOld <= (CELL_LIST_MEM_PER_WARP<<SHIFT))
          atomicAdd(&g_cell_guard_window, 1u);
        return make_uint2(0xFFFFFFFF,0xFFFFFFFF);
      }

#if 1
      /* if so populate next level stack in gmem */
      if (splitNode)
      {
        const int scatterIdx = cellListOffset + nCells + nextLevelCellCounter + childScatter.x;
        for (int i = 0; i < nChildren; i++)
          cellList[ringAddr<SHIFT>(scatterIdx + i)] = firstChild + i;
      }
#else  /* use scan operation to accomplish steps above, doesn't bring performance benefit */
      int nChildren  = childScatter.y;
      int nProcessed = 0;
      int2 scanVal   = make_int2(0,0);
      const int offset = cellListOffset + nCells + nextLevelCellCounter;
      while (nChildren > 0)
      {
        tmpList[laneIdx] = 1;
        if (splitNode && (childScatter.x - nProcessed < WARP_SIZE))
        {
          splitNode = false;
          tmpList[childScatter.x - nProcessed] = -1-firstChild;
        }
        scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
        if (laneIdx < nChildren)
          cellList[ringAddr<SHIFT>(offset + nProcessed + laneIdx)] = scanVal.x;
        nChildren  -= WARP_SIZE;
        nProcessed += WARP_SIZE;
      }
#endif
      nextLevelCellCounter += childScatter.y;  /* increment nextLevelCounter by total # of children */
    }

#if 1
    {
      /***********************************/
      /******       APPROX          ******/
      /***********************************/

      /* see which thread's cell can be used for approximate force calculation */
      const bool approxCell    = !splitCell && !rejectCell && useCell;
      const int2 approxScatter = warpBinExclusiveScan(approxCell);

      /* store index of the cell */
      const int scatterIdx = approxCounter + approxScatter.x;
      tmpList[laneIdx] = approxCellIdx;
      if (approxCell && scatterIdx < WARP_SIZE)
        tmpList[scatterIdx] = cellIdx;

      approxCounter += approxScatter.y;

      /* compute approximate forces */
      if (approxCounter >= WARP_SIZE)
      {
        /* evalute cells stored in shmem */
        approxAccT<NI,true>(acc_i, pos_i, dens_i, tmpList[laneIdx], h_i, type_i, multipole_data, nodeSoftInfo);

        approxCounter -= WARP_SIZE;
        const int scatterIdx = approxCounter + approxScatter.x - approxScatter.y;
        if (approxCell && scatterIdx >= 0)
          tmpList[scatterIdx] = cellIdx;
        if (INTCOUNT)
          interactionCounters.x += WARP_SIZE*NI;
      }
      approxCellIdx = tmpList[laneIdx];
    }
#endif

#if 1
    {
      /***********************************/
      /******       DIRECT          ******/
      /***********************************/

      const bool isLeaf = !isNode;
      bool isDirect = splitCell && isLeaf && !rejectCell && useCell;

      const int firstBody =   cellData & BODYMASK;
      const int     nBody = ((cellData & INVBMASK) >> LEAFBIT)+1;

      const int2 childScatter = warpIntExclusiveScan(nBody & (-isDirect));
      int nParticle  = childScatter.y;
      int nProcessed = 0;
      int2 scanVal   = make_int2(0,0);

      /* conduct segmented scan for all leaves that need to be expanded */
      while (nParticle > 0)
      {
        tmpList[laneIdx] = 1;
        if (isDirect && (childScatter.x - nProcessed < WARP_SIZE))
        {
          isDirect = false;
          tmpList[childScatter.x - nProcessed] = -1-firstBody;
        }
        scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
        const int  ptclIdx = scanVal.x;

        if (nParticle >= WARP_SIZE)
        {
          directAccT<NI,true>(acc_i, pos_i, ptclIdx, h_i, type_i, dens_i, body_pos, body_forceSoftening);
          nParticle  -= WARP_SIZE;
          nProcessed += WARP_SIZE;
          if (INTCOUNT)
            interactionCounters.y += WARP_SIZE*NI;
        }
        else
        {
          const int scatterIdx = directCounter + laneIdx;
          tmpList[laneIdx] = directPtclIdx;
          if (scatterIdx < WARP_SIZE)
            tmpList[scatterIdx] = ptclIdx;

          directCounter += nParticle;

          if (directCounter >= WARP_SIZE)
          {
            /* evalute cells stored in shmem */
            directAccT<NI,true>(acc_i, pos_i, tmpList[laneIdx], h_i, type_i, dens_i, body_pos, body_forceSoftening);
            directCounter -= WARP_SIZE;
            const int scatterIdx = directCounter + laneIdx - nParticle;
            if (scatterIdx >= 0)
              tmpList[scatterIdx] = ptclIdx;
            if (INTCOUNT)
              interactionCounters.y += WARP_SIZE*NI;
          }
          directPtclIdx = tmpList[laneIdx];

          nParticle = 0;
        }
      }
    }
#endif

    /* if the current level is processed, schedule the next level */
    if (cellListBlock >= nCells)
    {
      cellListOffset += nCells;
      nCells = nextLevelCellCounter;
      cellListBlock = nextLevelCellCounter = 0;
    }

  }  /* level completed */
#endif

  if (approxCounter > 0)
  {
    approxAccT<NI,false>(acc_i, pos_i, dens_i, laneIdx < approxCounter ? approxCellIdx : -1, h_i, type_i, multipole_data, nodeSoftInfo);
    if (INTCOUNT)
      interactionCounters.x += approxCounter * NI;
    approxCounter = 0;
  }

  if (directCounter > 0)
  {
    directAccT<NI,false>(acc_i, pos_i, laneIdx < directCounter ? directPtclIdx : -1, h_i, type_i, dens_i, body_pos, body_forceSoftening);
    if (INTCOUNT)
      interactionCounters.y += directCounter * NI;
    directCounter = 0;
  }

  return interactionCounters;
}

template<int SHIFT2, int BLOCKDIM2, bool ACCUMULATE>
static __device__
bool treewalk(
    const int bid,
    const float *body_forceSoftening,
    const int   *body_type,
    const uint2 node_begend,
    const int    *active_groups,
    const real4  *group_body_pos,
    const float4  *groupSizeInfo,
    const float4  *groupCenterInfo,
    int *shmem,
    int *lmem,
    float4 *acc_out,
    int2   *interactions,
    int    *ngb_out,
    int    *active_inout,
    float  *body_h,
    float2 *body_dens_out,
    const real4  *body_pos,
    const real4 *multipole_data,
    const float4  *boxSizeInfo,
    const float4  *boxCenterInfo,
    const float   *groupMaxAccInfo = NULL,
    const float   errTolForceAcc = 0.0f,
    const float   *cellSizeInfo = NULL,  //T26: per-node fixed octree-cell size, parallel to boxSizeInfo
    const float   *nodeSoftInfo = NULL,  //T28/C-A-04: per-node (maxSoft, mixed) summary
    const float   *groupMaxSofteningInfo = NULL,  //T27: per-group max Gadget-2 ForceSoftening
    const float4  *cellCenterInfo = NULL)  //C-A-01(b): per-node fixed octree-cell centre
{

  /*********** set necessary thread constants **********/
#ifdef DO_BLOCK_TIMESTEP
  real4 curGroupSize    = groupSizeInfo[active_groups[bid]];
#else
  real4 curGroupSize    = groupSizeInfo[bid];
#endif
  const int   groupData       = __float_as_int(curGroupSize.w);
  const uint body_addr        =   groupData & CRITMASK;
  const uint nb_i             = ((groupData & INVCMASK) >> CRITBIT) + 1;

#ifdef DO_BLOCK_TIMESTEP
  real4 group_pos       = groupCenterInfo[active_groups[bid]];
  const float groupMaxAcc = groupMaxAccInfo ? groupMaxAccInfo[active_groups[bid]] : 0.0f;
  const float groupMaxSoftening = groupMaxSofteningInfo ? groupMaxSofteningInfo[active_groups[bid]] : 0.0f;
#else
  real4 group_pos       = groupCenterInfo[bid];
  const float groupMaxAcc = groupMaxAccInfo ? groupMaxAccInfo[bid] : 0.0f;
  const float groupMaxSoftening = groupMaxSofteningInfo ? groupMaxSofteningInfo[bid] : 0.0f;
#endif


  uint body_i[2];
  const int ni = nb_i <= WARP_SIZE ? 1 : 2;
  body_i[0] = body_addr + laneId%nb_i;
  body_i[1] = body_addr + WARP_SIZE + laneId%(nb_i - WARP_SIZE);

  float4 pos_i[2];
  ACC_VEC_T acc_i[2];
  float2 dens_i[2];

  pos_i[0]   = group_body_pos[body_i[0]];
#if BONSAI_DENSITY
  pos_i[0].w = 1.0f/body_h[body_i[0]];
  pos_i[0].w *= pos_i[0].w;  /* .w stores 1/h^2 to speed up computations */
#else
  pos_i[0].w = 0.0f;   // .w is the density kernel's 1/h^2 and nothing else; see node_specs.h
#endif
  if(ni > 1){       //Only read if we actually have ni == 2
    pos_i[1]   = group_body_pos[body_i[1]];
#if BONSAI_DENSITY
    pos_i[1].w = 1.0f/body_h[body_i[1]];
    pos_i[1].w *= pos_i[1].w;  /* .w stores 1/h^2 to speed up computations */
#else
    pos_i[1].w = 0.0f;
#endif
  }

  // Ticket 20: zero-init written component-wise (not via make_float4()) so this compiles
  // identically whether ACC_VEC_T is float4 (original) or double4 (this ticket's test macro).
  acc_i[0].x = acc_i[0].y = acc_i[0].z = acc_i[0].w = 0;
  acc_i[1] = acc_i[0];
  dens_i[0] = dens_i[1] = make_float2(0.0f, 0.0f);

  // Phase 5 ticket 03 (PLAN.md): target's own resolved Gadget-2 ForceSoftening -- unrelated to
  // pos_i[k].w above (Bonsai's own 1/h^2 density/neighbour-search field, a different mechanism).
  float h_i[2];
  h_i[0] = body_forceSoftening[body_i[0]];
  if (ni > 1) h_i[1] = body_forceSoftening[body_i[1]];

  // Phase 5 ticket 07 (PLAN.md): target's own Gadget-2 particle type, used only to pick which
  // (Rcut,Asmth) pair add_acc()'s PMGRID short-range suppression uses (inert when
  // GADGET_HIP_HIGHRES isn't compiled in, or when the runtime zoom mask is 0).
  int type_i[2];
  type_i[0] = body_type[body_i[0]];
  if (ni > 1) type_i[1] = body_type[body_i[1]];

// Interaction counting is diagnostic only: it feeds one log line and costs an int2 per particle
// (1 GB at 512^3). Off by default. Flip this #if to 0 to restore it -- and also re-enable the
// buffer in build.cpp and the host reporting in gpu_iterate.cpp, all three are keyed to this.
#if 1
  const bool INTCOUNT = false;
#else
  const bool INTCOUNT = true;
#endif
  uint2 counters = make_uint2(0,0);
  {
    if (ni == 1)
      counters = approximate_gravity<SHIFT2, BLOCKDIM2, 1,INTCOUNT>(
          acc_i,
          pos_i,
          group_pos,
          h_i,
          type_i,
          node_begend,
          shmem,
          lmem,
          curGroupSize,
          dens_i,
          body_pos,
          body_forceSoftening,
          multipole_data,
          boxSizeInfo,
          boxCenterInfo,
          groupMaxAcc,
          errTolForceAcc,
          cellSizeInfo,
          nodeSoftInfo,
          groupMaxSoftening);
    else
      counters = approximate_gravity<SHIFT2, BLOCKDIM2, 2,INTCOUNT>(
          acc_i,
          pos_i,
          group_pos,
          h_i,
          type_i,
          node_begend,
          shmem,
          lmem,
          curGroupSize,
          dens_i,
          body_pos,
          body_forceSoftening,
          multipole_data,
          boxSizeInfo,
          boxCenterInfo,
          groupMaxAcc,
          errTolForceAcc,
          cellSizeInfo,
          nodeSoftInfo,
          groupMaxSoftening);
  }
  if(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF)
    return false;

#if 0
  /* CUDA 8RC work around */
  if(bid < 0) // bid ==0 && laneId < nb_i && && threadIdx.x == 0)
  {
	  printf("TEST\n");
  	//printf("ON DEV [%d %d : %d %d ] ACC: %f %f %f %f INT: %d %d \n",
  	//		bid, threadIdx.x, nb_i, body_i[0],
  	//		acc_i[0].x,acc_i[0].y,acc_i[0].z,acc_i[0].w,
  	//		counters.x, counters.y);
  }
#endif

  if (laneId < nb_i)
  {
    const int addr = body_i[0];
#if BONSAI_DENSITY
    {
      const float hinv = 1.0f/body_h[addr];
      const float C   = 3465.0f/(512.0f*M_PI)*hinv*hinv*hinv;
      dens_i[0].x *= C;  /* scale rho */
    }
#endif
    if (ACCUMULATE)
    {
      // Ticket 20: explicit (float) narrowing casts -- acc_i[0] may be double4 under this
      // ticket's test macro; a no-op cast when it's still float4 (original behavior). acc_out
      // (bodies_acc1) itself stays float4 unconditionally either way, per the ticket's method.
      acc_out     [addr].x += (float)acc_i[0].x;
      acc_out     [addr].y += (float)acc_i[0].y;
      acc_out     [addr].z += (float)acc_i[0].z;
      acc_out     [addr].w += (float)acc_i[0].w;


#if BONSAI_DENSITY
      body_dens_out[addr].x += dens_i[0].x;
      body_dens_out[addr].y += dens_i[0].y;
#endif
    }
    else
    {
      acc_out      [addr] =  make_float4((float)acc_i[0].x, (float)acc_i[0].y, (float)acc_i[0].z, (float)acc_i[0].w);
#if BONSAI_DENSITY
      body_dens_out[addr] = dens_i[0];
#endif
    }
    //       ngb_out     [addr] = ngb_i;
    // ngb_out removed: write-only (stored each particle's own index), never read.
    active_inout[addr] = 1;
    if (ACCUMULATE)
    {
      if (INTCOUNT) interactions[addr].x += counters.x / ni;
      if (INTCOUNT) interactions[addr].y += counters.y / ni ;
    }
    else
    {
      if (INTCOUNT) interactions[addr].x = counters.x / ni;
      if (INTCOUNT) interactions[addr].y = counters.y / ni ;
    }
    if (ni == 2)
    {
      const int addr = body_i[1];
#if BONSAI_DENSITY
      {
        const float hinv = 1.0f/body_h[addr];
        const float C   = 3465.0f/(512.0f*M_PI)*hinv*hinv*hinv;
        dens_i[1].x *= C;  /* scale rho */
      }
#endif
      if (ACCUMULATE)
      {
        acc_out     [addr].x += (float)acc_i[1].x;
        acc_out     [addr].y += (float)acc_i[1].y;
        acc_out     [addr].z += (float)acc_i[1].z;
        acc_out     [addr].w += (float)acc_i[1].w;

#if BONSAI_DENSITY
        body_dens_out[addr].x += dens_i[1].x;
      	body_dens_out[addr].y += dens_i[1].y;
#endif
      }
      else
      {
        acc_out      [addr] =  make_float4((float)acc_i[1].x, (float)acc_i[1].y, (float)acc_i[1].z, (float)acc_i[1].w);
#if BONSAI_DENSITY
        body_dens_out[addr] = dens_i[1];
#endif

//	body_h[addr] = adjustH(body_h[addr], dens_i[1].y);
      }

      //         ngb_out     [addr] = ngb_i;
      // ngb_out removed: write-only (stored each particle's own index), never read.
      active_inout[addr] = 1;
      if (ACCUMULATE)
      {
        if (INTCOUNT) interactions[addr].x += counters.x / ni;
        if (INTCOUNT) interactions[addr].y += counters.y / ni;
      }
      else
      {
        if (INTCOUNT) interactions[addr].x = counters.x / ni;
        if (INTCOUNT) interactions[addr].y = counters.y / ni;
      }
    }
  }

  return true;
}

template<bool ACCUMULATE, int BLOCKDIM2>
static __device__
void approximate_gravity_main(
    const int n_active_groups,
    int    n_bodies,
    float *body_forceSoftening,
    int   *body_type,
    uint2 node_begend,
    int    *active_groups,
    real4  *body_pos,
    real4  *multipole_data,
    float4 *acc_out,
    real4  *group_body_pos,           //This can be different from body_pos
    int    *ngb_out,
    int    *active_inout,
    int2   *interactions,
    float4  *boxSizeInfo,
    float4  *groupSizeInfo,
    float4  *boxCenterInfo,
    float4  *groupCenterInfo,
    real4   *body_vel,
    int     *MEM_BUF,
    float   *body_h,
    float2  *body_dens,
    const float *groupMaxAccInfo = NULL,
    const float errTolForceAcc = 0.0f,
    const float *cellSizeInfo = NULL,  //T26: per-node fixed octree-cell size, parallel to boxSizeInfo
    const float   *nodeSoftInfo = NULL,  //T28/C-A-04: per-node (maxSoft, mixed) summary
    const float *groupMaxSofteningInfo = NULL,  //T27: per-group max Gadget-2 ForceSoftening
    const float4 *cellCenterInfo = NULL)  //C-A-01(b): per-node fixed octree-cell centre
{
  const int blockDim2 = BLOCKDIM2;
  const int shMemSize = 1 * (1 << blockDim2);
  __shared__ int shmem_pool[shMemSize];

  const int nWarps2 = blockDim2 - WARP_SIZE2;

  const int sh_offs = (shMemSize >> nWarps2) * warpId;
  int *shmem = shmem_pool + sh_offs;
  volatile int *shmemv = shmem;


#if 0
#define SHMODE
#endif

#ifdef SHMODE
  const int nWarps  = 1<<nWarps2;
  const int MAXFAILED = 64;
  __shared__ int failedList[MAXFAILED];
  __shared__ unsigned int failed;

  if (threadIdx.x == 0)
    failed = 0;
#endif

  __syncthreads();

  /*********** check if this block is linked to a leaf **********/

  int  bid  = gridDim.x * blockIdx.y + blockIdx.x;

  while(true)
  {
    if(laneId == 0)
    {
      bid         = atomicAdd(&active_inout[n_bodies], 1);
      shmemv[0]    = bid;
    }

    bid   = shmemv[0];

    if (bid >= n_active_groups) return;

    int *lmem = &MEM_BUF[(CELL_LIST_MEM_PER_WARP<<nWarps2)*blockIdx.x + CELL_LIST_MEM_PER_WARP*warpId];
    const bool success = treewalk<0,blockDim2,ACCUMULATE>(
        bid,
        body_forceSoftening,
        body_type,
        node_begend,
        active_groups,
        group_body_pos,
        groupSizeInfo,
        groupCenterInfo,
        shmem,
        lmem,
        acc_out,
        interactions,
        ngb_out,
        active_inout,
        body_h,
        body_dens,
        body_pos,
        multipole_data,
        boxSizeInfo,
        boxCenterInfo,
        groupMaxAccInfo,
        errTolForceAcc,
        cellSizeInfo,
        nodeSoftInfo,
        groupMaxSofteningInfo,
        cellCenterInfo);

    // See g_walk_bailouts: a false here means this group's walk ran out of cell-list stack and its
    // particles received no force. One atomic, only on the failure path.
    if (!success && laneId == 0) atomicAdd(&g_walk_bailouts, 1u);

#if 0
    if (bid % 10 == 0)
      success = false;
#endif

#ifdef SHMODE
    if (!success)
      if (laneId == 0)
        failedList[atomicAdd(&failed,1)] = bid;

    if (failed + nWarps >= MAXFAILED)
    {
      __syncthreads();
      if (warpId == 0)
      {
        int *lmem1 = &MEM_BUF[(CELL_LIST_MEM_PER_WARP<<nWarps2)*blockIdx.x];
        const int n = failed;
        failed = 0;
        for (int it = 0; it < n; it++)
        {
          const bool success = treewalk<nWarp2,blockDim2,ACCUMULATE>(
              failedList[it],
              eps2,
              node_begend,
              active_groups,
              group_body_pos,
              groupSizeInfo,
              groupCenterInfo,
              shmem,
              lmem1,
              acc_out,
              interactions,
              ngb_out,
              active_inout);
          assert(success);
        }
      }
      __syncthreads();
    }

#else

    //Try to get access to the big stack, only one block per time is allowed
    if (!success)
    {
      if(laneId == 0)
      {
        int res = atomicExch(&active_inout[n_bodies+1], 1); //If the old value (res) is 0 we can go otherwise sleep
        int waitCounter  = 0;
        while(res != 0)
        {
          //Sleep
          for(int i=0; i < (1024); i++)
            waitCounter += 1;

          //Test again
          shmem[0] = waitCounter;
          res = atomicExch(&active_inout[n_bodies+1], 1);
        }
      }

      if (laneId == 0) atomicAdd(&g_walk_retries, 1u);
      int *lmem1 = &MEM_BUF[gridDim.x*(CELL_LIST_MEM_PER_WARP<<nWarps2)];
      const bool bigOk = treewalk<8,blockDim2,ACCUMULATE>(
          bid,
          body_forceSoftening,
          body_type,
          node_begend,
          active_groups,
          group_body_pos,
          groupSizeInfo,
          groupCenterInfo,
          shmem,
          lmem1,
          acc_out,
          interactions,
          ngb_out,
          active_inout,
          body_h,
          body_dens,
          body_pos,
          multipole_data,
          boxSizeInfo,
          boxCenterInfo,
          groupMaxAccInfo,
          errTolForceAcc,
          cellSizeInfo,
          nodeSoftInfo,
          groupMaxSofteningInfo,
        cellCenterInfo);
      // assert() is compiled out by NDEBUG in a Release build, so a failed retry used to be
      // completely silent. Count it instead: this, not g_walk_bailouts, is the condition under
      // which a particle really did go without a force.
      if (!bigOk && laneId == 0) atomicAdd(&g_walk_retry_fail, 1u);

      if(laneId == 0)
        atomicExch(&active_inout[n_bodies+1], 0); //Release the lock
    }
#endif /* SHMODE */
  }     //end while
#undef SHMODE
}


  extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_approximate_gravity(
      const int n_active_groups,
      int    n_bodies,
      float *body_forceSoftening,
      int    *body_type,
      uint2 node_begend,
      int    *active_groups,
      real4  *body_pos,
      real4  *multipole_data,
      float4 *acc_out,
      real4  *group_body_pos,           //This can be different from body_pos
      int    *ngb_out,
      int    *active_inout,
      int2   *interactions,
      float4  *boxSizeInfo,
      float4  *groupSizeInfo,
      float4  *boxCenterInfo,
      float4  *groupCenterInfo,
      real4   *body_vel,
      int     *MEM_BUF,
      float   *body_h,
      float2  *body_dens,
      const float *groupMaxAccInfo,
      const float errTolForceAcc,
      const float *cellSizeInfo,  //T26: per-node fixed octree-cell size, parallel to boxSizeInfo
      const float *nodeSoftInfo,  //T28/C-A-04: per-node (maxSoft, mixed) summary
      const float *groupMaxSofteningInfo,  //T27: per-group max Gadget-2 ForceSoftening
      const float4 *cellCenterInfo)  //C-A-01(b): per-node fixed octree-cell centre
{
  approximate_gravity_main<false, NTHREAD2>(
      n_active_groups,
      n_bodies,
      body_forceSoftening,
      body_type,
      node_begend,
      active_groups,
      body_pos,
      multipole_data,
      acc_out,
      group_body_pos,           //This can be different from body_pos
      ngb_out,
      active_inout,
      interactions,
      boxSizeInfo,
      groupSizeInfo,
      boxCenterInfo,
      groupCenterInfo,
      body_vel,
      MEM_BUF,
      body_h,
      body_dens,
      groupMaxAccInfo,
      errTolForceAcc,
      cellSizeInfo,
      nodeSoftInfo,
      groupMaxSofteningInfo,
      cellCenterInfo);
}


  extern "C"
__launch_bounds__(NTHREAD,1024/NTHREAD)
  __global__ void
  dev_approximate_gravity_let(
      const int n_active_groups,
      int    n_bodies,
      float *body_forceSoftening,
      int    *body_type,
      uint2 node_begend,
      int    *active_groups,
      real4  *body_pos,
      real4  *multipole_data,
      float4 *acc_out,
      real4  *group_body_pos,           //This can be different from body_pos
      int    *ngb_out,
      int    *active_inout,
      int2   *interactions,
      float4  *boxSizeInfo,
      float4  *groupSizeInfo,
      float4  *boxCenterInfo,
      float4  *groupCenterInfo,
      real4   *body_vel,
      int     *MEM_BUF,
      float   *body_h,
      float2  *body_dens)
{
  approximate_gravity_main<true, NTHREAD2>(
      n_active_groups,
      n_bodies,
      body_forceSoftening,
      body_type,
      node_begend,
      active_groups,
      body_pos,
      multipole_data,
      acc_out,
      group_body_pos,           //This can be different from body_pos
      ngb_out,
      active_inout,
      interactions,
      boxSizeInfo,
      groupSizeInfo,
      boxCenterInfo,
      groupCenterInfo,
      body_vel,
      MEM_BUF,
      body_h,
      body_dens);
}


#ifdef PMGRID
// Phase 4 (PLAN.md): isolated correctness test for the tree-side short-range term above --
// exercises the EXACT same add_acc() (minimum-image wrap when PERIODIC, erfc shortrange_table
// suppression always -- LOG.md §34) the real tree walk uses, for a single target-at-origin/source
// pair per thread, with no tree build / domain decomposition involved at all (sidesteps the known
// small-N domain-decomposition divide-by-zero crash noted elsewhere in this project for N<~500).
// Combined with the PM solver's own force on the same pair (pm_compute_forces_periodic or
// pm_compute_forces_isolated), this is the actual "does short-range + long-range sum to the
// correct total force" check -- validated externally against an Ewald-summation ground truth for
// the periodic case (include/ewald_ref.h) or plain Newtonian G*m/r^2 for the isolated case,
// independent of both the tree and PM code either way.
// `type` (per-sample TARGET type) is threaded through so Ticket 07's own zoom-vs-coarse dual-grid
// test (--zoom-treepm-force-test) can reuse this exact "no tree build" harness with a high-res
// type; every OTHER existing caller passes type=0, for which add_acc()'s selection is unconditionally
// grid 0 regardless of whether GADGET_HIP_HIGHRES's zoom mask is even active (mask=0 by default,
// and no other diagnostic in this file ever calls pm_zoom_upload_rcut_asmth).
__global__ void dev_pm_test_add_acc_pair(const float3 *sep, const float *mass, const int *type,
                                          int n, float4 *outAcc)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float4 pos0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float2 density = make_float2(0.0f, 0.0f);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  acc = add_acc(acc, pos0, mass[i], sep[i], 0.0f, type[i], density);
  outAcc[i] = acc;
}

void pm_test_add_acc_pair(const float3 *d_sep, const float *d_mass, const int *d_type, int n,
                           float4 *d_outAcc, hipStream_t stream)
{
  const int block = 128;
  const int grid  = (n + block - 1) / block;
  hipLaunchKernelGGL(dev_pm_test_add_acc_pair, dim3(grid), dim3(block), 0, stream, d_sep, d_mass,
                      d_type, n, d_outAcc);
}
#endif

// Phase 5 ticket 03 (PLAN.md): pure add_acc() unit test for the spline-softening formula and the
// max(target,source)-ForceSoftening combination rule -- see pm.h's doc comment for the full
// rationale. Deliberately NOT gated by PMGRID (unlike dev_pm_test_add_acc_pair above), so it
// reads add_acc() in isolation from the PM short-range erfc suppression.
__global__ void dev_softening_test_pair(const float3 *sep, const float *mass,
                                         const float *h_i, const float *h_j,
                                         int n, float4 *outAcc)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float4 pos0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float2 density = make_float2(0.0f, 0.0f);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const float h = fmaxf(h_i[i], h_j[i]); // matches directAcc()'s exact combination rule
  // type=0: irrelevant, PMGRID suppression isn't exercised by this test at all (see its own doc
  // comment above).
  acc = add_acc(acc, pos0, mass[i], sep[i], h, 0, density);
  outAcc[i] = acc;
}

void softening_test_pair(const float3 *d_sep, const float *d_mass, const float *d_h_i,
                          const float *d_h_j, int n, float4 *d_outAcc, hipStream_t stream)
{
  const int block = 128;
  const int grid  = (n + block - 1) / block;
  hipLaunchKernelGGL(dev_softening_test_pair, dim3(grid), dim3(block), 0, stream,
                      d_sep, d_mass, d_h_i, d_h_j, n, d_outAcc);
}
