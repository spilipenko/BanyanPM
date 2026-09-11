#include "hip/hip_runtime.h"
#include "bonsai.h"
#include <vector>
#ifdef __DEVICE_EMULATION__
  #define EMUSYNC __syncthreads();
#else
  #define EMUSYNC
#endif
#include "../profiling/bonsai_timing.h"
PROF_MODULE(timestep);

#include "node_specs.h"
#include "gadget_driftfac.h"


//Reduce function to get the minimum timestep
static __device__ __forceinline__ void get_TnextD(const int n_bodies,
                                     float2 *time,
                                     float *tnext, volatile float *sdata) {
  //float2 time : x is time begin, y is time end

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const int blockSize   = blockDim.x;
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 1.0e10f;
  float tmin = 1.0e10f;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n_bodies) {
    if (i             < n_bodies) tmin = fminf(tmin, time[i            ].y);
    if (i + blockSize < n_bodies) tmin = fminf(tmin, time[i + blockSize].y);

    i += gridSize;
  }

  sdata[tid] = tmin;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] = tmin = fminf(tmin, sdata[tid + 256]); } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] = tmin = fminf(tmin, sdata[tid + 128]); } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] = tmin = fminf(tmin, sdata[tid +  64]); } __syncthreads(); }
  // Unsynchronised warp-reduction tail replaced (see gpu_boundaryReduction in build_tree.cu for
  // the instance where this was caught failing). EMUSYNC expands to NOTHING here, so these were
  // dependent shared-memory steps with no barrier, relying on lockstep lanes. Synchronised loop
  // instead; __syncthreads() is reached by every thread in the block.
  for (unsigned int s = 32; s > 0; s >>= 1)
  {
    if (blockSize >= 2 * s && tid < s) { sdata[tid] = tmin = fminf(tmin, sdata[tid + s]); }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) tnext[blockIdx.x] = sdata[0];
}

KERNEL_DECLARE(get_Tnext)(const int n_bodies,
                                     float2 *time,
                                     float *tnext) {
  extern __shared__ float sdata[];
  get_TnextD(n_bodies, time, tnext, sdata);
}


//Reduce function to get the number of active particles
static __device__ void get_nactiveD(const int n_bodies,
                                       uint *valid,
                                       uint *tnact, volatile int *sdataInt) {
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const int blockSize   = blockDim.x;
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdataInt[tid] = 0;
  int sum       = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n_bodies) {
    if (i             < n_bodies) sum = sum + valid[i            ];
    if (i + blockSize < n_bodies) sum = sum + valid[i + blockSize];

    i += gridSize;
  }
  sdataInt[tid] = sum;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { sdataInt[tid] = sum = sum + sdataInt[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdataInt[tid] = sum = sum + sdataInt[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdataInt[tid] = sum = sum + sdataInt[tid +  64]; } __syncthreads(); }


  // Unsynchronised warp-reduction tail replaced (see gpu_boundaryReduction in build_tree.cu for
  // the instance where this was caught failing). EMUSYNC expands to NOTHING here, so these were
  // dependent shared-memory steps with no barrier, relying on lockstep lanes. Synchronised loop
  // instead; __syncthreads() is reached by every thread in the block.
  for (unsigned int s = 32; s > 0; s >>= 1)
  {
    if (blockSize >= 2 * s && tid < s) { sdataInt[tid] = sum = sum + sdataInt[tid + s]; }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) tnact[blockIdx.x] = sdataInt[0];
}

//Reduce function to get the number of active particles
KERNEL_DECLARE(get_nactive)(const int n_bodies,
                                       uint *valid,
                                       uint *tnact) {
  extern __shared__ int sdataInt[];
  get_nactiveD(n_bodies, valid, tnact, sdataInt);
}

// Phase 5 ticket 05 (PLAN.md): driftfac.c lookup, device-side counterpart of
// gadget_driftfac.cpp's host lookupTable() -- same formula, kept as a separate implementation
// (not shared code) since one runs on the device and one on the host, but validated against each
// other by --driftfac-test.
// T29 / C-D-06: `loga` and `u` are computed in DOUBLE; the table stays float.
//
// The table coordinate `u` runs 0..1000 over the whole run, so one float32 ULP of `u` near the top
// end is 6.1e-05 cells = 2.54e-07 dloga -- which is 0.98 of ONE tick of this port's own 2^24
// timeline. In float the lookup therefore cannot resolve a single tick, while the finest timestep
// actually used (span/2^15) is only 0.03 cells wide, i.e. ~500 ULPs of `u`. Gadget-2 builds `u` in
// double from an integer tick and over-resolves its own finer 2^28 tick by seven orders.
//
// Fixing the ARITHMETIC is what matters, not the storage. Measured relative error against a double
// reference at the finest bin: 8.3e-03 as shipped, 4.0e-03 with a double TABLE and float math, and
// **1.9e-04 with a float table and double math** -- 44x, versus 2x for the storage fix the contract
// originally proposed. Storage contributes little because `table[i] - table[i-1]` subtracts two
// floats within a factor of two of each other and is therefore EXACT by Sterbenz; the storage error
// enters only through the slope.
//
// The same reasoning, and the same mistake, appear in `snapToTimelineTick` above: double arithmetic
// is worthless if its INPUT was already rounded to float (see tickets/T25's CORRECTION and T29).
static __device__ __forceinline__ float lookupDriftTableD(const float *table, double logTimeBegin,
                                                            double logTimeMax, double time)
{
  const double loga = log(time);
  const double u = (loga - logTimeBegin) / (logTimeMax - logTimeBegin) * GADGET_DRIFT_TABLE_LENGTH;
  int i = (int) u;
  if (i >= GADGET_DRIFT_TABLE_LENGTH) i = GADGET_DRIFT_TABLE_LENGTH - 1;
  if (i <= 1) return (float) (u * (double) table[0]);
  return (float) ((double) table[i - 1] + ((double) table[i] - (double) table[i - 1]) * (u - i));
}

// ---------------------------------------------------------------------------------------------
// T25 / C-D-05: the global integer timeline.
//
// Gadget-2 stores each particle's step end as an INTEGER count of TIMEBASE ticks of
// Timebase_interval (allvars.h:25, init.c:51), so two particles that should re-synchronise at the
// same moment hold bit-identical values and a power-of-two bin boundary is exact. This port stores
// physical times and advanced them multiplicatively, where
//     tc*expf(d)*expf(d)  !=  tc*expf(2d)
// so a fine-bin particle taking two half-steps and a coarse-bin particle taking one full step miss
// each other. Measured at 512^3 (tickets/T24): two sync points exactly 1.00 float32 ULP apart, and
// a whole extra system step -- tree walk, PM solve, kick -- burned to advance the clock by 9.3e-10
// against a typical step of 9.6e-5.
//
// Fix: every end-of-step time is snapped onto the global tick grid, so it is a function of an
// INTEGER tick alone. Equal intended ticks then give bit-identical floats, which is the property
// Gadget gets for free from integer storage.
//
// The arithmetic is done in DOUBLE deliberately. In float32 the log of a ~ 0.01 carries an
// absolute error of ~|log a|*1e-7 ~ 4.6e-7, comparable to a tick itself, so the rounding could not
// identify the tick; in double the error is ~1e-15 against a tick of ~2.7e-7, i.e. unambiguous by
// eight orders of magnitude. That headroom is what lets the snap be idempotent -- snapping an
// already-snapped value returns it unchanged -- so repeated steps stay exactly on the grid instead
// of drifting off it.
/* 2^24 = 24 bin levels. Capped here (vs Gadget-2's 2^28) because the SNAPPED TIME is stored
 * back as float32: at 2^26 the tick spacing falls below one float32 ULP and adjacent ticks
 * collapse onto the same float. Margin at 2^24 is 2.3-3.7 ULP across a=0.01..1. Lifting this
 * needs integer tick storage, not a bigger constant -- see tickets/T25. */
#define GADGET_TIMEBASE_TICKS 16777216.0

// T38: the margin quoted above is a function of the run's SPAN, and 2^24 only holds it for a FULL
// run. In a-space one tick is span*a/T and one float32 ULP is ~a*2^-24, so the ratio is
// span*2^24/T -- independent of epoch, proportional to span. A full run (TimeBegin=0.01,
// span=4.605) gives the 2.3-3.7 ULP the comment claims. Restart from a late SNAPSHOT and the span
// collapses -- from snapshot 11 at a=0.896, span=0.1096 -- so the same 2^24 puts ~10 ticks inside
// one ULP. The snap can then no longer make two bins' rendezvous land on the same float, and
// C-D-05 fails exactly as T24 described: a small group (1e4-1e5 particles) splits off by 1 ULP and
// the run spends a whole system step -- tree build, PM solve, kick -- advancing the clock by one
// ULP. Measured on a restart from snapshot 11: HALF of all steps were such phantoms, against 1 in
// 600 in the production run.
//
// So scale the tick count with the span, capped at the historical 2^24. The rule deliberately
// keeps 2^24 for every run whose tick is ALREADY at least one ULP wide -- production (span 4.605)
// and the A3 acceptance case (span 2.361) both keep it, and stay bit-identical -- and engages only
// where the invariant is genuinely broken.
//
// This is a mitigation, not the real fix. The real fix is storing tick INDICES as integers instead
// of round-tripping them through a float32 time every step (tickets/T25); this only keeps the
// float32 representation adequate for the span in use.
__host__ __device__ __forceinline__ double gadget_timebase_ticks(double span)
{
  double t = GADGET_TIMEBASE_TICKS;
  if (span > 0.0)
  {
    // Largest power of two <= span * 2^23, i.e. keep one tick at >= ~1 float32 ULP.
    const double lim = exp2(floor(log2(span * 8388608.0)));
    if (lim < t) t = lim;
    if (t < 1024.0) t = 1024.0;               // never collapse the ladder to fewer than 2^10 bins
  }
  return t;
}

// T25/C-D-05 fix: `tNew` is a DOUBLE. It must be, and the earlier float signature is why the
// original snap only half-worked. One tick is only 2.3-3.7 float32 ULP wide (see the constant's
// comment above), so quantising the step-end time to float32 BEFORE snapping injects up to ~0.9
// tick of error -- expf's own error plus the product's rounding plus the stored tc's quantisation.
// Measured on the shipped float version: 15% of single steps landed on the wrong tick, and the
// rendezvous property this whole mechanism exists to provide (one step of level k must end on the
// same float as two steps of level k+1) FAILED 27.8% of the time. With the product formed in
// double both numbers are exactly 0. The double arithmetic *inside* the snap was never the issue;
// the precision of its INPUT was.
static __device__ __forceinline__ float snapToTimelineTick(
    double tNew, float origin, float span, int comovingFlag)
{
  // `origin`/`span` are in the run's OWN axis units, matching timelineSpan: log(a) under comoving
  // integration, linear t otherwise (Gadget-2's Timebase_interval, init.c:51/56).
  if (!(span > 0.0f)) return (float) tNew;         // ladder disabled -- leave the value untouched
  const double dtick = (double) span / gadget_timebase_ticks((double) span);
  const double x     = comovingFlag ? log(tNew) : tNew;
  const double n     = floor((x - (double) origin) / dtick + 0.5);
  const double snapped = (double) origin + n * dtick;
  return (float) (comovingFlag ? exp(snapped) : snapped);
}
// ---------------------------------------------------------------------------------------------

static __device__ __forceinline__ float driftFactorD(const float *table, double logTimeBegin,
                                                       double logTimeMax, double t0, double t1)
{
  return lookupDriftTableD(table, logTimeBegin, logTimeMax, t1) -
         lookupDriftTableD(table, logTimeBegin, logTimeMax, t0);
}

// T6 (contract C-D-01): select the drift anchor. 0 = legacy per-particle Ti_begstep anchor,
// 1 = Gadget-2's global previous-sync anchor. Env-gated via GADGET_HIP_T6_DRIFT_TP so both can be
// A/B'd from one binary; inert either way when every particle is active every step.
__constant__ int g_t6_drift_from_tp = 0;

void t6_set_drift_anchor(int fromTp)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t6_drift_from_tp), &fromTp, sizeof(int));
  fprintf(stderr, "[T6-DRIFT-ANCHOR] %s\n", fromTp ? "global tp (Gadget-2)" : "per-particle tb (legacy)");
}

// C-D-02: use the particle's own STORED Ti_endstep as the end of the old kick interval, instead of
// the global current time. Gadget-2 (timestep.c:254) computes tstart = (Ti_begstep+Ti_endstep)/2,
// and its activation predicate guarantees Ti_endstep == Ti_Current. The port activates at GROUP
// granularity, so a particle can be kicked before its own scheduled end -- and then the interval
// [mid(tb,tc), mid(tb,te_sched)] is covered by BOTH the previous kick and this one, injecting
// energy. Using the stored end restores exact kick tiling even with group activation.
__constant__ int g_cd02_kick_tend = 0;

// C-D-03: kick and re-bin only particles whose own Ti_endstep equals the current sync time, which
// is Gadget-2's exact predicate (timestep.c:185). Group granularity is a real GPU constraint for
// the tree WALK (a group is the warp's work unit) but not for this per-particle kernel, which is
// already indexed by idx and already testing a flag. A not-due particle takes the pass-through
// branch, so the early dt computed for it is discarded.
__constant__ int g_cd03_strict_active = 0;

// C-D-03b: use oriParticleOrder to read bodies_time in setActiveGroups (see that kernel's comment).
__constant__ int g_cd03b_fix_index = 0;

// T14/energy: apply Gadget-2's kick-time velocity extrapolation in the energy diagnostic.
__constant__ int g_energy_extrap = 0;

// C-D-04: anchor the power-of-2 timestep ladder to the global timeline span, as Gadget-2 does
// (init.c:51/56 -> Timebase_interval = span/TIMEBASE; timestep.c:189-193 halves TIMEBASE down),
// instead of to MaxSizeTimestep. Seeding from MaxSizeTimestep makes every bin a power-of-2
// subdivision of a parameter rather than of the run's own time axis, so the port silently runs
// on a coarser ladder than the parameter file implies.
__constant__ int g_cd04_timeline_ladder = 0;

// Energy diagnostic: Gadget-2 runs a FRESH, all-particle potential pass (compute_potential(),
// potential.c) immediately before each energy statistic (run.c:55), because an inactive particle's
// stored potential is stale -- the field around it has moved since its last active step. The port
// reuses acc.w, which is only written for particles in the current walk's active set, so de is
// only exact at moments when every particle happens to be due (measured: 4 of 2005 samples).
// Setting this makes setActiveGroups mark every group active for one iteration, so the walk
// refreshes acc.w for all particles. Safe by construction: a tree walk's result for particle i does
// not depend on which OTHER particles are active, so due particles get bit-identical forces; and
// with C-D-03 strict activation the kick is gated on Ti_endstep == t_current, not on this flag, so
// no extra particle is kicked or re-binned.
__constant__ int g_force_all_active = 0;

void cd02_set_kick_tend(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_cd02_kick_tend), &on, sizeof(int));
  fprintf(stderr, "[CD02-KICK-TEND] old-kick interval ends at %s\n",
          on ? "the particle's stored Ti_endstep (Gadget-2)" : "the global current time (legacy)");
}

void energy_set_force_all_active(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_force_all_active), &on, sizeof(int));
}

void cd04_set_timeline_ladder(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_cd04_timeline_ladder), &on, sizeof(int));
  fprintf(stderr, "[CD04-LADDER] timestep ladder anchored to %s\n",
          on ? "the global timeline span (Gadget-2)" : "MaxSizeTimestep (legacy)");
}

void energy_set_extrap(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_energy_extrap), &on, sizeof(int));
  fprintf(stderr, "[ENERGY-EXTRAP] kinetic energy uses %s\n",
          on ? "kick-time extrapolated velocities (Gadget-2)" : "raw stored velocities (legacy)");
}

void cd03b_set_fix_index(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_cd03b_fix_index), &on, sizeof(int));
  fprintf(stderr, "[CD03B-ACTIVE-INDEX] setActiveGroups reads bodies_time via %s\n",
          on ? "oriParticleOrder (correct)" : "the raw sorted index (legacy)");
}

void cd03_set_strict_active(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_cd03_strict_active), &on, sizeof(int));
  fprintf(stderr, "[CD03-STRICT-ACTIVE] kick predicate: %s\n",
          on ? "per-particle Ti_endstep == t_current (Gadget-2)" : "group activation flag (legacy)");
}

// T7 (contract C-D-07): count of particles whose committed timestep failed to advance the clock.
// Nonzero means the run is stuck; the host turns it into a clear abort instead of an infinite loop.
__device__ unsigned int g_t7_nonadvancing = 0;
// C-D-14: Gadget-2's endrun(818) prints the offending particle's ID, dt, ti_step, ac and position
// (timestep.c:537-552) -- a bare count cannot be acted on, so capture the first offender's numbers
// too. The thread that gets ticket 0 from the atomic is the only writer, so no further sync needed.
// C-D-14 fault injection (GADGET_HIP_T7_INJECT=1): force particle 0's step to not advance, so the
// abort path can be exercised on purpose. A guard that has never been seen to fire is
// indistinguishable from one that cannot fire -- this session's T26 marker was exactly that case.
// Off by default; one device-side int compare when on.
// C-D-13 instrument: how many times the SYNCHRONIZATION rule actually blocked a growth this step.
// A rule never observed to fire is indistinguishable from one that cannot fire (PLAN.md rule 8),
// so this is reported, not assumed.
// A/B control for C-D-13 (GADGET_HIP_CD13=0 disables the rule). Same binary both ways, so an A/B
// cannot be confounded by a rebuild. Default ON.
__device__ int g_cd13_sync = 1;
__device__ unsigned int g_cd13_blocked = 0;
__device__ unsigned int g_cd13_reached = 0;   // guard passed, rule evaluated
__device__ unsigned int g_cd13_grow    = 0;   // a growth was actually attempted (sNew > sOld)
__device__ int   g_t7_injectStall = 0;
__device__ int   g_t7_firstIdx = -1;
__device__ float g_t7_firstTc  = 0.0f;
__device__ float g_t7_firstEnd = 0.0f;
__device__ float g_t7_firstDt  = 0.0f;
__device__ float g_t7_firstAc  = 0.0f;

void t7_reset_nonadvancing()
{
  unsigned int z = 0;
  int          m = -1;
  float        f = 0.0f;
  hipMemcpyToSymbol(HIP_SYMBOL(g_t7_nonadvancing), &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t7_firstIdx), &m, sizeof(m));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t7_firstTc),  &f, sizeof(f));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t7_firstEnd), &f, sizeof(f));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t7_firstDt),  &f, sizeof(f));
  hipMemcpyToSymbol(HIP_SYMBOL(g_t7_firstAc),  &f, sizeof(f));
}

unsigned int t7_read_nonadvancing()
{
  unsigned int v = 0;
  hipMemcpyFromSymbol(&v, HIP_SYMBOL(g_t7_nonadvancing), sizeof(v));
  return v;
}

void cd13_set_enabled(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_cd13_sync), &on, sizeof(on));
}

void cd13_reset_blocked()
{
  unsigned int z = 0;
  hipMemcpyToSymbol(HIP_SYMBOL(g_cd13_blocked), &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_cd13_reached), &z, sizeof(z));
  hipMemcpyToSymbol(HIP_SYMBOL(g_cd13_grow),    &z, sizeof(z));
}

unsigned int cd13_read_blocked()
{
  unsigned int v = 0;
  hipMemcpyFromSymbol(&v, HIP_SYMBOL(g_cd13_blocked), sizeof(v));
  return v;
}

void cd13_read_detail(unsigned int *reached, unsigned int *grow, unsigned int *blocked)
{
  hipMemcpyFromSymbol(reached, HIP_SYMBOL(g_cd13_reached), sizeof(*reached));
  hipMemcpyFromSymbol(grow,    HIP_SYMBOL(g_cd13_grow),    sizeof(*grow));
  hipMemcpyFromSymbol(blocked, HIP_SYMBOL(g_cd13_blocked), sizeof(*blocked));
}

void t7_set_inject_stall(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t7_injectStall), &on, sizeof(on));
}

void t7_read_first_offender(int *idx, float *tc, float *newEnd, float *dt, float *ac)
{
  hipMemcpyFromSymbol(idx,    HIP_SYMBOL(g_t7_firstIdx), sizeof(*idx));
  hipMemcpyFromSymbol(tc,     HIP_SYMBOL(g_t7_firstTc),  sizeof(*tc));
  hipMemcpyFromSymbol(newEnd, HIP_SYMBOL(g_t7_firstEnd), sizeof(*newEnd));
  hipMemcpyFromSymbol(dt,     HIP_SYMBOL(g_t7_firstDt),  sizeof(*dt));
  hipMemcpyFromSymbol(ac,     HIP_SYMBOL(g_t7_firstAc),  sizeof(*ac));
}

KERNEL_DECLARE(predict_particles)(const int 	n_bodies,
										float 	tc,
										float 	tp,
										real4 	*pos,
										real4 	*vel,
										real4 	*acc,
										float2 	*time,
										real4 	*pPos,
										real4 	*pVel,
										int     comovingIntegrationOn,
										const float *driftTable,
										const float *gravKickTable,
										double  logTimeBegin,   // T29: double
										double  logTimeMax){
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint idx = bid * blockDim.x + tid;


  if (idx >= n_bodies) return;

  float4 p = pos [idx];
  float4 v = vel [idx];
  (void) acc; // no longer used: Gadget-2's DM drift has no acceleration term, see below
  float tb = time[idx].x;

  #ifdef DO_BLOCK_TIMESTEP
    // T6 (contract C-D-01): the drift anchor must be the GLOBAL previous sync time, not this
    // particle's own Ti_begstep. Gadget-2 drifts every particle from All.Ti_Current to min_glob
    // (run.c:227 move_particles(All.Ti_Current, min_glob) -> predict.c:42,53-54) and never uses
    // Ti_begstep as a drift anchor -- Ti_begstep is KICK bookkeeping only (timestep.c:254), which
    // is exactly how correct_particles() below still uses it at :369/:383.
    //
    // Anchoring at tb is only correct if the committed position is NOT advanced between a
    // particle's own sync points. It is: correct_particles writes pos[idx]=pPos[idx] for ACTIVE
    // particles (:375) and, since Ticket 12's aliasing fix, for INACTIVE ones too (:351). So for a
    // particle that stays inactive across consecutive system steps, pos has already been carried
    // forward to tp while time[].x still reads its old tb, and drifting from tb re-adds the whole
    // interval [tb, tp] on top of it -- a full surplus drift of v*D(tb,tp), compounding every
    // skipped step. Latent whenever every particle is active every step (then tb == tp exactly,
    // because the active branch set time[].x = tc on the previous step), which is why every
    // uniform/forced-global-timestep test in this project passes regardless.
    //
    // Deliberately does NOT write time[idx].x (unlike the dead #else branch below, which clobbers
    // it with tp) -- that field is this particle's Ti_begstep and correct_particles needs it
    // intact for the kick midpoint.
    static_assert(true, "");
    const float t0_drift = g_t6_drift_from_tp ? tp : tb;
    float dt_cb  = tc - t0_drift;
    const float t0 = t0_drift, t1 = tc;
  #else
    float dt_cb  = tc - tp;
    time[idx].x  = tp;
    const float t0 = tp, t1 = tc;
  #endif

//   float dt_pb  = tp - tb;

  // Real Gadget-2 time integration is pure kick-drift-kick (KDK), not the predictor-corrector
  // this port used to run: `move_particles()` (predict.c:54) drifts every particle's position by
  // `Pos += Vel*dt_drift` ONLY -- no acceleration term at all -- because for DM particles (no SPH)
  // velocity is never predicted/extrapolated either (VelPred in predict.c:56-65 is behind
  // `P[i].Type==0`, gas-only). All of the force's effect on the trajectory enters through the two
  // half-interval kicks in advance_and_find_timesteps() (timestep.c:254-282), never through this
  // drift step. The previous version of this kernel added a `0.5*a*dt_drift*dt_kick` term to the
  // position update and a `v += a*dt_kick` term here, which was this port's own predictor-corrector
  // (PEC) leapfrog convention, not Gadget-2's -- algebraically equivalent to KDK only when the
  // step size is constant between consecutive steps, which fails exactly during a close encounter
  // (the regime that drives individual timesteps hardest). Replaced with Gadget-2's exact formula;
  // see correct_particles() below for where the real (asymmetric, midpoint-to-midpoint) kick now
  // lives instead.
  float dt_drift = dt_cb;
  if (comovingIntegrationOn)
    dt_drift = driftFactorD(driftTable, logTimeBegin, logTimeMax, t0, t1);

  p.x += v.x*dt_drift;
  p.y += v.y*dt_drift;
  p.z += v.z*dt_drift;

  pPos[idx] = p;
  pVel[idx] = v;
}


// C-D-03b: `time` (bodies_time) is NEVER reordered by sort_bodies()'s default per-iteration path,
// while `body2grouplist` is rebuilt by store_group_list AFTER the sort and is therefore in the NEW
// sorted order. Indexing both with the same raw `idx` pairs a particle's group with a DIFFERENT
// particle's end-of-step time, so the wrong groups are activated: genuinely due particles are
// missed and unrelated ones are woken. Because a missed particle is never re-binned, it keeps the
// global minimum Ti_endstep, t_current is pinned, and the driver spins. This is the same
// index-space hazard as the old Tickets 12/23/27 -- the fix is the same oriParticleOrder
// translation correct_particles() already applies to this exact buffer.
KERNEL_DECLARE(setActiveGroups)(const int n_bodies,
                                            float tc,
                                            float2 *time,
                                            uint  *body2grouplist,
                                            uint  *valid_list,
                                            const int n_groups,
                                            uint  *oriParticleOrder){
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint idx = bid * blockDim.x + tid;

  if (idx >= n_bodies) return;

  // C-D-03b: translate into the space bodies_time actually lives in.
  const uint timeIdx = g_cd03b_fix_index ? oriParticleOrder[idx] : idx;
  float te = time[timeIdx].y;

  //Set the group to active if the time current = time end of
  //this particle. Can be that multiple particles write to the
  //same location but the net result is the same
  int grpID = body2grouplist[idx];

  // BUG4-DIAG: temporary instrumentation to confirm/refute the out-of-bounds
  // hypothesis for the t_current corruption bug (PLAN.md open item). Does not
  // change behavior (write is unguarded exactly as before) -- just reports it.
  if (grpID < 0 || grpID >= n_groups) {
    printf("[BUG4-DIAG] OOB grpID write: idx=%u grpID=%d n_groups=%d n_bodies=%d tc=%.9g te=%.9g\n",
           idx, grpID, n_groups, n_bodies, tc, te);
  }

  //valid_list[grpID] = grpID | ((tc == te) << 31);

  // g_force_all_active: one-iteration full activation for Gadget-2's fresh-potential pass.
  if(g_force_all_active || tc == te)
  {
    valid_list[grpID] = grpID | (1 << 31);
  }
}


static __device__ __forceinline__ float adjustH(const float h_old, const float nnb)
{
	const float nbDesired 	= 32;
	const float f      	= 0.5f * (1.0f + cbrtf(nbDesired / nnb));
	const float fScale 	= max(min(f, 2.0), 0.5);
	return (h_old*fScale);
}

// Gadget-2 time integration rework: midpoint of an interval on Gadget-2's own integer timeline
// (timestep.c:254-255, `(Ti_begstep+Ti_endstep)/2`) is an ARITHMETIC mean of ti-ticks -- but ticks
// are uniform in log(a) under comoving integration (Timebase_interval = dlog(a)/TIMEBASE), while
// this port's time values are the scale factor `a` itself, not log(a) or a tick count. The tick-
// space arithmetic mean of two ticks therefore corresponds to the GEOMETRIC mean of the two `a`
// values (exp(0.5*(log a1 + log a2)) = sqrt(a1*a2)), not their arithmetic mean. Under non-comoving
// integration, ticks are uniform in linear physical time, matching this port's linear `tc`
// directly, so the arithmetic mean is correct there instead.
static __device__ __forceinline__ float tiMidpointD(const float a1, const float a2, const int comovingIntegrationOn)
{
  return comovingIntegrationOn ? sqrtf(a1 * a2) : 0.5f * (a1 + a2);
}

KERNEL_DECLARE(correct_particles)(const int n_bodies,
                                  /*  1 */   float tc,
                                  /*  2 */   float2 *time,
                                  /*  3 */   uint   *active_list,
                                  /*  4 */   real4 *vel,
                                  /*  5 */   real4 *acc0,
                                  /*  6 */   real4 *acc1,
                                  /*  7 */   float   *body_h,
                                  /*  8 */   float2  *body_dens,
                                  /*  9 */   real4 *pos,
                                  /* 10 */   real4 *pPos,
                                  /* 11 */   real4 *pVel,
                                  /* 12 */   uint  *unsorted,
                                  /* 13 */   real4 *acc0_new,
                                  /* 14 */   float2 *time_new,
                                  /* 15 */   int    comovingIntegrationOn,
                                  /* 16 */   const float *gravKickTable,
                                  /* 17 */   double logTimeBegin,   // T29: double
                                  /* 18 */   double logTimeMax,
                                  /* 19 */   const float *newEnd)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid =  threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;
  if (idx >= n_bodies) return;

  //Check if particle is set to active during approx grav
  #ifdef DO_BLOCK_TIMESTEP
    // BUG4 FIX: acc0_new/time_new (real4Buffer1/float2Buffer, aliased views into the shared,
    // reused generalBuffer1 scratch pool) are only ever written for ACTIVE particles below.
    // The caller (gpu_iterate.cpp's correct()) unconditionally copies ALL n_bodies entries of
    // these scratch buffers back into the real bodies_acc0/bodies_time -- so an inactive
    // particle's slot, if simply skipped here, retains whatever unrelated data generalBuffer1's
    // last consumer left there, silently clobbering that particle's real acc0/time with garbage.
    // This was invisible in every prior test because they all had every particle active every
    // step (a uniform shared timestep); it was exposed by a real close encounter that, for the
    // first time in a run, drove a small subset of particles onto a much finer step than the
    // rest -- confirmed empirically (nActive dropped from 32768/32768 to 32/32768 exactly at the
    // corruption iteration, matching every corrupted probe index to active_list[idx]==0). Fix:
    // pass through the particle's existing (old-order) acc0/time unchanged instead of leaving its
    // scratch slot untouched, using the same unsortedIdx indirection the active path uses below.
    const uint cd03_unsorted = unsorted[idx];
    // C-D-03: a particle promoted by its group but not actually due keeps drifting, is not kicked,
    // and keeps its existing time bin -- exactly what Gadget-2 does for a non-due particle.
    const bool cd03_notDue = g_cd03_strict_active && (time[cd03_unsorted].y != tc);
    if (active_list[idx] != 1 || cd03_notDue)
    {
      const uint unsortedIdx = cd03_unsorted;
      acc0_new[idx] = acc0[unsortedIdx];
      time_new[idx] = time[unsortedIdx];

      // TICKET T12 FIX: pos/vel (bodies_pos/bodies_vel) are NEVER reordered by sort_bodies()'s
      // default per-iteration resort (only bodies_Ppos/bodies_ids/bodies_h are -- see
      // sort_bodies_gpu.cpp's !doFullShuffle branch, the branch the normal per-step call in
      // gpu_iterate.cpp actually takes). Before this fix, this early return left pos[idx]/
      // vel[idx] completely untouched for inactive particles, so the raw memory slot `idx`
      // silently kept whatever a DIFFERENT particle had written there the last time IT was
      // active at that same raw index -- because bodies_ids DOES get reordered every step but
      // bodies_pos/bodies_vel do not, "idx"'s owning particle identity churns underneath them.
      // This produced a bit-exact cross-particle position+velocity aliasing bug (that particle's
      // own acc0/time staying self-consistent while pos/vel silently became another particle's
      // old state) -- confirmed via direct mechanism-level trace, not inference: see
      // tickets/T12-fix-correct-particles-buffer-staging.md Resolution for the exact log lines
      // (dumpUnsortedMapping()'s pre-correct() raw dump caught id=9547 sitting inactive at
      // idx=6011 with preCorrectPos/Vel already bit-identical to id=9291's own AFTER_CORRECT
      // commit from the previous iteration, before this iteration's correct_particles() kernel
      // had run at all).
      //
      // Fix: explicitly carry this particle's own correct state forward using buffers that ARE
      // already correctly ordered/computed for it regardless of active status:
      //  - pPos[idx] (bodies_Ppos) IS reordered into current order every iteration, and
      //    predict_particles() computes it for EVERY particle unconditionally (pure drift
      //    extrapolation, no active-list check there at all) -- this is not a hack, it matches
      //    real Gadget-2 semantics directly: an inactive particle's true position between its own
      //    sync points genuinely is just its drift prediction (see predict_particles()'s own
      //    comment above on Gadget-2's KDK integrator).
      //  - pVel[unsortedIdx] (bodies_Pvel) is NOT reordered (same default-branch omission as
      //    acc0/time just above), so the unsortedIdx translation the active path already relies
      //    on below is equally valid here to fetch this particle's own correct pre-resort
      //    velocity (inactive particles are never kicked, so no kick term is applied, matching
      //    Gadget-2: velocity only changes at a particle's own sync event).
      // Neither pPos nor pVel is ever WRITTEN by this kernel (only read), and pos/vel are only
      // ever written at each thread's own idx (never read by this kernel at all) -- so unlike
      // acc0/time above, this needs no separate scratch+copyback buffer; writing directly into
      // pos[idx]/vel[idx] here is race-free.
      pos[idx] = pPos[idx];
      vel[idx] = pVel[unsortedIdx];

      unsorted[idx] = idx;
      return;
    }
  #endif


#if 0
  float4 v  = vel [idx];
  float4 a1 = acc1[idx];
  float  tb = time[idx].x;
  v = pVel[idx];
#else
  const uint unsortedIdx = unsorted[idx];

  float4 a1 = acc1[idx];
  float  tb = time[unsortedIdx].x;   // old Ti_begstep (of the step that just ended)
  float4 v  = pVel[unsortedIdx];

#endif

  //Store the predicted position as the one to use
  pos[idx] = pPos[idx];

  // Gadget-2 time integration rework (was: PEC leapfrog `v += 0.5*(a1-a0)*dt_kick` over the single
  // just-completed interval [tb, tc] -- see predict_particles' comment for why that was wrong).
  // Real Gadget-2's kick (timestep.c:254-282) spans the midpoint of the OLD step [tb,tc] to the
  // midpoint of the NEW step [tc,newEnd[idx]], applied with only the newly-computed force `a1` --
  // there is no averaging with the old force `a0` at all, because the kick that would have used
  // a0 was already applied, symmetrically, at the PREVIOUS sync event.
  // C-D-02: end of the step that just ended -- the particle's own stored Ti_endstep, not the
  // global sync time (identical whenever the particle is genuinely due; different exactly when
  // group activation promoted it early).
  const float teOld  = g_cd02_kick_tend ? time[unsortedIdx].y : tc;
  const float tstart = tiMidpointD(tb, teOld, comovingIntegrationOn);        // midpoint of old step
  const float tend   = tiMidpointD(tc, newEnd[idx], comovingIntegrationOn);  // midpoint of new step

  float dt_gravkick = tend - tstart;
  if (comovingIntegrationOn)
    dt_gravkick = driftFactorD(gravKickTable, logTimeBegin, logTimeMax, tstart, tend);

  v.x += a1.x*dt_gravkick;
  v.y += a1.y*dt_gravkick;
  v.z += a1.z*dt_gravkick;


  //Store the corrected velocity, accelaration and the new time step info
  vel     [idx] = v;
  acc0_new[idx] = a1;
  time_new[idx] = make_float2(tc, newEnd[idx]);  // Ti_begstep = old Ti_endstep; Ti_endstep = new
  unsorted[idx] = idx;  //Have to reset it in case we do not resort the particles

  //Adjust the search radius for the next iteration to get closer to the
  //requested number of neighbours
#if BONSAI_DENSITY
  body_h[idx] = adjustH(body_h[idx], body_dens[idx].y);
#endif
}



// Phase 5 ticket 04 (PLAN.md): Gadget-2's exact get_timestep() criterion 0
// (PHASE5_ZOOM_COMOVING_SPEC.md Sec 3.1, timestep.c:466-535), replacing Bonsai's own eta/
// neighbor-distance heuristic entirely -- that heuristic turned out to already be fully dead code
// in this build (every code path below it was unconditionally overwritten by `dt = timeStep`, a
// flat global timestep for every particle regardless of any physics; found while reading this
// kernel to plan the replacement, not something this ticket broke).
//
// All host-computed scalars (errTolIntAccuracy, atime, fac1, hubble_a, maxSizeTimestep,
// minSizeTimestep, dtDisplacement) come from GadgetParams + gadget_hubble_a()/
// gadget_find_dt_displacement_constraint() (gadget_cosmology.h) -- see gpu_iterate.cpp for where
// they're derived; under comoving integration atime=Time, fac1=1/Time^2, hubble_a=H(a), otherwise
// all three are 1 (matching timestep.c:47-60 exactly).
//
// bodies_forceSoftening holds each particle's already-resolved ForceSoftening (=2.8*
// SofteningTable, ticket 03) -- Gadget-2's own formula uses the UNSCALED SofteningTable, so it's
// divided back out here (a plain compile-time constant division, not a second device array).
//
// Deliberately always behaves as if NOSTOP_WHEN_BELOW_MINTIMESTEP were defined (silently clamps
// up to minSizeTimestep instead of aborting the run) -- aborting a GPU kernel mid-launch isn't a
// real option, and this is an explicit, documented choice, not a silently-assumed default.
//
// Power-of-two snapping (timestep.c:187-193's own `while(ti_min>ti_step) ti_min>>=1`, done at the
// *call site* in real Gadget-2, not inside get_timestep() itself) is done here relative to
// maxSizeTimestep rather than Gadget-2's own integer TIMEBASE-tick timeline: this port's `time[]`
// is a native floating-point time value (not an integer tick count), and building the full
// integer-timeline/log(a) machinery Gadget-2 uses to synchronize power-of-two steps across
// TimeBegin..TimeMax is comoving-integration machinery -- ticket 05's job, not this one's. Snapping
// relative to maxSizeTimestep instead reproduces the same qualitative hierarchical/synchronizable
// power-of-two behavior without requiring that machinery; revisit once ticket 05 lands.
extern "C"  __global__ void compute_dt(const int n_bodies,
                                       float    tc,
                                       float    errTolIntAccuracy,
                                       float    atime,
                                       float    fac1,
                                       float    hubble_a,
                                       float    maxSizeTimestep,
                                       float    minSizeTimestep,
                                       float    dtDisplacement,
                                       int      comovingFlag,
                                       float    *newEnd,
                                       float2   *time,          /* C-D-13: old (begin,end) pair */
                                       uint     *unsorted,      /* oriParticleOrder: time[] is in ORIGINAL order */
                                       real4    *bodies_acc,
                                       float    *bodies_forceSoftening,
                                       uint     *active_list,
                                       float    timelineSpan,
                                       float    timelineOrigin,
                                       float    timeMax){      /* C-D-10: end of the timespan */
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid =  threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;
  if (idx >= n_bodies) return;

  //Check if particle is set to active during approx grav
  if (active_list[idx] != 1) return;

  const real4 a = bodies_acc[idx];
  float ac = fac1 * sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
  if (ac == 0.0f) ac = 1.0e-30f;

  const float softeningTable = bodies_forceSoftening[idx] / 2.8f;

  float dt = sqrtf(2.0f * errTolIntAccuracy * atime * softeningTable / ac);
  dt *= hubble_a;

  if (dt >= maxSizeTimestep) dt = maxSizeTimestep;
  if (dt >= dtDisplacement)  dt = dtDisplacement;
  if (dt <  minSizeTimestep) dt = minSizeTimestep;

  // Phase 5 ticket 08 (PLAN.md): a hard floor `dt` can never go below, regardless of how small
  // `minSizeTimestep` is set (0.0 in every .param file this ticket tested, matching real
  // Gadget-2's own common convention). Real Gadget-2 has an analogous, if much finer, floor:
  // its integer timeline (`ti_step = dt/Timebase_interval`) makes ANY dt smaller than one
  // `Timebase_interval` tick (~1e-9 in delta-log(a) units for a typical TIMEBASE=2^29 and this
  // project's own TimeBegin/TimeMax range) impossible to represent, and it explicitly detects and
  // `endrun()`s on that condition (timestep.c ~528-541: "A timestep of size zero was assigned on
  // the integer timeline!") rather than silently continuing. This port has no such integer
  // timeline (dt is a plain float, no inherent minimum resolution) so nothing was stopping the
  // `while(q>dt) q*=0.5f` loop below from halving all the way to a literal 0.0f when `dt` itself
  // underflowed toward zero -- caught via this ticket's own end-to-end run: a genuine close
  // encounter (or accumulated float error near one) drove the required dt pathologically small,
  // `time[idx].y` collapsed to `tc+0` (or `tc*exp(0)=tc`, after the ticket's own log-time fix
  // above), the particle never advanced past that sync time again, and the GLOBAL `t_current`
  // (the min over all particles' next-sync-time) froze there permanently -- an infinite loop, not
  // a crash. `1e-8` is comparable to Gadget-2's own real integer-timeline floor for this project's
  // own time ranges, chosen to almost never bind for a well-resolved run while still guaranteeing
  // forward progress instead of a hang if a close encounter ever does need something this fine.
  const float dtFloor = 1.0e-8f;
  if (dt < dtFloor) dt = dtFloor;

  // T7 (contract C-D-07): the committed step must be large enough that the clock actually
  // ADVANCES in float32. Confirmed by direct probe of this kernel (probes/cd07_probe.cpp): the
  // old code applied `dtFloor` BEFORE the halving loop and the loop then halved straight past it
  // (0.03 -> ... -> 7.15e-9), after which `tc + dt == tc` for any tc >~ 0.1 and
  // `tc * expf(dt) == tc` for dt below ~1.2e-7 -- so `newEnd == tc`, get_Tnext's min stayed pinned
  // there, and the run froze. That is precisely the hang the floor was added to prevent.
  //
  // Gadget-2 cannot have this bug: its floor IS the integer timeline, and it VERIFIES rather than
  // assumes (timestep.c:535-552, `endrun(818)` on a zero-length step). The representability limit
  // is tc-dependent -- ~3e-8 at tc=0.5, ~2.4e-7 at tc=5 -- so no single constant works; it has to
  // be derived from tc and enforced AFTER (here, during) the power-of-2 snapping, not before it.
  // Folding it into the loop condition keeps the bin ladder intact rather than bolting a floor on
  // afterwards, which would break the power-of-2 structure.
  const float minRep = comovingFlag
      ? 4.0f * 1.1920929e-07f                       // 4 ulp of 1.0: expf(dt) must exceed 1.0f
      : 4.0f * (nextafterf(tc, 3.402823466e+38f) - tc);   // 4 ulp of tc: tc+dt must exceed tc

  // C-D-04: Gadget-2 halves the FULL timeline span down (timestep.c:189-193 on TIMEBASE), so a
  // bin is span/2^k. Seeding from maxSizeTimestep instead makes the bins powers of two of a
  // parameter, which is a different -- and generally coarser -- ladder.
  const float ladderSeed = (g_cd04_timeline_ladder && timelineSpan > 0.0f) ? timelineSpan
                                                                           : maxSizeTimestep;
  float q = ladderSeed;
  while (q > dt && (q * 0.5f) >= minRep) q *= 0.5f;
  dt = q;

  // C-D-13 (Gadget-2's SYNCHRONIZATION, timestep.c:239-245, live via its Makefile):
  //
  //     if(ti_step > (P[i].Ti_endstep - P[i].Ti_begstep))       /* wants to increase */
  //       if(((TIMEBASE - P[i].Ti_endstep) % ti_step) > 0)
  //         ti_step = P[i].Ti_endstep - P[i].Ti_begstep;        /* leave at old step */
  //
  // Every ti_step is a power of two and TIMEBASE is 2^k, so `(TIMEBASE - n) % s == 0` iff `s | n`.
  // The rule is therefore: you may GROW to step `s` only if your current end-tick is a multiple of
  // `s`. That maintains the invariant "a particle in bin 2^j sits on the 2^j lattice", which is what
  // keeps sync points on a nested lattice instead of letting bins drift into independent phases.
  // Shrinking is always allowed and preserves the invariant trivially.
  //
  // Why the C-D-05 tick snap does NOT already provide this: the snap makes every end time an exact
  // tick (representability); it says nothing about that tick being DIVISIBLE by the step
  // (commensurability). Both are needed.
  //
  // All of this is done in double deliberately -- see tickets/T29. One tick is only ~2.3-3.7 float32
  // ULP wide, so a float32 tick index cannot be trusted to be exact, and this rule is a divisibility
  // test where being off by one tick inverts the answer.
  if (g_cd04_timeline_ladder && timelineSpan > 0.0f)
  {
    const double dtick = (double) timelineSpan / gadget_timebase_ticks((double) timelineSpan);
    const double tcAx  = comovingFlag ? log((double) tc) : (double) tc;
    const long long n  = (long long) floor((tcAx - (double) timelineOrigin) / dtick + 0.5);

    // INDEX SPACE: bodies_time is NOT resorted by the per-iteration sort, so it is in ORIGINAL
    // order and must be read through oriParticleOrder -- exactly as correct_particles does
    // (`time[cd03_unsorted]`, :574). Reading time[idx] here compared the WRONG particle's old step
    // length; this rule is a divisibility test on that value, so a wrong read silently inverts the
    // decision. Same class of defect as C-B-13/C-C-18/C-D-08.
    const float2 told  = time[unsorted[idx]];
    const double begAx = comovingFlag ? log((double) told.x) : (double) told.x;
    const double endAx = comovingFlag ? log((double) told.y) : (double) told.y;
    const long long sOld = (long long) floor((endAx - begAx) / dtick + 0.5);
    const long long sNew = (long long) floor((double) dt / dtick + 0.5);
    atomicAdd(&g_cd13_reached, 1u);
    if (sOld > 0 && sNew > sOld) atomicAdd(&g_cd13_grow, 1u);

    // sOld <= 0 is the first step (Gadget starts every particle at Ti_begstep == Ti_endstep == 0,
    // where the test passes trivially because n == 0); leave those alone.
    if (g_cd13_sync && sOld > 0 && sNew > sOld && (n % sNew) != 0)
    {
      dt = (float) ((double) sOld * dtick);
      atomicAdd(&g_cd13_blocked, 1u);
    }
  }

  // Phase 5 ticket 08 (PLAN.md): a real, serious bug found via this ticket's own end-to-end
  // comoving run -- confirmed by direct source citation, not assumption. Gadget-2's own
  // get_timestep() (timestep.c:423-548) produces `dt` in DELTA-LOG(a) units under comoving
  // integration (its own comment: "convert the physical timestep to dloga"; MaxSizeTimestep/
  // MinSizeTimestep, matched exactly above, are real Gadget-2 .param values ALSO documented and
  // used as dloga ceilings there) -- Gadget-2's own integer timeline is log(a)-uniform
  // (Timebase_interval = (log(TimeMax)-log(TimeBegin))/TIMEBASE), so advancing by `dt` there
  // means `a_new = a_old * exp(dt)`, never `a_old + dt`. This port has no such integer timeline
  // (`tc` IS the literal scale factor `a`, a plain float, confirmed against every other caller of
  // t_current) but had been doing `tc + dt` regardless -- adding a delta-LOG(a) value directly to
  // a linear `a`. Caught empirically: a real phase0 run showed a CONSTANT a-space step size
  // (0.03, exactly MaxSizeTimestep) across early iterations instead of a constant LOG(a)-space
  // one, then diverged (Epot collapsing from -1.58e9 to -9.6e6 in one step, then state freezing
  // with a corrupted `t_current`) a few steps later -- log(0.045625/0.015625)=1.07, not 0.03,
  // proving the ceiling was constraining the wrong quantity 30x too loosely at this early, small-a
  // part of the run where linear- and log-space steps differ most. Only applied when
  // comovingFlag!=0 (this port's own existing `haveGadgetComovingTables` idiom, already used
  // identically by correctParticles's `correctComovingFlag`) -- a non-comoving or non-`--param`
  // run's `dt` is genuinely linear physical time (Gadget-2's own non-comoving branch: no `a` at
  // all, Timebase_interval is linear), where `tc + dt` was already correct and must stay that way.
  //
  // Gadget-2 time integration rework: this kernel no longer writes `time[]` in place. Real
  // Gadget-2's advance_and_find_timesteps() (timestep.c:254-282) needs the OLD Ti_begstep/
  // Ti_endstep pair still intact when it computes the actual kick (the kick spans the midpoint of
  // the step that just ended to the midpoint of this new step) -- so overwriting time[idx].x here,
  // before correct_particles() gets a chance to read the old begin time, would destroy exactly the
  // information the new kick formula needs. The new end-of-step time is instead handed to
  // correct_particles() via this separate scratch buffer; it commits the new (begin,end) pair to
  // `time[]` itself, after computing the kick from the still-original values.
  // T25: land on the global tick grid rather than wherever the float multiply falls, so particles
  // that should re-synchronise do so bit-exactly. Guarded: if snapping would fail to advance the
  // clock (dt below one tick) keep the unsnapped value, which still advances and is what the T7
  // check below then reports -- never trade a real advance for an exact-but-stalled one.
  {
    // Form the step-end time in DOUBLE and hand the snap that value directly -- see
    // snapToTimelineTick's comment. Rounding to float32 here first is what made the snap miss the
    // tick (and hence the rendezvous) in a measured 27.8% of cases.
    const double rawD    = comovingFlag ? (double) tc * exp((double) dt)
                                        : (double) tc + (double) dt;
    const float  raw     = (float) rawD;
    const float  snapped = snapToTimelineTick(rawD, timelineOrigin, timelineSpan, comovingFlag);
    float end = (snapped > tc) ? snapped : raw;

    // C-D-10: no particle may be scheduled beyond the end of the simulated timespan. Gadget-2
    // truncates the step so the last one lands exactly on TIMEBASE (timestep.c:248-252):
    //     if(All.Ti_Current == TIMEBASE)              ti_step = 0;
    //     if((TIMEBASE - All.Ti_Current) < ti_step)   ti_step = TIMEBASE - All.Ti_Current;
    // so All.Time == All.TimeMax exactly at the end and the final snapshot sits at the requested
    // epoch. The port had no such truncation -- `tEnd` was used ONLY in the loop-exit test, which
    // runs after predict/gravity/correct, so the run finished at whatever time the last step
    // happened to land on. The overshoot is bounded by one system step, but it means the port's
    // final state is at a different epoch from Gadget's, which silently corrupts any final-state
    // cross-code comparison.
    //
    // Guarded on `timeMax > tc` so a particle already at (or past) the end is left alone rather
    // than given a zero-length step, which C-D-14's check would correctly treat as fatal; the
    // driver's loop-exit test ends the run in that case, exactly as Gadget's does.
    if (timeMax > 0.0f && timeMax > tc && end > timeMax)
      end = timeMax;

    newEnd[idx] = end;
  }

  // C-D-14: deliberate stall injection, off unless GADGET_HIP_T7_INJECT=1 (see g_t7_injectStall).
  if (g_t7_injectStall && idx == 0) newEnd[idx] = tc;

  // T7: Gadget-2's endrun(818) equivalent -- verify, do not assume. If the clock still fails to
  // advance the run is provably stuck, so record it for the host rather than hanging silently.
  // C-D-14 + C-C-07 interaction: on an energy-statistics step `energy_set_force_all_active` marks
  // EVERY group active so the potential pass is complete, so this kernel runs for all N particles --
  // including ones that are not actually due, whose newEnd correct_particles discards moments later
  // (its `cd03_notDue` pass-through, :574-575). Without this gate the abort fires for such a
  // particle and kills a healthy run, and only ever on statistics iterations, which is maximally
  // confusing. Report only for particles that are genuinely due, i.e. the ones whose step is really
  // committed.
  const bool t7_notDue = g_cd03_strict_active && (time[unsorted[idx]].y != tc);
  if (!t7_notDue && !(newEnd[idx] > tc))
  {
    if (atomicAdd(&g_t7_nonadvancing, 1u) == 0u)
    {
      g_t7_firstIdx = idx;
      g_t7_firstTc  = tc;
      g_t7_firstEnd = newEnd[idx];
      g_t7_firstDt  = dt;
      g_t7_firstAc  = ac;
    }
  }
}



// T30 (C-C-06 / C-C-09 / C-C-10): Gadget-2 lets the potential walk include a particle's
// interaction with ITSELF and then subtracts it afterwards (potential.c:246-254). The port's walk
// produces the identical self term -- target and source are the same buffer, there is no self-skip,
// and at r == 0 the spline gives wp = -2.8, so the pair contributes -G*m/SofteningTable -- but the
// port never removed it. Measured on the phase0 IC: the self term is 87% of the entire reported
// |Epot|, and the reported Epot was 3.11x too deep against a COMPUTE_POTENTIAL_ENERGY reference.
//
// Two terms, and they must ship together: removing only the self term overshoots the reference by
// ~38% in the other direction.
//
//   self  : Potential += m / SofteningTable          (potential.c:249)
//   comov : Potential -= 2.8372975 * m^(2/3) * (Omega0*3*H^2/(8*pi*G))^(1/3)   (potential.c:251-254)
//
// both then scaled by All.G (potential.c:261). The port's acc.w is ALREADY G-scaled
// (pm_scale_acc_masked), so G is carried in the coefficients below.
//
// `forceSoftening` is 2.8 * SofteningTable, hence the 2.8 in the numerator -- the removal uses
// SofteningTable, NOT ForceSoftening. Getting that wrong is a factor 2.8.
//
// Deliberately NOT attenuated by erfc(u0): Gadget's own removal at potential.c:249 carries no such
// factor, so it over-removes by 0.17% -- and copying that exactly is what reproduces the reference
// (residual 1e-6 relative; the "corrected" attenuated form leaves 0.46%). Faithfulness beats
// local improvement here.
//
// selfCoef = G * 2.8 ; cc10Coef = G * 2.8372975 * rho_mean^(1/3), or 0 when the run is not
// comoving+periodic (Gadget guards it the same way).
static __device__ __forceinline__ double epotSelfCorrection(
    double mass, float forceSoftening, double selfCoef, double cc10Coef)
{
  double c = 0.0;
  if (forceSoftening > 0.0f)
    c += 0.5 * mass * selfCoef * mass / (double) forceSoftening;
  if (cc10Coef != 0.0)
    c -= 0.5 * mass * cc10Coef * cbrt(mass * mass);
  return c;
}

//Reduce function to get the energy of the system in double precision
static __device__ void compute_energy_doubleD(const int n_bodies,
                                            real4 *pos,
                                            real4 *vel,
                                            real4 *acc,
                                            const float *forceSoftening,
                                            double selfCoef,
                                            double cc10Coef,
                                            double2 *energy, volatile double *shDDataKin) {

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const int blockSize   = blockDim.x;
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  volatile double *shDDataPot = (double*)&shDDataKin [blockSize];
  double eKin, ePot;
  shDDataKin[tid] = eKin = 0;   //Stores Ekin
  shDDataPot[tid] = ePot = 0;   //Stores Epot

  real4 temp;
  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n_bodies) {
    if (i             < n_bodies)
    {
      temp  = vel[i];
      eKin += pos[i].w*0.5*(temp.x*temp.x + temp.y*temp.y + temp.z*temp.z);
      ePot += pos[i].w*0.5*acc[i].w;
      ePot += epotSelfCorrection((double) pos[i].w,
                                 forceSoftening ? forceSoftening[i] : 0.0f,
                                 selfCoef, cc10Coef);   // T30
    }

    if (i + blockSize < n_bodies)
    {
      temp  = vel[i + blockSize];
      eKin += pos[i + blockSize].w*0.5*(temp.x*temp.x + temp.y*temp.y + temp.z*temp.z);
      ePot += pos[i + blockSize].w*0.5*acc[i + blockSize].w;
    }

    i += gridSize;
  }
  shDDataKin[tid] = eKin;
  shDDataPot[tid] = ePot;

  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 256];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 256];   } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 128];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 128];   } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 64];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 64];   } __syncthreads(); }


  // Unsynchronised warp-reduction tail replaced (see gpu_boundaryReduction in build_tree.cu for
  // the instance where this was caught failing). EMUSYNC expands to NOTHING here, so these were
  // dependent shared-memory steps with no barrier, relying on lockstep lanes. Synchronised loop
  // instead; __syncthreads() is reached by every thread in the block.
  for (unsigned int s = 32; s > 0; s >>= 1)
  {
    if (blockSize >= 2 * s && tid < s) { shDDataKin[tid] = eKin = eKin + shDDataKin[tid + s]; shDDataPot[tid] = ePot = ePot + shDDataPot[tid + s]; }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) energy[blockIdx.x] = make_double2(shDDataKin[0], shDDataPot[0]);
}


//Reduce function to get the energy of the system
KERNEL_DECLARE(compute_energy_double)(const int n_bodies,
                                            real4 *pos,
                                            real4 *vel,
                                            real4 *acc,
                                            const float *forceSoftening,
                                            double selfCoef,
                                            double cc10Coef,
                                            double2 *energy) {
  extern __shared__ double shDDataKin[];
  compute_energy_doubleD(n_bodies, pos, vel, acc, forceSoftening, selfCoef, cc10Coef,
                          energy, shDDataKin);
}

// T19 (2026-09-02): energy diagnostic sampled at the WRONG point in the KDK cycle relative to
// Gadget-2's own convention. gpu_iterate.cpp's compute_energies() was called AFTER correct()
// (using the POST-kick velocity, valid at the midpoint of the NEW step) while Gadget-2's own
// energy_statistics() (Gadget2/run.c:47-59) runs BEFORE advance_and_find_timesteps()'s kick --
// i.e. it uses the PRE-kick velocity (valid at the midpoint of the OLD step) paired with the
// freshly-drifted position and freshly-computed potential. This kernel reproduces Gadget-2's
// actual sampling point: `pos`/`acc` are this step's already-fresh (pPos/acc1) values (in the
// current, possibly-just-resorted tree order), but `vel` must be looked up via the SAME
// `unsorted[]` (oriParticleOrder) indirection correct_particles() itself uses just below
// (`pVel[unsortedIdx]`) -- because bodies_Ppos/bodies_acc1 are in this iteration's freshly-sorted
// tree order while bodies_Pvel (unlike bodies_Ppos) is NOT resorted, the exact asymmetry Ticket 12
// already root-caused for correct_particles(). Getting this indirection wrong would silently pair
// the wrong particle's velocity with the wrong particle's position/potential.
// T14/energy: Gadget-2 extrapolates each particle's velocity to the CURRENT sync time before
// summing kinetic energy (global.c:60-72):
//     dt_gravkick = gravkick(Ti_begstep, Ti_Current) - gravkick(Ti_begstep, mid(Ti_beg,Ti_end))
//                 = gravkick(mid, Ti_Current)
//     vel_used    = Vel + GravAccel * dt_gravkick
// In KDK the stored velocity is valid at the MIDPOINT of the particle's own step, so without this
// term a particle that is mid-step contributes its kinetic energy at the wrong time. Exactly inert
// when every particle is active (then mid == tc for all of them), which is why this never showed up
// in any synchronous test. Gated by g_energy_extrap.
static __device__ void compute_energy_double_prekickD(const int n_bodies,
                                            real4 *pos,
                                            real4 *vel,
                                            real4 *acc,
                                            uint  *unsorted,
                                            float2 *time,
                                            float  tc,
                                            int    comovingFlag,
                                            const float *gravKickTable,
                                            double logTimeBegin,   // T29: double
                                            double logTimeMax,
                                            const float *forceSoftening,
                                            double selfCoef,
                                            double cc10Coef,
                                            double2 *energy, volatile double *shDDataKin) {

  const int blockSize   = blockDim.x;
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  volatile double *shDDataPot = (double*)&shDDataKin [blockSize];
  double eKin, ePot;
  shDDataKin[tid] = eKin = 0;
  shDDataPot[tid] = ePot = 0;

  while (i < n_bodies) {
    for (int half = 0; half < 2; half++)
    {
      const unsigned int j = i + (half ? blockSize : 0);
      if (j >= n_bodies) continue;
      const uint  u = unsorted[j];
      real4 temp = vel[u];
      if (g_energy_extrap)
      {
        const float2 t    = time[u];
        const float  tmid = tiMidpointD(t.x, t.y, comovingFlag);
        const float  dtk  = comovingFlag
            ? driftFactorD(gravKickTable, logTimeBegin, logTimeMax, tmid, tc)
            : (tc - tmid);
        const real4  a    = acc[j];
        temp.x += a.x * dtk;
        temp.y += a.y * dtk;
        temp.z += a.z * dtk;
      }
      eKin += pos[j].w*0.5*(temp.x*temp.x + temp.y*temp.y + temp.z*temp.z);
      ePot += pos[j].w*0.5*acc[j].w;
      // T30: remove the self-potential the walk necessarily included, and apply Gadget's
      // comoving+periodic normalisation. See epotSelfCorrection above.
      ePot += epotSelfCorrection((double) pos[j].w,
                                 forceSoftening ? forceSoftening[j] : 0.0f,
                                 selfCoef, cc10Coef);
    }
    i += gridSize;
  }
  shDDataKin[tid] = eKin;
  shDDataPot[tid] = ePot;

  __syncthreads();

  if (blockSize >= 512) { if (tid < 256) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 256];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 256];   } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 128];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 128];   } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 64];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 64];   } __syncthreads(); }

  // Unsynchronised warp-reduction tail replaced (see gpu_boundaryReduction in build_tree.cu for
  // the instance where this was caught failing). EMUSYNC expands to NOTHING here, so these were
  // dependent shared-memory steps with no barrier, relying on lockstep lanes. Synchronised loop
  // instead; __syncthreads() is reached by every thread in the block.
  for (unsigned int s = 32; s > 0; s >>= 1)
  {
    if (blockSize >= 2 * s && tid < s) { shDDataKin[tid] = eKin = eKin + shDDataKin[tid + s]; shDDataPot[tid] = ePot = ePot + shDDataPot[tid + s]; }
    __syncthreads();
  }

  if (tid == 0) energy[blockIdx.x] = make_double2(shDDataKin[0], shDDataPot[0]);
}

KERNEL_DECLARE(compute_energy_double_prekick)(const int n_bodies,
                                            real4 *pos,
                                            real4 *vel,
                                            real4 *acc,
                                            uint  *unsorted,
                                            float2 *time,
                                            float  tc,
                                            int    comovingFlag,
                                            const float *gravKickTable,
                                            double logTimeBegin,   // T29: double
                                            double logTimeMax,
                                            const float *forceSoftening,
                                            double selfCoef,
                                            double cc10Coef,
                                            double2 *energy) {
  extern __shared__ double shDDataKin[];
  compute_energy_double_prekickD(n_bodies, pos, vel, acc, unsorted, time, tc, comovingFlag,
                                  gravKickTable, logTimeBegin, logTimeMax,
                                  forceSoftening, selfCoef, cc10Coef, energy, shDDataKin);
}

// Phase 5 ticket 04 (PLAN.md): direct-launch test wrapper for compute_dt() above, bypassing the
// legacy my_dev::kernel .set_args()/.execute2() wrapper entirely -- same idiom as pm.h's
// softening_test_pair -- so `--timestep-test` can exercise the real kernel with synthetic
// per-particle inputs and no tree/IC/MPI setup at all.
void timestep_test_compute_dt(int n, float tc, float errTolIntAccuracy, float atime, float fac1,
                               float hubble_a, float maxSizeTimestep, float minSizeTimestep,
                               float dtDisplacement, const float4 *h_acc,
                               const float *h_forceSoftening, float *h_dtOut, hipStream_t stream,
                               int comovingFlag)
{
  real4  *d_acc = nullptr;
  float  *d_soft = nullptr;
  float  *d_newEnd = nullptr;
  uint   *d_active = nullptr;

  hipMalloc((void**)&d_acc,   n * sizeof(real4));
  hipMalloc((void**)&d_soft,  n * sizeof(float));
  hipMalloc((void**)&d_newEnd,  n * sizeof(float));
  hipMalloc((void**)&d_active, n * sizeof(uint));

  hipMemcpy(d_acc,  h_acc,          n * sizeof(real4), hipMemcpyHostToDevice);
  hipMemcpy(d_soft, h_forceSoftening, n * sizeof(float), hipMemcpyHostToDevice);
  std::vector<uint> ones(n, 1u);
  hipMemcpy(d_active, ones.data(), n * sizeof(uint), hipMemcpyHostToDevice);
  hipMemset(d_newEnd, 0, n * sizeof(float));

  // C-D-13 added a `time` (begin,end) input. This harness passes timelineSpan = 0, which disables
  // both the ladder and the SYNCHRONIZATION rule, so the contents are never read -- but the buffer
  // must still be a valid allocation. Seed it with (tc,tc), the same "first step" state main.cpp
  // initialises real particles to, so the harness stays meaningful if the span is ever turned on.
  float2 *d_time = nullptr;
  CU_SAFE_CALL(hipMalloc((void**)&d_time, n * sizeof(float2)));
  {
    std::vector<float2> h_time(n, make_float2(tc, tc));
    hipMemcpy(d_time, h_time.data(), n * sizeof(float2), hipMemcpyHostToDevice);
  }
  // time[] is indexed through oriParticleOrder in the real call path; the harness is a flat
  // synthetic set, so identity is the correct mapping here.
  uint *d_unsorted = nullptr;
  CU_SAFE_CALL(hipMalloc((void**)&d_unsorted, n * sizeof(uint)));
  {
    std::vector<uint> h_uns(n);
    for (int i = 0; i < n; i++) h_uns[i] = (uint) i;
    hipMemcpy(d_unsorted, h_uns.data(), n * sizeof(uint), hipMemcpyHostToDevice);
  }

  const int block = 128;
  const dim3 grid((n + block - 1) / block, 1);
  hipLaunchKernelGGL(compute_dt, grid, dim3(block), 0, stream,
                      n, tc, errTolIntAccuracy, atime, fac1, hubble_a, maxSizeTimestep,
                      minSizeTimestep, dtDisplacement, comovingFlag, d_newEnd, d_time, d_unsorted,
                      d_acc, d_soft, d_active, /*timelineSpan=*/0.0f, /*timelineOrigin=*/0.0f,
                      /*timeMax=*/0.0f);   // C-D-10: clamp disabled in the synthetic harness
  hipDeviceSynchronize();

  std::vector<float> h_newEnd(n);
  hipMemcpy(h_newEnd.data(), d_newEnd, n * sizeof(float), hipMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) h_dtOut[i] = h_newEnd[i] - tc;

  hipFree(d_acc); hipFree(d_soft); hipFree(d_newEnd); hipFree(d_active); hipFree(d_time); hipFree(d_unsorted);
}

// Phase 5 ticket 05 (PLAN.md): direct-launch test for the device-side driftFactorD() lookup
// (same idiom as the two test wrappers above) -- confirms the DEVICE table interpolation matches
// the HOST one (gadget_driftfac.cpp), which is validated independently against SciPy elsewhere.
// One thread per (t0,t1,table-select) test case; `wantKick[i]` selects gravKickTable when
// nonzero, driftTable otherwise.
__global__ void dev_driftfac_test(const float *driftTable, const float *gravKickTable,
                                   double logTimeBegin, double logTimeMax,
                                   const float *t0, const float *t1, const int *wantKick,
                                   int n, float *out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float *table = wantKick[i] ? gravKickTable : driftTable;
  out[i] = driftFactorD(table, logTimeBegin, logTimeMax, t0[i], t1[i]);
}

void driftfac_test_lookup(const float *h_driftTable, const float *h_gravKickTable,
                           double logTimeBegin, double logTimeMax, const float *h_t0,
                           const float *h_t1, const int *h_wantKick, int n, float *h_out)
{
  float *d_drift = nullptr, *d_kick = nullptr, *d_t0 = nullptr, *d_t1 = nullptr, *d_out = nullptr;
  int   *d_want = nullptr;
  hipMalloc((void**)&d_drift, GADGET_DRIFT_TABLE_LENGTH * sizeof(float));
  hipMalloc((void**)&d_kick,  GADGET_DRIFT_TABLE_LENGTH * sizeof(float));
  hipMalloc((void**)&d_t0, n * sizeof(float));
  hipMalloc((void**)&d_t1, n * sizeof(float));
  hipMalloc((void**)&d_want, n * sizeof(int));
  hipMalloc((void**)&d_out, n * sizeof(float));

  hipMemcpy(d_drift, h_driftTable,    GADGET_DRIFT_TABLE_LENGTH * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_kick,  h_gravKickTable, GADGET_DRIFT_TABLE_LENGTH * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_t0, h_t0, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_t1, h_t1, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_want, h_wantKick, n * sizeof(int), hipMemcpyHostToDevice);

  const int block = 64;
  const int grid  = (n + block - 1) / block;
  hipLaunchKernelGGL(dev_driftfac_test, dim3(grid), dim3(block), 0, 0,
                      d_drift, d_kick, logTimeBegin, logTimeMax, d_t0, d_t1, d_want, n, d_out);
  hipDeviceSynchronize();

  hipMemcpy(h_out, d_out, n * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_drift); hipFree(d_kick); hipFree(d_t0); hipFree(d_t1); hipFree(d_want); hipFree(d_out);
}




// T35: per-particle softening/type refresh, on the device.
//
// Replaces a host round trip that ran every tree rebuild: d2h of bodies_ids (1.07 GB at 512^3),
// 134M std::unordered_map lookups, two O(N) host loops, then h2d of bodies_forceSoftening and
// bodies_typeDevice (1.07 GB back). Measured at 1.78 s of a 5.43 s sparse step -- 33% of the step,
// to recompute a value that is always one of six table entries.
//
// The RULE is unchanged: type = typeById[id], forceSoftening = table[type]. bodies_ids is already
// on the device and already in the current (post-sort) order, so no index-space question arises --
// this reads exactly the array the host path read, at exactly the same point in the step.
KERNEL_DECLARE(gadget_refresh_softening)(const int n,
                                         const ullong *ids,
                                         const unsigned char *typeById,
                                         const ullong maxId,
                                         const float4 forceSoftLo,   // table entries 0..3
                                         const float2 forceSoftHi,   // table entries 4..5
                                         float *forceSoftening,
                                         int   *typeDevice)
{
  // 2D-aware index, matching every other per-element kernel in this port (compute_scaling's own
  // decode). setWork() builds a 2D grid because 134M elements at 128 threads/block needs ~1.05M
  // blocks, past the 1D grid limit -- plain `blockIdx.x*blockDim.x + threadIdx.x` silently covers
  // only the first row of blocks and leaves most particles unwritten. That is exactly what it did:
  // wrong softenings, a timestep that would not advance, and Etot off in the fourth digit.
  const int bid = blockIdx.y * gridDim.x + blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int idx = bid * (blockDim.x * blockDim.y) + tid;
  if (idx >= n) return;

  const ullong id = ids[idx];
  // Out-of-range ids cannot happen once setGadgetTypeMap has validated the id space, but a silent
  // out-of-bounds read here would be a memory error rather than a wrong answer, so clamp to type 0
  // -- the same type an unmapped id would have had on the host path (`.at()` would have thrown,
  // which at 512^3 is a crash; this is strictly safer and equally visible in the softening report).
  const int t = (id <= maxId) ? (int) typeById[id] : 0;

  float fs;
  switch (t)
  {
    case 0:  fs = forceSoftLo.x; break;
    case 1:  fs = forceSoftLo.y; break;
    case 2:  fs = forceSoftLo.z; break;
    case 3:  fs = forceSoftLo.w; break;
    case 4:  fs = forceSoftHi.x; break;
    default: fs = forceSoftHi.y; break;
  }
  forceSoftening[idx] = fs;
  typeDevice[idx]     = t;
}

// T35: per-type (count, sum |v|^2, min mass) reduction on the device.
//
// Replaces, per tree rebuild: a 4.8 GB device-to-host copy (bodies_Ppos + bodies_vel +
// oriParticleOrder), two 134M-element host vector allocations, and a 134M-element host loop --
// all to produce eighteen numbers. Measured at 1.06 s of a 3.63 s sparse step at 512^3.
//
// The rule is copied exactly from the host loop it replaces, INCLUDING the C-D-08 index
// translation: mass and type come from the current (post-sort) order, while the velocity is read
// through oriParticleOrder because bodies_vel is not resorted. Reading vel[idx] directly here
// would reintroduce precisely the defect C-D-08 fixed -- it is invisible for a single-species run
// (one bin) and wrong the moment a second populated type exists.
//
// Accumulation is in double via atomicAdd on 6 bins. The bin count is tiny and contention is
// therefore high, so each block reduces privately in shared memory first and commits once.
KERNEL_DECLARE(gadget_timestep_globals)(const int n,
                                        const real4 *pos,          // current order: mass in .w
                                        const real4 *vel,          // ORIGINAL order
                                        const uint  *unsorted,     // oriParticleOrder
                                        const int   *typeDevice,   // current order
                                        double *outVSum,           // [6]
                                        double *outMinMass,        // [6]
                                        unsigned long long *outCount) // [6]
{
  __shared__ double sVSum[6];
  __shared__ double sMinMass[6];
  __shared__ unsigned long long sCount[6];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (tid < 6) { sVSum[tid] = 0.0; sMinMass[tid] = 1.0e30; sCount[tid] = 0; }
  __syncthreads();

  const int bid = blockIdx.y * gridDim.x + blockIdx.x;
  const int idx = bid * (blockDim.x * blockDim.y) + tid;

  if (idx < n)
  {
    const int t = typeDevice[idx];
    if (t >= 0 && t < 6)
    {
      const uint  o = unsorted[idx];
      const real4 v = vel[o];
      const double v2 = (double) v.x * v.x + (double) v.y * v.y + (double) v.z * v.z;
      atomicAdd(&sVSum[t], v2);
      atomicAdd(&sCount[t], 1ull);
      // No double atomicMin; CAS on the bit pattern. Masses here are positive, so the ordering of
      // positive doubles matches the ordering of their bit patterns as unsigned integers.
      const double m = (double) pos[idx].w;
      unsigned long long mBits = __double_as_longlong(m);
      unsigned long long old = __double_as_longlong(sMinMass[t]);
      while (mBits < old)
      {
        const unsigned long long prev =
            atomicCAS((unsigned long long *) &sMinMass[t], old, mBits);
        if (prev == old) break;
        old = prev;
      }
    }
  }
  __syncthreads();

  if (tid < 6)
  {
    if (sCount[tid] > 0)
    {
      atomicAdd(&outVSum[tid], sVSum[tid]);
      atomicAdd(&outCount[tid], sCount[tid]);
      unsigned long long mBits = __double_as_longlong(sMinMass[tid]);
      unsigned long long old   = __double_as_longlong(outMinMass[tid]);
      while (mBits < old)
      {
        const unsigned long long prev =
            atomicCAS((unsigned long long *) &outMinMass[tid], old, mBits);
        if (prev == old) break;
        old = prev;
      }
    }
  }
}


// T37: Gadget-2's do_box_wrapping (predict.c:103-129), which this port never had.
//
// Gadget maps particles back onto the box at every domain decomposition (domain.c:81), i.e. after
// the drift and BEFORE the tree is built and before the PM solve. That ordering is not incidental:
// domain.c:67-73 forces a decomposition whenever a PM step is due, with the comment "this is
// needed to make sure that the particles are wrapped into the box". In the reference, in-box
// coordinates are a precondition of the mesh assignment.
//
// Without this the coordinates accumulate each particle's full unwrapped trajectory. Measured on
// the 512^3 -> z=0 run: 0.7% of particles outside [0,BoxSize) at a=0.05, growing to 11.9% and
// -15.2..113.7 Mpc/h by z=0. Every downstream consumer that assumes periodic coordinates -- the
// halo finder that caught this, power spectra, projections -- then reads those literally.
//
// One array, not two: bodies_pos is the C-A-22 memory fold of bodies_Ppos -- the same allocation
// under two names, so correct_particles' `pos[idx] = pPos[idx]` is a self-assignment and Gadget's
// single P[].Pos is what both names refer to. bodies_pos's own dev_mem handle is a one-element
// stub (build.cpp:45), so writing n elements through it faults; wrapping Ppos wraps both.
//
// This changes no physics. Every consumer of these coordinates already works modulo BoxSize (the
// walk's nearest-image convention; the CIC's own bin modulo in pm_cic.cu), so translating a
// coordinate by a whole box is exactly the identity for them -- it only stops the excursion from
// growing without bound, and with it the tree's root box, which had inflated from 100 to ~129
// Mpc/h by z=0 and coarsened the key quantisation for no reason.
__device__ __forceinline__ float t37_wrap_coord(float x, const float boxSize)
{
  // fmodf takes the sign of x, so a negative coordinate needs one further add. The final guard is
  // the float32 edge case: a value just below boxSize can round UP to exactly boxSize when the add
  // is rounded, which would land outside [0,boxSize) again. Gadget's while-loops cannot hit this
  // because it works in double.
  float r = fmodf(x, boxSize);
  if (r < 0.0f) r += boxSize;
  if (r >= boxSize) r = 0.0f;
  return r;
}

KERNEL_DECLARE(gadget_box_wrap)(const int n_bodies, const float boxSize, real4 *pPos)
{
  // 2D-aware index: setWork() builds a 2D grid because 134M elements at 128 threads/block needs
  // more blocks than a 1D grid allows, and a plain blockIdx.x decode silently covers only the
  // first row -- the exact bug that made gadget_refresh_softening leave most particles unwritten.
  const int bid = blockIdx.y * gridDim.x + blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int idx = bid * (blockDim.x * blockDim.y) + tid;
  if (idx >= n_bodies) return;

  // .w is mass and must survive untouched.
  real4 pp = pPos[idx];
  pp.x = t37_wrap_coord(pp.x, boxSize);
  pp.y = t37_wrap_coord(pp.y, boxSize);
  pp.z = t37_wrap_coord(pp.z, boxSize);
  pPos[idx] = pp;
}
