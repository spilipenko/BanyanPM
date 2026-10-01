#pragma once
#include <hip/hip_runtime.h>
#include "gadget_units.h"   // GFreeAcc: the PM solve is called with G = 1

// Phase 4 (PLAN.md): PM long-range gravity solver building blocks. New code, deliberately NOT
// routed through Bonsai's legacy my_dev::kernel create()/set_args()/execute() wrapper -- that
// machinery exists to support the project's old dual OpenCL/CUDA backend and buys nothing for
// fresh HIP-only code. Plain __global__ kernels, launched directly via hipLaunchKernelGGL from a
// plain host function.

#ifdef PERIODIC
// Uploads the periodic tree-walk constants (box size, Rcut, Asmth, and the precomputed
// Gadget-2 shortrange_table) into the gravity kernel's __constant__ memory
// (CUDAkernels/dev_approximate_gravity_warp_new.cu, where the symbols and the function itself are
// defined -- hipMemcpyToSymbol needs same-translation-unit visibility of the symbols). Call once
// per run, before the first gravity kernel launch -- these don't change step to step, matching
// Gadget-2's own All.BoxSize/All.Rcut/All.Asmth globals and its one-time force_treeallocate table
// init (GADGET2_NOTES.md).
void pm_periodic_setup_gravity_kernel(float boxSize, float rcut, float asmth);
// C-C-03: the box size the gravity kernels were set up with, for translation units that cannot
// read the device __constant__ (e.g. dev_direct_gravity.cu). 0 when PERIODIC is not compiled.
float pm_get_periodic_boxsize();
#endif

// Ticket T27 (gadget-audit map): softening-based bJ floor for split_node_grav_springel
// (dev_approximate_gravity_warp_new.cu). Unconditional (no #ifdef) -- unlike the PM/PERIODIC setup
// functions above, this needs no box size or PM grid, so it applies to every build including
// non-periodic, non-PMGRID isolated tests. Call once from main.cpp right after IC load, same place
// as the PM setup calls above. defaultCoef is normally 0.0f (disabled); GADGET_HIP_T27_SOFT_COEF
// env var overrides it at runtime for calibration without a rebuild.
void t27_set_soft_floor_coef(float defaultCoef);

// Ticket T4 (contract C-B-02): Springel-MAC open census, primary relative test vs. proximity
// box vs. bJ<=0 force-open. Defined in dev_approximate_gravity_warp_new.cu (the __device__
// symbols need same-translation-unit visibility for hipMemcpyToSymbol, same as traceY_*).
// Ticket T6 (contract C-D-01): drift-anchor selector, defined in timestep.cu.
void t6_set_drift_anchor(int fromTp);

// Ticket T9 (contracts C-B-06 / C-B-10b): fixed octree cell size as the MAC node-size term.
void t9_set_cellsize_geo(int on);
void t9_set_cellsize_bj(int on);

// Phase 0b: out-of-range oriParticleOrder entries seen by gadget_timestep_globals.
void dtglobals_reset_badidx(void);
unsigned int dtglobals_read_badidx(void);
void t9_set_no_s(int on);

// Ticket T10: per-rebuild group-boundary rotation (decorrelates the group-corner force bias).
void t10_set_group_shift(int enabled);

// Ticket T11 (contract C-B-05): proximity box referenced to the node geometric centre.
void t11_set_prox_center(int on);
void ca01_set_cell_center(int on);
void ca01_set_cc_skip(int on);

// Ticket T7 (contract C-D-07): non-advancing-timestep detector (Gadget-2 endrun(818) equivalent).
// Contracts C-D-02 / C-D-03: kick-interval end point and per-particle activation predicate.
void cd02_set_kick_tend(int on);
void cd03_set_strict_active(int on);
void cd03b_set_fix_index(int on);
void energy_set_extrap(int on);
void cd04_set_timeline_ladder(int on);
void energy_set_force_all_active(int on);

void t7_reset_nonadvancing();
unsigned int t7_read_nonadvancing();
// C-D-14: details of the first particle whose step failed to advance the clock, for the abort
// message (Gadget-2's endrun(818) reports the same quantities, timestep.c:537-552).
void t7_read_first_offender(int *idx, float *tc, float *newEnd, float *dt, float *ac);
// C-D-14 fault injection: force a non-advancing step so the abort can be tested (GADGET_HIP_T7_INJECT=1).
void t7_set_inject_stall(int on);
void t31_set_force_geo(int on);   // T31 diagnostic
// Walk stack-overflow bail-outs: groups whose tree walk ran out of cell-list stack and therefore
// received NO force. Must be zero; nonzero means the step's forces are silently incomplete.
void walk_reset_bailouts();
unsigned int walk_read_bailouts();
// T28/C-B-15: mixed-softening forced-descent counters.
void t28_set_disable(int on);   // A/B: turn the mixed-softening rules off at runtime
void cb15_reset_counters();
void cb15_read_counters(unsigned int *forced, unsigned int *undefinedNode);
// C-D-13: count of growth events blocked by the SYNCHRONIZATION rule in the last compute_dt.
void cd13_set_enabled(int on);
void cd13_reset_blocked();
unsigned int cd13_read_blocked();
void cd13_read_detail(unsigned int *reached, unsigned int *grow, unsigned int *blocked);

void t4_counters_set(int enabled);
void t4_counters_reset();
void t4_counters_read(unsigned long long &primary, unsigned long long &prox,
                      unsigned long long &bjzero,  unsigned long long &reject);

#if defined(PMGRID) && !defined(PERIODIC)
// Isolated (non-periodic) TreePM counterpart to pm_periodic_setup_gravity_kernel above (LOG.md
// §34): uploads the same shared Rcut/Asmth/shortrange-table constants, but with no box size (no
// wraparound at all for an isolated system) -- Rcut/Asmth here should be derived from
// TotalMeshSize/GRID (rcut=RCUT*asmth, asmth=ASMTH*meshSize/(2*PMGRID)), matching
// pm_nonperiodic.c:125-126's own formula shape exactly. Gadget-2 never runs PM without a paired
// tree short-range correction (confirmed with the user directly) -- this closes that gap for the
// isolated case, mirroring the periodic tree-side wiring exactly.
void pm_isolated_setup_gravity_kernel(float rcut, float asmth);
#endif

#if defined(PMGRID) && defined(GADGET_HIP_HIGHRES)
// Phase 5 ticket 07 (PLAN.md): uploads the zoom (PLACEHIGHRESREGION) fine grid's own
// (Rcut[1],Asmth[1]) pair plus the runtime zoom-mask bitmask into the SAME __constant__ memory
// pm_periodic_setup_gravity_kernel/pm_isolated_setup_gravity_kernel use above (defined alongside
// them in dev_approximate_gravity_warp_new.cu -- same reasoning: hipMemcpyToSymbol needs
// same-translation-unit visibility). Unlike those two (called once at IC load), this is called
// every time the zoom region is (re)computed (octree::recomputeZoomRegion(), gpu_iterate.cpp) --
// Rcut[1]/Asmth[1] can change reactively during a run.
void pm_zoom_upload_rcut_asmth(float rcut1, float asmth1, unsigned int zoomMask);
#endif

// CIC (cloud-in-cell) mass assignment: deposits each particle's mass onto the 8 surrounding
// cells of a periodic gridSize^3 density grid, matching Gadget-2's own scheme exactly
// (Gadget-2.0.7/Gadget2/pm_periodic.c:255-360, verified in GADGET2_NOTES.md).
//
// d_density must already be zeroed by the caller (hipMemsetAsync) and sized
// gridSize*gridSize*gridPitch floats, where gridPitch = 2*(gridSize/2+1) -- the standard in-place
// real-to-complex FFT padding convention (matches Gadget-2's own PMGRID2, and rocFFT's in-place
// R2C layout requirement).
//
// Particle positions are assumed to already lie in a periodic box of side boxSize (any origin
// convention is fine as long as it's consistent with what the force-interpolation/tree-walk side
// uses later) -- wrapping happens at the CIC cell-index level (periodic modulo), matching
// Gadget-2's own approach of wrapping bin indices rather than pre-wrapping positions, so
// out-of-range positions still deposit correctly rather than needing a separate wrap pass.
void pm_cic_assign_mass(const float4 *d_bodies_pos, int n, float *d_density,
                         int gridSize, int gridPitch, float boxSize, hipStream_t stream);

// Gather counterpart of pm_cic_assign_mass: trilinearly samples d_grid at each particle's
// position using the identical 8-cell weights/periodic-wrap geometry (Gadget-2's own force
// read-out, pm_periodic.c:640-671). Writes (does not accumulate) d_out[i] for each particle --
// callers wanting all 3 force components loop this once per axis with a different d_grid/d_out,
// matching pm_finite_diff_force's own per-axis convention below.
void pm_cic_interpolate(const float4 *d_bodies_pos, int n, const float *d_grid, float *d_out,
                         int gridSize, int gridPitch, float boxSize, hipStream_t stream);

// 4-point (5-point-stencil, O(dx^4)) finite-difference force extraction along one axis, matching
// Gadget-2's exact formula (pm_periodic.c:590-627, GADGET2_NOTES.md) -- called once per axis
// (0=x,1=y,2=z) with d_force reused across calls, matching Gadget-2's own 2-grid-buffer approach
// rather than materializing 3 separate force grids (LOG.md §18's design decision). Unlike
// Gadget-2's own MPI-distributed version, this port holds the whole grid on one GPU, so periodic
// neighbor wraparound is a plain modulo here, no ghost-cell/patch machinery needed.
//
// The returned values are extraScale * gridScale * finite_difference(d_potential), gridScale =
// 1/(2*cellSize). extraScale defaults to 1 (matching every --pm-test-* caller so far, which
// checks the stencil in isolation) -- pm_compute_forces_periodic passes Gadget-2's own physical
// prefactor (fac = G/(pi*BoxSize)) here instead of folding it into the kernel unconditionally, so
// the stencil's own correctness (see --pm-test-force and GADGET2_NOTES.md) stays checkable
// independently of that trivial scalar.
void pm_finite_diff_force(const float *d_potential, float *d_force, int gridSize, int gridPitch,
                           int axis, float boxSize, hipStream_t stream, float extraScale = 1.0f);

// Elementwise in-place scale (d_buf[i] *= scale). Implemented in pm_cic.cu, which -- like this
// declaration -- is unconditional regardless of GADGET_HIP_PMGRID (runtime arguments, not
// compile-time constants, same reasoning as pm_finite_diff_force above). Used internally by
// pm_compute_forces_periodic to convert its raw CIC-interpolated grid potential into physical
// units (LOG.md §30), and by gpu_iterate.cpp's real per-step loop (Phase 5 ticket 08, PLAN.md) to
// apply gadgetParams.G to the fully-summed tree+PM force/potential in every --param build,
// including ones without GADGET_HIP_PMGRID -- which is exactly why this moved out of the
// `#ifdef PMGRID` block below (a real build failure this ticket's own regression sweep caught,
// not hypothetical): Gadget-2 applies `All.G` to the tree's own contribution independent of
// whether PMGRID is compiled in at all (gravtree.c:328), so gating this declaration behind
// PMGRID would have silently reintroduced the G=1 bug for exactly the builds it's absent from.
void pm_scale_buffer(float *d_buf, int n, float scale, hipStream_t stream);

// T23 masked variants -- see pm_cic.cu. `d_active` is the tree walk's own output mask, marking the
// particles whose bodies_acc1 it refreshed this iteration. Applying either of these to the whole
// buffer re-processes stale entries and compounds every iteration the active set is sparse.
void pm_scale_acc_masked(float4 *d_acc, int n, float scale, const int *d_active,
                         hipStream_t stream);
// T36: PM force kept in ID space so it survives between PM steps (see pm_cic.cu for why current
// sort order cannot be used). scatter at PM steps, gather-and-add every step for active particles.
// The three force components are tagged GFreeAcc at the point of storage: the PM solve runs with
// gravityConstant = 1.0f, so G is still owed, and every consumer has to say which G it applies.
// d_outPot is deliberately NOT tagged -- the potential's G travels with the t30 self-potential and
// comoving-normalisation coefficients in compute_energies(), which is a different and more tangled
// path than the force's single multiplication; tagging it would need that path converted too.
void pm_scatter_by_id(const unsigned long long *d_ids, const float *d_fx, const float *d_fy,
                      const float *d_fz, const float *d_pot,
                      GFreeAcc *d_outX, GFreeAcc *d_outY, GFreeAcc *d_outZ, float *d_outPot,
                      int n, hipStream_t stream);
void pm_add_by_id_masked(float4 *d_acc, const unsigned long long *d_ids,
                         const GFreeAcc *d_byIdX, const GFreeAcc *d_byIdY,
                         const GFreeAcc *d_byIdZ,
                         const float *d_byIdPot, int n, const int *d_active, hipStream_t stream);

void pm_add_force_to_acc_masked(float4 *d_acc, const float *d_fx, const float *d_fy,
                                const float *d_fz, const float *d_pot, int n,
                                const int *d_active, hipStream_t stream);

#ifdef PMGRID
// rocFFT plumbing + periodic Green's-function multiply. Declared only when PMGRID is compiled in
// (GADGET_HIP_PMGRID CMake option) -- the implementation (pm_fft.cpp) is only built in that case,
// so a non-PM build failing to *compile* a stray call is a clearer error than a link failure.
// void* handles rather than the real rocFFT types so this header doesn't need rocfft.h -- only
// pm_fft.cpp does.
struct PMFFTPlans
{
  void *forwardPlan = nullptr;   // rocfft_plan, real_forward, in-place, unnormalized
  void *inversePlan = nullptr;   // rocfft_plan, real_inverse, in-place, scaled by 1/gridSize^3
  void *execInfo    = nullptr;   // rocfft_execution_info, holds the HIP stream association
  int   gridSize    = 0;
};

// Call once per process before creating any plans / after the last plan is destroyed,
// respectively (wraps rocfft_setup()/rocfft_cleanup()).
void pm_fft_init();
void pm_fft_shutdown();

// Creates the forward+inverse plan pair for a cubic periodic PM grid of gridSize^3 cells, using
// the same padded layout pm_cic_assign_mass already writes (gridPitch = 2*(gridSize/2+1) floats
// along the fastest axis, reinterpreted in place as gridSize/2+1 complex elements after the
// forward transform). The inverse plan's scale factor is set to 1/gridSize^3 at creation time, so
// a forward+inverse round trip with no k-space modification in between reproduces the original
// real data (verified by --pm-test-fft, not just assumed).
PMFFTPlans pm_fft_create_plans(int gridSize);
void pm_fft_destroy_plans(PMFFTPlans &plans);

// Executes the forward/inverse transform in place on d_grid (gridSize*gridSize*gridPitch floats).
void pm_fft_forward(PMFFTPlans &plans, float *d_grid, hipStream_t stream);
void pm_fft_inverse(PMFFTPlans &plans, float *d_grid, hipStream_t stream);

// Periodic Green's-function multiply, applied to the hermitian-complex grid in place between the
// forward and inverse transforms (d_grid reinterpreted as gridSize*gridSize*(gridSize/2+1)
// float2s -- the exact in-place overlay of the same buffer pm_fft_forward wrote into). Implements
// Gadget-2's own formula exactly (pm_periodic.c:396-421, GADGET2_NOTES.md):
// smth(k) = -exp(-k^2 * asmth2) / k^2, asmth2 = (2*pi*Asmth/boxSize)^2, Asmth = ASMTH*boxSize/gridSize
// (ASMTH=1.25, Gadget-2's own compile-time constant, allvars.h:83). The k=0 mode is zeroed
// (removes the mean/DC density component, matching pm_periodic.c:431 -- periodic self-gravity of
// a nonzero mean density has no solution otherwise, the standard "Jeans swindle").
// applyCicDeconvolution selects whether the CIC-window deconvolution factor (sin(x)/x per axis,
// raised to the 4th power) is also applied -- correct when the density grid came from
// pm_cic_assign_mass, but must be OFF when testing the Green's-function/FFT plumbing against an
// analytically-seeded density mode that didn't go through CIC assignment (--pm-test-poisson).
void pm_greens_multiply_periodic(float *d_grid, int gridSize, float boxSize,
                                  bool applyCicDeconvolution, hipStream_t stream);

// Owns the grid buffers + FFT plans for a periodic PM solve at a fixed gridSize/boxSize, so a
// real simulation loop can call pm_compute_forces_periodic() once per PM timestep without
// re-allocating grids or re-creating rocFFT plans every time. Only 2 grid buffers total
// (d_potential, reused for density->potential in place, and d_forceScratch, reused across the 3
// axes) -- matches Gadget-2's own memory footprint (GADGET2_NOTES.md), not one grid per force
// component.
struct PMPeriodicSolver
{
  int gridSize    = 0;
  int gridPitch   = 0;
  float boxSize   = 0;
  PMFFTPlans plans;
  float *d_potential    = nullptr;
  float *d_forceScratch = nullptr;
};

PMPeriodicSolver pm_solver_create_periodic(int gridSize, float boxSize);
void pm_solver_destroy_periodic(PMPeriodicSolver &solver);

// The full periodic PM step: CIC-deposits d_bodies_pos (n particles, .w = mass) onto the density
// grid, solves for the potential (forward FFT -> Green's function WITH CIC deconvolution enabled,
// since real particles went through CIC assignment -> inverse FFT), then for each axis
// finite-differences the force (scaled by gravityConstant/(pi*boxSize), Gadget-2's own physical
// prefactor -- pm_periodic.c:206-210) and CIC-interpolates it into d_force_x/y/z[i] (plain writes,
// not float4, to avoid a separate scatter kernel -- matches every kernel in this file's own
// per-axis convention). gravityConstant defaults to 1 (this port's native unit system pre-Phase-5,
// same convention already used for split_node_grav_springel's tree-side MAC -- LOG.md §11).
// d_potential (optional, nullptr by default so every existing call site is unaffected): if
// non-null, also CIC-interpolates the solved (Green's-function-filtered) potential grid at each
// particle's position and scales it by the SAME physicalScale used for force (LOG.md §28's fix --
// physicalScale = gravityConstant*gridSize^3/(pi*boxSize) already equals the exact constant that
// converts the raw grid potential into physical units, independent of the finite-difference
// gridScale which is force-specific) writing the result (a plain per-unit-mass potential value,
// matching bodies_acc.w's own convention -- GADGET2_NOTES.md, mass is multiplied in only at the
// later host-side energy sum) into d_potential[i]. Added LOG.md §30, alongside a matching fix to
// the tree's own potential-suppression table (a separate, previously-undiscovered bug -- see that
// section).
void pm_compute_forces_periodic(PMPeriodicSolver &solver, const float4 *d_bodies_pos, int n,
                                 float *d_force_x, float *d_force_y, float *d_force_z,
                                 hipStream_t stream, float gravityConstant = 1.0f,
                                 float *d_potential_out = nullptr);

// Set once from main.cpp right after IC load (same place pm_periodic_setup_gravity_kernel is
// called), so gpu_iterate.cpp -- which has no other access to the run's boxSize/PMGRID -- can run
// PM: both the GADGET_HIP_DUMP_ACC diagnostic's one-shot force computation, and (since LOG.md §29)
// the real per-step PM long-range force added into the actual simulation loop.
void pm_set_dump_context(float boxSize, int gridSize);
bool pm_get_dump_context(float &boxSize, int &gridSize);

// (pm_scale_buffer moved above, out of this PMGRID-gated block -- see its own comment.)

// Accumulates PM's per-axis force (as produced by pm_compute_forces_periodic) directly into a
// float4 acceleration buffer's .xyz components. Used by gpu_iterate.cpp's real per-step loop to
// add PM's long-range contribution on top of the tree's already-computed short-range force in
// bodies_acc1.
void pm_add_force_to_acc(float4 *d_acc, const float *d_fx, const float *d_fy, const float *d_fz,
                          int n, hipStream_t stream);

// Accumulates PM's own potential (from pm_compute_forces_periodic's optional d_potential output,
// LOG.md §30) into a float4 acceleration buffer's .w component -- the counterpart to
// pm_add_force_to_acc for the .xyz components, closing the previously-flagged gap (LOG.md §29)
// where PM's force was added to the real simulation loop but its potential wasn't, leaving energy
// accounting short-range-only.
void pm_add_potential_to_acc(float4 *d_acc, const float *d_potential, int n, hipStream_t stream);

// Non-periodic (isolated/vacuum-boundary) PM long-range solver (PLAN.md Phase 4's deferred item).
// Coexists with the periodic solver above rather than replacing it -- both are compiled into
// every PMGRID build; which one a real run should use is an orthogonal, later decision (matching
// Gadget-2's own design, where periodicity and "use PM" are independent choices, GADGET2_NOTES.md).
//
// Builds the doubled-grid (GRID=2*gridSize) real-space Green's function kernel once, in
// dimensionless grid-relative units matching Gadget-2 exactly (pm_nonperiodic.c:321-362):
// kernel(r) = -(1-erfc(u))/r, u = 0.5*r/(ASMTH/GRID), with the analytic r=0 limit substituted.
void pm_build_isolated_kernel(float *d_kernel, int gridSize, int gridPitch, hipStream_t stream);

// Applies only the CIC deconvolution factor to the kernel's own (already-forward-transformed)
// FFT -- no analytic 1/k^2 factor (the real-space kernel values already encode the full Green's
// function) and no k=0/DC removal (an isolated, decaying kernel has no Jeans-swindle pathology).
void pm_isolated_kernel_deconvolve(float *d_kernelHat, int gridSize, hipStream_t stream);

// Ordinary complex multiply (density_hat *= kernel_hat, convolution theorem) -- the entire
// per-step "Green's function application" for the isolated case, in place on the density buffer.
void pm_complex_multiply(float *d_density, const float *d_kernelHat, int gridSize, hipStream_t stream);

// Owns the cached kernel FFT + grid buffers + FFT plans for an isolated PM solve at a fixed
// gridSize (the caller's nominal PMGRID resolution -- the actual FFT grid is GRID=2*gridSize
// internally, PLAN.md's "doubled grid" zero-padding trick) and meshSize (the PHYSICAL size of the
// full doubled grid -- particles must lie within [0, meshSize/2) in each axis, the occupied
// "first half," for correct isolated-boundary behavior; the second half is the zero-padding
// buffer the FFT convolution needs to avoid wrapping real structure back onto itself).
struct PMIsolatedSolver
{
  int gridSize    = 0;   // nominal (PMGRID) resolution; internal FFT grid is 2*gridSize
  int fftGridSize = 0;   // = 2*gridSize, the actual doubled FFT grid
  int gridPitch   = 0;   // padded real-space pitch for fftGridSize
  float meshSize  = 0;   // physical size of the full doubled grid
  PMFFTPlans plans;
  float *d_kernelHat     = nullptr; // cached, FFT'd, CIC-deconvolved kernel (complex, persistent)
  float *d_potential     = nullptr; // scratch: density in, potential out, reused every call
  float *d_forceScratch  = nullptr;
};

PMIsolatedSolver pm_solver_create_isolated(int gridSize, float meshSize);
void pm_solver_destroy_isolated(PMIsolatedSolver &solver);

// The full isolated PM step, mirroring pm_compute_forces_periodic's own signature and behavior
// exactly (CIC-deposit -> forward FFT -> complex-multiply by the cached kernel (not an analytic
// Green's function) -> inverse FFT -> finite-difference force + CIC-interpolate, optionally also
// interpolating+scaling the potential into d_potential_out). physicalScale here is NOT
// Gadget-2's own non-periodic "fac" formula copied verbatim -- see LOG.md for why that would
// repeat the exact FFT-normalization-convention mismatch already found and fixed for the periodic
// case (§27-28): this port's properly-normalized (1/N^3-on-inverse) rocFFT convention needs its
// own re-derived constant, validated against an exact isolated-point-mass Newtonian reference
// rather than trusted from Gadget-2's formula by inspection alone.
void pm_compute_forces_isolated(PMIsolatedSolver &solver, const float4 *d_bodies_pos, int n,
                                 float *d_force_x, float *d_force_y, float *d_force_z,
                                 hipStream_t stream, float gravityConstant = 1.0f,
                                 float *d_potential_out = nullptr);
#endif

#ifdef PMGRID
// Phase 4 (PLAN.md) tree+PM force-matching validation: runs the tree walk's own add_acc()
// (dev_approximate_gravity_warp_new.cu -- minimum-image wrap when PERIODIC, erfc
// shortrange_table suppression always, LOG.md §34) directly on n independent
// (target-at-origin, source) pairs, one per thread, with no tree build involved. d_sep[i] is the
// source's position relative to the target (any octant; wrapped internally the same way the real
// tree walk wraps it when PERIODIC), d_mass[i] its mass, eps2=0 (no softening, matching the
// ground-truth references this feeds -- Ewald for periodic, plain Newtonian for isolated -- which
// also assume point masses). d_outAcc[i].xyz is the resulting short-range acceleration. `d_type[i]`
// is the target's own Gadget-2 type (Ticket 07, PLAN.md) -- selects Rcut[0]/Asmth[0] vs.
// Rcut[1]/Asmth[1] exactly like the real tree walk; pass all-zero for every caller that isn't
// specifically exercising the zoom dual-grid split (--zoom-treepm-force-test).
void pm_test_add_acc_pair(const float3 *d_sep, const float *d_mass, const int *d_type, int n,
                           float4 *d_outAcc, hipStream_t stream);
#endif

// Phase 5 ticket 03 (PLAN.md): pure add_acc() unit test for Gadget-2's cubic-spline softening
// kernel (PHASE5_ZOOM_COMOVING_SPEC.md Sec 4.1), deliberately NOT gated by PMGRID (unlike
// pm_test_add_acc_pair above) -- reads add_acc()'s softening math in isolation, without the PM
// short-range erfc suppression (which only exists under PMGRID) mixing in. Run on a non-PMGRID
// build for a clean reading of the raw formula; a PMGRID build would additionally erfc-suppress
// the result, which is correct there but not what this specific unit test isolates.
//
// For test point i: target at the origin, source at d_sep[i] with mass d_mass[i]. The softening
// length used is fmaxf(d_h_i[i], d_h_j[i]) -- the exact same combination rule directAcc() applies
// between a target and a leaf-particle source of a (possibly) different type -- so passing equal
// values exercises the same-type case and differing values exercises the max-of-two-types case,
// both from PLAN.md Phase 5.5's ticket.
void softening_test_pair(const float3 *d_sep, const float *d_mass, const float *d_h_i,
                          const float *d_h_j, int n, float4 *d_outAcc, hipStream_t stream);
