#pragma once
#include <hip/hip_runtime.h>
#include "pm.h"
#include "pm_zoom_region.h"

// Phase 5 ticket 06 (PLAN.md): PLACEHIGHRESREGION zoom dual-grid PM -- the fine ("grid 1") solve
// layered on top of the existing coarse (periodic or isolated) solver from Phase 4. Full
// mechanism traced in PHASE5_ZOOM_COMOVING_SPEC.md Sec 1; every citation below refers to that.
// `ZoomRegion`/`pm_zoom_compute_region()`/etc. live in pm_zoom_region.h (no HIP dependency, so
// that piece stays unit-testable with plain g++); this header adds the device-kernel-launching
// counterparts that DO need HIP types.
//
// Runtime (not compile-time) config, a deliberate documented divergence from upstream: real
// Gadget-2 bakes the high-res particle-type bitmask and ENLARGEREGION into the Makefile
// (`-DPLACEHIGHRESREGION=<mask>`, `-DENLARGEREGION=<factor>`) -- this port takes them as ordinary
// runtime settings (`--zoom-mask`/`--zoom-enlarge`) instead, so one build serves many different
// `.param`-driven zoom configurations. `GADGET_HIP_HIGHRES` (the CMake option gating whether this
// code is compiled in at all) still exists as a genuine build-time choice, matching PERIODIC's own
// precedent, since it changes the tree kernel's constant-memory layout (Ticket 07).

#ifdef GADGET_HIP_HIGHRES

// Builds the fine grid's DIFFERENTIAL Green's-function kernel -- NOT a copy of
// pm_build_isolated_kernel's standalone vacuum kernel. Telescopes the coarse-grid long-range
// solution down to the fine grid's own smoothing scale: `fac = erfc(u*asmthRatio) - erfc(u)`,
// `asmthRatio = Asmth[1]/Asmth[0]`, computed as a plain ratio of the two grids' own (independently
// correct, not assumed to algebraically cancel) asmth values -- see pm_zoom_region.h's doc comment
// on `pm_zoom_compute_region()` for why "the factors cancel" is only true when the coarse grid is
// ALSO isolated, and wrong when it's periodic (verified against Gadget-2.0.7/Gadget2/
// pm_periodic.c:60 vs. pm_nonperiodic.c:29,123 -- PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.4's original
// "cancels in the ratio" framing didn't check this). Same grid-relative `u` convention as
// pm_build_isolated_kernel. `r==0` uses the analytic limit (`fac = 1 - asmthRatio`), not a runtime
// branch-free hack.
void pm_build_finegrid_kernel(float *d_kernel, int gridSize, int gridPitch, float asmthRatio,
                               hipStream_t stream);

// CIC mass assignment/force readback variants for the fine grid: identical trilinear stencil math
// to pm_cic_assign_mass/pm_cic_interpolate, but CLIP (skip entirely) any particle whose shifted
// local coordinate falls outside the valid [0,gridSize) cell range on any axis, rather than
// periodically wrapping. This is a real, necessary correctness difference, not a style choice: the
// fine grid is a small sub-region of a much larger box, so a particle far outside it must not
// alias its mass onto some essentially-random cell via modulo wraparound (which
// pm_cic_assign_mass's own periodic-wrap convention would otherwise do) -- matches Gadget-2's own
// explicit bounds check before depositing (pm_nonperiodic.c:579-584), not an approximation.
// `corner` shifts global coordinates into the grid's own local frame internally (`local = global -
// corner`) -- callers pass the SAME global-coordinate arrays used everywhere else, no separate
// shifted-position buffer needed.
// T50 item G: `depositCells` bounds the deposit to the INNER half of the doubled grid, which is what
// keeps the outer half genuine zero padding and therefore what makes the doubled-grid FFT an ISOLATED
// convolution rather than a periodic one. Gadget uses (GRID/2 - 1) cells
// (pm_nonperiodic.c:120 UpperCorner, enforced at :579-583 by skipping everything outside). Passing
// `gridSize` here instead -- the ARRAY bound -- is what let surrounding mass contaminate the padding.
void pm_cic_assign_mass_clipped(const float4 *d_bodies_pos, int n, float *d_density, int gridSize,
                                 int gridPitch, float cellSize, float3 corner, int depositCells,
                                 hipStream_t stream);
// Same window on the gather: outside the inner half the solved field is wraparound-contaminated, and
// Gadget's own readback (pm_nonperiodic.c:751-753) applies the identical bound. A particle outside it
// gets 0, which is what Gadget's "skip" amounts to.
void pm_cic_interpolate_clipped(const float4 *d_bodies_pos, int n, const float *d_grid,
                                 float *d_out, int gridSize, int gridPitch, float cellSize,
                                 float3 corner, int depositCells, hipStream_t stream);

// Creates a fine-grid solver: same PMIsolatedSolver struct as the standalone isolated case (pm.h),
// built with pm_build_finegrid_kernel() instead of pm_build_isolated_kernel().
PMIsolatedSolver pm_solver_create_finegrid(int gridSize, float meshSize, float asmthRatio);

// The full fine-grid PM step -- mirrors pm_compute_forces_isolated()'s structure exactly (CIC
// deposit -> FFT -> complex-multiply by the cached differential kernel -> inverse FFT ->
// finite-difference + CIC-interpolate), but using the CLIPPED CIC variants above (mass assignment
// by spatial location, unfiltered by type -- PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.1 -- so `d_bodies_pos`
// should be the FULL particle set, not just high-res-typed ones) and `region.corner` for the
// coordinate shift. Force/potential outputs are still per-particle (n entries) for every particle
// passed in -- the caller (gpu_iterate.cpp) is responsible for the type-filtered *readback*
// (Sec 1.1's second effect), via pm_add_masked_force_to_acc() below, not this function.
void pm_compute_forces_finegrid(PMIsolatedSolver &solver, const ZoomRegion &region,
                                 const float4 *d_bodies_pos, int n, float *d_force_x,
                                 float *d_force_y, float *d_force_z, hipStream_t stream,
                                 float gravityConstant = 1.0f, float *d_potential_out = nullptr);

// Type-filtered counterpart to pm_add_force_to_acc()/pm_add_potential_to_acc(): only accumulates
// for particles whose type bit is set in `highResMask` (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.1's
// second effect -- force/potential readback IS filtered by type, unlike mass assignment).
// `d_types` is a device-resident copy of the per-particle type array (tree.bodies_type is
// host-only elsewhere in this port -- ticket 06 is the first consumer needing it on the device).
// T58: add the fine mesh's force and potential into the coarse mesh's own output arrays, for the
// zoom-mask types only, so that the two meshes become one long-range force from there on -- which
// is how Gadget does it (both solves `+=` into GravPM) and is what puts the fine mesh on the PM
// cadence together with every other consumer of the long-range force.
void pm_accumulate_masked_longrange(float *d_fx, float *d_fy, float *d_fz, float *d_pot,
                                     const float *d_zx, const float *d_zy, const float *d_zz,
                                     const float *d_zpot, const int *d_types,
                                     unsigned int highResMask, int n, hipStream_t stream);

void pm_add_masked_force_to_acc(float4 *d_acc, const float *d_fx, const float *d_fy,
                                 const float *d_fz, const int *d_types, unsigned int highResMask,
                                 int n, hipStream_t stream);
// T50 section 3A: "has any high-res particle left the current region?" answered on the DEVICE.
// This replaces a per-rebuild-step host scan that copied all N positions device->host (1.84 GB at
// 115M particles), built an N-element std::vector<Vec3f> (1.38 GB, reallocated every step), and then
// walked it single-threaded -- measured at 0.58 s/step steady state and 3.74 s on the first step,
// for a question that is one masked reduction. Returns true if ANY particle whose type bit is set in
// `highResMask` lies outside [corner, corner+totalMeshSize] on any axis, matching
// pm_zoom_region_out_of_range()'s own inclusive-boundary convention exactly.
bool pm_zoom_any_out_of_range(const float4 *d_bodies_pos, const int *d_types,
                              unsigned int highResMask, float3 corner, float totalMeshSize,
                              int n, hipStream_t stream);

void pm_add_masked_potential_to_acc(float4 *d_acc, const float *d_potential, const int *d_types,
                                     unsigned int highResMask, int n, hipStream_t stream);

#endif // GADGET_HIP_HIGHRES
