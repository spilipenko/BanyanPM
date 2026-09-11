#pragma once
#include <vector>

// Phase 5 ticket 06 (PLAN.md): pure host region-sizing logic, deliberately kept free of any
// HIP/CUDA dependency (unlike pm_zoom.h, which declares the device-kernel-launching counterparts
// and includes this header) so it can be unit-tested standalone with plain g++ -- matching this
// project's own convention for gadget_softening.h/gadget_cosmology.h/gadget_driftfac.h. Uses a
// plain `Vec3f` instead of HIP's `float4` for exactly this reason; callers elsewhere in this port
// (which do have float4 available) do the trivial 3-float conversion at the call site.
//
// See pm_zoom.h for the full PLACEHIGHRESREGION mechanism citations (PHASE5_ZOOM_COMOVING_SPEC.md
// Sec 1) -- not repeated here to avoid the two headers drifting out of sync.

#ifdef GADGET_HIP_HIGHRES

struct Vec3f { float x, y, z; };

struct ZoomRegion
{
  float corner[3]     = {0, 0, 0}; // global coords of the fine grid's local (0,0,0)
  float totalMeshSize = 0.0f;      // "science" box side length (post-ENLARGEREGION, pre-doubling)
  float meshSize      = 0.0f;      // = 2*totalMeshSize, the doubled grid's physical size
  float asmth1        = 0.0f;      // ASMTH * totalMeshSize / (2*gridSize) -- ALWAYS doubled, see below
  float rcut1         = 0.0f;      // RCUT  * asmth1
  bool  valid         = false;
};

// pm_init_regionsize(), pm_nonperiodic.c:58-179. See pm_zoom.h for the full rationale (bounding
// box over high-res-typed particles only, ENLARGEREGION, the Rcut-buffer-enlarge special case,
// and why this port's buffer padding deliberately doesn't copy Gadget-2's own corner-shift
// formula). Returns false if no particle matches `highResMask`.
//
// `asmth1` is always computed as `ASMTH*totalMeshSize/(2*gridSize)` -- the zoom/fine grid is
// ALWAYS an isolated (non-periodic, zero-padded) FFT solve internally, regardless of whether the
// COARSE grid (whose own `asmth0`/`rcut0` are passed in here) is periodic or isolated. Verified
// directly against Gadget-2.0.7/Gadget2/pm_nonperiodic.c:29 (`#define GRID (2*PMGRID)`, not
// guarded by `#ifdef PERIODIC`) and its own `All.Asmth[1] = ASMTH*All.TotalMeshSize[1]/GRID`
// (line 123) -- matches this port's own pm_solve_zoom.cpp, whose `fftGridSize = 2*gridSize` is
// likewise unconditional. `asmth0` itself may or may not include this same doubling (periodic
// coarse grid: no doubling, `ASMTH*boxSize/gridSize`; isolated coarse grid: doubling,
// `ASMTH*meshSize0/(2*gridSize)`) -- that distinction lives entirely in the CALLER's own asmth0
// computation and is irrelevant here. `asmth1/asmth0` is used downstream as a plain ratio, not
// assumed to algebraically simplify to totalMeshSize/meshSize0 (an earlier, unverified version of
// this function computed asmth1 with a divisor chosen to match the coarse grid's own convention
// instead of always doubling it -- wrong for a periodic+zoom combination; caught and fixed during
// Ticket 06's own end-to-end zoom test, LOG.md).
bool pm_zoom_compute_region(const std::vector<Vec3f> &h_pos, const std::vector<int> &h_types,
                             unsigned int highResMask, double enlargeRegion, int gridSize,
                             double asmth0, double rcut0, ZoomRegion &region);

// "High-res zone may not cross a periodic box boundary" (Makefile/main.c's own documented
// modeling requirement -- Gadget-2 itself has no runtime assertion for this). Pass boxSize<=0 to
// skip (non-periodic zoom has no box to cross).
bool pm_zoom_region_fits_in_box(const ZoomRegion &region, float boxSize);

// True if any high-res-typed particle lies outside `region`'s own science box -- the reactive-
// recompute trigger (pm_nonperiodic.c:515-528, confirmed unconditional in source).
bool pm_zoom_region_out_of_range(const ZoomRegion &region, const std::vector<Vec3f> &h_pos,
                                  const std::vector<int> &h_types, unsigned int highResMask);

#endif // GADGET_HIP_HIGHRES
