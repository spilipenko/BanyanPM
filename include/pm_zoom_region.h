#pragma once
#include <vector>
#include <cstddef>

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
  float asmth1        = 0.0f;      // ASMTH * cellSize, i.e. ASMTH*meshSize/fftGridSize (T50 3D)
  float rcut1         = 0.0f;      // RCUT  * asmth1
  // T50 item 3F: `corner` is inset below the data span by this much (2 fine cells, Gadget-2's own
  // pm_nonperiodic.c:119 `- 2.0005*TotalMeshSize/GRID`), so the 4-point finite-difference stencil
  // never has to reach outside the grid for a cell any particle gathers from. The DATA therefore
  // occupies [corner + cornerInset, corner + cornerInset + totalMeshSize]; keep that in mind
  // whenever `corner` is used as if it were the start of the data.
  float cornerInset   = 0.0f;
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
// T51: the periodic centre of the high-res species, and the shift that moves it to the box centre.
//
// Why this is not a centre of mass. In a periodic box the mean of the coordinates is meaningless for
// a clump straddling a boundary: particles at x=0.1 and x=L-0.1 are neighbours, but their mean is
// L/2, the far side of the box. The circular mean (mean of the unit vectors at angle 2*pi*x/L) fixes
// that but is biased for broad distributions and gives no bounding box.
//
// What this does instead is exact for the case that actually matters, and tells you when it is not
// applicable. Pick any one high-res particle as a reference; express every other high-res particle's
// coordinate as its MINIMUM-IMAGE offset from that reference; take min and max of those offsets. If
// the true occupied interval is shorter than L/2 then every member lies within L/2 of the reference
// along the shortest path, so each minimum-image offset IS the true signed offset, and min/max give
// the true interval -- wrap and all. The centre is then the midpoint of that interval, wrapped.
//
// The precondition is self-checking: if the resulting extent comes back >= L/2 on any axis, the
// method cannot be trusted (the species is spread over half the box or more, or the mask is wrong),
// and that is reported rather than papered over. A zoom region is by construction a small part of
// the box, so a failure here means the setup is not a zoom.
//
// `pos` is read with a stride so it can run straight over the float4 position array with no copy --
// the whole point of avoiding an N-element temporary (T50 item 3A). Returns false and sets `*errOut`
// to a static string on no-match or precondition failure.
bool pm_zoom_periodic_center(const float *pos, size_t posStride, const int *types, size_t n,
                             unsigned int highResMask, double boxSize,
                             double centerOut[3], double extentOut[3], const char **errOut);

// Translate every particle (not just the high-res ones -- a periodic box is translation invariant
// only if it all moves together) by `shift`, wrapping into [0, boxSize). Used both to apply the
// recentring at IC load and, with the negated shift, to undo it when writing a snapshot.
void pm_zoom_shift_positions(float *pos, size_t posStride, size_t n,
                             const double shift[3], double boxSize);

bool pm_zoom_region_fits_in_box(const ZoomRegion &region, float boxSize);

// True if any high-res-typed particle lies outside `region`'s own science box -- the reactive-
// recompute trigger (pm_nonperiodic.c:515-528, confirmed unconditional in source).
bool pm_zoom_region_out_of_range(const ZoomRegion &region, const std::vector<Vec3f> &h_pos,
                                  const std::vector<int> &h_types, unsigned int highResMask);

#endif // GADGET_HIP_HIGHRES
