#ifdef GADGET_HIP_HIGHRES
#include "pm_zoom_region.h"

#include <algorithm>
#include <limits>

// Phase 5 ticket 06 (PLAN.md): pure host region-sizing logic -- see pm_zoom_region.h/pm_zoom.h for
// the full rationale. Kept in its own translation unit, no HIP dependency, so it can be
// unit-tested standalone with plain g++, matching gadget_softening.cpp/gadget_cosmology.cpp/
// gadget_driftfac.cpp's own convention.

namespace {
// Gadget-2's own compile-time short/long-range split constants (allvars.h:82-89), duplicated here
// rather than shared across translation units -- same convention CUDAkernels/pm_isolated.cu
// already uses for the identical constants.
constexpr double PM_ASMTH = 1.25;
constexpr double PM_RCUT  = 4.5;
}

bool pm_zoom_compute_region(const std::vector<Vec3f> &h_pos, const std::vector<int> &h_types,
                             unsigned int highResMask, double enlargeRegion, int gridSize,
                             double asmth0, double rcut0, ZoomRegion &region)
{
  region = ZoomRegion();

  double xmin[3] = {  std::numeric_limits<double>::max(),  std::numeric_limits<double>::max(),  std::numeric_limits<double>::max() };
  double xmax[3] = { -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max() };
  bool any = false;

  const size_t n = h_pos.size();
  for (size_t i = 0; i < n; ++i)
  {
    const int t = h_types[i];
    if (!((1u << t) & highResMask)) continue;
    any = true;
    const double c[3] = { h_pos[i].x, h_pos[i].y, h_pos[i].z };
    for (int a = 0; a < 3; ++a)
    {
      xmin[a] = std::min(xmin[a], c[a]);
      xmax[a] = std::max(xmax[a], c[a]);
    }
  }

  if (!any)
    return false; // no particle matches the mask -- nothing to build a region around

  // pm_init_regionsize(), pm_nonperiodic.c:90-102: a single scalar cube side length (the largest
  // of the three axis extents), re-centered on the bounding box's own geometric midpoint -- NOT
  // mass-weighted, matches source exactly.
  double extent = 0.0;
  double center[3];
  for (int a = 0; a < 3; ++a)
  {
    extent = std::max(extent, xmax[a] - xmin[a]);
    center[a] = 0.5 * (xmin[a] + xmax[a]);
  }

  // ENLARGEREGION applied here and only here (pm_nonperiodic.c:93-94).
  double totalMeshSize = extent * enlargeRegion;

  // Asmth[1] ALWAYS uses the DOUBLED divisor 2*gridSize, unconditionally -- verified directly
  // against Gadget-2.0.7/Gadget2/pm_nonperiodic.c:29 (`#define GRID (2*PMGRID)`, not guarded by
  // `#ifdef PERIODIC`) and its own `All.Asmth[1] = ASMTH*All.TotalMeshSize[1]/GRID` (line 123):
  // the zoom/fine grid is always an isolated (non-periodic, zero-padded) FFT solve internally,
  // regardless of whether the COARSE grid is periodic or isolated -- matching this port's own
  // pm_solve_zoom.cpp, whose `fftGridSize = 2*gridSize` is likewise unconditional. The caller's own
  // `asmth0` may or may not include this same doubling (periodic coarse: no doubling,
  // pm_periodic.c:60; isolated coarse: doubling, pm_nonperiodic.c:123) -- irrelevant here, since
  // this function only computes the fine grid's own asmth1, and `asmth1/asmth0` is used as a plain
  // ratio downstream, not assumed to algebraically simplify to totalMeshSize/meshSize0 (an earlier,
  // unverified version of this code and its doc comments wrongly assumed it always did, which
  // silently used the coarse grid's own -- sometimes undoubled -- divisor for asmth1 instead).
  const double asmthDivisor = 2.0 * gridSize;
  auto asmthFor = [&](double tms) { return PM_ASMTH * tms / asmthDivisor; };
  auto rcutFor  = [&](double asmth) { return PM_RCUT * asmth; };

  // pm_nonperiodic.c:135-158: ensure the fine grid extends with at least Rcut[0]-worth of buffer
  // beyond the natural high-res bounding box on every side, so the coarse grid's own short-range
  // cutoff radius is fully contained inside the fine grid -- otherwise the two grids' force fields
  // wouldn't hand off correctly at the region boundary. Real Gadget-2's own formula
  // (`TotalMeshSize[1] = 2*(meshinner[1] + 2*Rcut[0])*GRID/(GRID-2)`, with further corner-shift/
  // finite-difference fudge factors) is tied to its own explicit corner-shifted, slack-cell grid
  // layout, which doesn't transfer to this port's simpler doubled-grid convention (`region.corner
  // = center - 0.5*totalMeshSize`, no separate slack-cell bookkeeping).
  //
  // An earlier version of this function instead enlarged `totalMeshSize` MULTIPLICATIVELY until
  // `rcut1 >= rcut0` (i.e., until the fine grid's own resolution matched the coarse grid's) --
  // this looked like a reasonable paraphrase of "enlarge for a Rcut[0] buffer" but is a
  // fundamentally different, much more aggressive correction: since asmth1/asmth0 (both using the
  // same PMGRID and the same doubled divisor whenever the coarse grid is ALSO isolated) reduces to
  // totalMeshSize/meshSize0, requiring rcut1>=rcut0 forces totalMeshSize >= meshSize0 UNCONDITIONALLY
  // -- meaning the fine grid could never be smaller (finer) than the entire coarse grid, defeating
  // the whole point of "zooming in." Caught during Ticket 06's own end-to-end zoom test (a tiny
  // high-res clump ballooned to match the full coarse mesh size instead of staying small); fixed by
  // switching to this additive buffer-pad, which only grows the region by O(rcut0) -- a genuinely
  // small quantity relative to the coarse grid's own physical extent in any realistic setup --
  // instead of O(meshSize0).
  const double neededMeshSize = extent + 2.0 * rcut0;
  if (neededMeshSize > totalMeshSize)
    totalMeshSize = neededMeshSize;

  const double asmth1 = asmthFor(totalMeshSize);
  const double rcut1  = rcutFor(asmth1);

  region.corner[0] = (float)(center[0] - 0.5 * totalMeshSize);
  region.corner[1] = (float)(center[1] - 0.5 * totalMeshSize);
  region.corner[2] = (float)(center[2] - 0.5 * totalMeshSize);
  region.totalMeshSize = (float) totalMeshSize;
  region.meshSize       = (float) (2.0 * totalMeshSize); // doubled-grid convention, pm.h
  region.asmth1         = (float) asmth1;
  region.rcut1          = (float) rcut1;
  region.valid          = true;
  return true;
}

bool pm_zoom_region_fits_in_box(const ZoomRegion &region, float boxSize)
{
  if (!region.valid || boxSize <= 0.0f)
    return true; // non-periodic zoom (or an invalid region, caught elsewhere) -- no box to cross
  for (int a = 0; a < 3; ++a)
  {
    if (region.corner[a] < 0.0f || region.corner[a] + region.totalMeshSize > boxSize)
      return false;
  }
  return true;
}

bool pm_zoom_region_out_of_range(const ZoomRegion &region, const std::vector<Vec3f> &h_pos,
                                  const std::vector<int> &h_types, unsigned int highResMask)
{
  if (!region.valid) return true;
  const size_t n = h_pos.size();
  for (size_t i = 0; i < n; ++i)
  {
    if (!((1u << h_types[i]) & highResMask)) continue;
    const float c[3] = { h_pos[i].x, h_pos[i].y, h_pos[i].z };
    for (int a = 0; a < 3; ++a)
      if (c[a] < region.corner[a] || c[a] > region.corner[a] + region.totalMeshSize)
        return true;
  }
  return false;
}
#endif // GADGET_HIP_HIGHRES
