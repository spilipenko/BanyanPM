#ifdef GADGET_HIP_HIGHRES
#include "pm_zoom_region.h"

#include <algorithm>
#include <limits>
#include <cstddef>

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

  // Asmth[1] = ASMTH * (full grid extent) / (full grid cell count) = ASMTH * cellSize, i.e. the
  // SAME "1.25 cells" convention the coarse grid uses -- verified against
  // Gadget-2.0.7/Gadget2/pm_nonperiodic.c, which needs three lines read together, not one:
  //
  //   :130  All.Asmth[1] = ASMTH * All.TotalMeshSize[1] / GRID;          // GRID = 2*PMGRID
  //   :112  All.TotalMeshSize[j] *= 2.001 * GRID / ((double)(GRID-2-8)); // <-- the DOUBLING lives here
  //   :142  All.UpperCorner[1][i] = All.Corner[1][i] + (GRID/2 - 1) * (All.TotalMeshSize[1]/GRID);
  //
  // Line 112 is the step that matters and the one an earlier version of this comment missed. By the
  // time :130 runs, `TotalMeshSize[1]` has ALREADY been multiplied by ~2.04 -- it is the extent of
  // the FULL doubled grid, not of the high-res data. Line 142 confirms it independently: the data
  // occupies only `GRID/2 - 1` of the GRID cells, each of size `TotalMeshSize[1]/GRID`. So GADGET's
  // Asmth[1] is 1.25 * cellSize, exactly like Asmth[0].
  //
  // This port's own names map onto GADGET's as:
  //     this function's `totalMeshSize`   = the DATA extent     (GADGET's `meshinner[1]`)
  //     `region.meshSize = 2*totalMeshSize`                     (GADGET's post-:112 TotalMeshSize[1])
  //     `fftGridSize = 2*gridSize`                              (GADGET's GRID)
  // and that mapping is confirmed independently by the CIC, whose `invCellSize = fftGridSize /
  // meshSize = gridSize / totalMeshSize` puts the data in indices [0, gridSize] of a 2*gridSize
  // grid -- GADGET's `GRID/2 - 1`, give or take its 2-cell finite-difference shift.
  //
  // THE BUG THIS REPLACES (T50 section 3D): the divisor was `2.0 * gridSize` applied to the DATA
  // extent, i.e. `1.25 * totalMeshSize / (2*gridSize)` = 0.625 * cellSize -- exactly HALF of
  // GADGET's value. It did not cancel anywhere, because pm_zoom.cu's own
  // pm_build_finegrid_kernel_kernel independently uses `asmthGrid = 1.25f/gridSize` in grid-fraction
  // units, which IS 1.25 * cellSize, i.e. correct. So the fine grid's Green's function used the
  // right Asmth[1] while (a) `asmthRatio = asmth1/asmth0` handed its coarse-side erfc term a
  // smoothing 2x too large, and (b) the tree's own high-res short-range cutoff (asmthfac_1, rcut2_1
  // via pm_zoom_upload_rcut_asmth) switched on at half the intended radius. The two halves of the
  // TreePM split disagreed for every high-res target.
  // T53 item H: a direct transcription of Gadget-2's own pm_nonperiodic.c:107-158, replacing this
  // port's earlier paraphrase. Verified against Gadget's own log for the zoom-ics set (PMGRID=128):
  // it reproduces totmeshsize=40.2472 and meshsize=0.157216 to the digit.
  //
  // The structural point the paraphrase missed: Gadget NEVER grows the data extent. The Rcut[0]
  // buffer goes into the FULL GRID and into the CORNER OFFSET, leaving `meshinner` -- the extent the
  // high-res particles actually occupy -- untouched. This port instead grew the data extent to
  // `extent + 2*rcut0` and then applied the :110 factor, which left the fine grid 9.1% coarser than
  // Gadget's on the same ICs (asmth1 0.178709 vs 0.196520). Both keep >= Rcut[0] of buffer, so
  // neither is unsafe; they simply disagree on the fine grid's resolution, and Gadget's is the
  // definition.
  //
  //   :109  meshinner[j] = TotalMeshSize[j];                    // the data extent, kept
  //   :110  TotalMeshSize[j] *= 2.001*GRID/(GRID-2-8);          // -> the FULL grid
  //   :118  Corner = Xmintot - 2.0005*TotalMeshSize/GRID;       // 2-cell finite-difference shift
  //   :135  if (2*TotalMeshSize[1]/GRID < Rcut[0]) {            // fine cell << coarse cutoff
  //   :137    TotalMeshSize[1] = 2*(meshinner + 2*Rcut[0])*GRID/(GRID-2);
  //   :141    Corner = Xmintot - 1.0001*Rcut[0];                // buffer moves INTO the corner
  //   :145    if (2*TotalMeshSize[1]/GRID > Rcut[0]) {          // (not reached in practice)
  //   :147      TotalMeshSize[1] = 2*(meshinner + 2*Rcut[0])*GRID/(GRID-10);
  //   :151      Corner = Xmintot - 1.0001*(Rcut[0] + 2*TotalMeshSize[1]/GRID); }
  //   :156    Asmth[1] = ASMTH*TotalMeshSize[1]/GRID; Rcut[1] = RCUT*Asmth[1]; }
  //
  // `meshinner` here is `totalMeshSize` (extent * ENLARGEREGION) and Gadget's GRID is `fftCells`.
  const int    fftCells  = 2 * gridSize;                 // Gadget's GRID
  const double meshinner = totalMeshSize;                // Gadget's meshinner[1] -- never grown

  double fullMesh = meshinner * 2.001 * (double) fftCells / (double) (fftCells - 2 - 8);
  double cell     = fullMesh / (double) fftCells;
  double inset    = 2.0005 * cell;

  if (2.0 * cell < rcut0)
  {
    fullMesh = 2.0 * (meshinner + 2.0 * rcut0) * (double) fftCells / (double) (fftCells - 2);
    cell     = fullMesh / (double) fftCells;
    inset    = 1.0001 * rcut0;
    if (2.0 * cell > rcut0)
    {
      fullMesh = 2.0 * (meshinner + 2.0 * rcut0) * (double) fftCells / (double) (fftCells - 10);
      cell     = fullMesh / (double) fftCells;
      inset    = 1.0001 * (rcut0 + 2.0 * cell);
    }
  }

  const double asmth1 = PM_ASMTH * cell;                 // ASMTH * cellSize, as :156/:130
  const double rcut1  = PM_RCUT  * asmth1;

  // Gadget's Corner is relative to Xmintot, the data region's own lower corner, which after the
  // symmetrisation above is center - meshinner/2.
  for (int a = 0; a < 3; ++a)
    region.corner[a] = (float)(center[a] - 0.5 * meshinner - inset);
  region.cornerInset   = (float) inset;
  region.totalMeshSize = (float) meshinner;
  region.meshSize      = (float) fullMesh;
  region.asmth1         = (float) asmth1;
  region.rcut1          = (float) rcut1;
  region.valid          = true;
  return true;
}

bool pm_zoom_periodic_center(const float *pos, size_t posStride, const int *types, size_t n,
                             unsigned int highResMask, double boxSize,
                             double centerOut[3], double extentOut[3], const char **errOut)
{
  if (errOut) *errOut = NULL;
  if (boxSize <= 0.0) { if (errOut) *errOut = "boxSize <= 0"; return false; }

  // Reference particle: the FIRST masked one, so the result is deterministic and independent of
  // iteration order or thread count. Any member works -- see the header for why.
  size_t ref = n;
  for (size_t i = 0; i < n; ++i)
    if ((1u << types[i]) & highResMask) { ref = i; break; }
  if (ref == n) { if (errOut) *errOut = "no particle matches the high-res mask"; return false; }

  const double x0[3] = { (double) pos[ref * posStride + 0],
                         (double) pos[ref * posStride + 1],
                         (double) pos[ref * posStride + 2] };
  double dmin[3] = { 0.0, 0.0, 0.0 };
  double dmax[3] = { 0.0, 0.0, 0.0 };

  const double halfBox = 0.5 * boxSize;
  for (size_t i = 0; i < n; ++i)
  {
    if (!((1u << types[i]) & highResMask)) continue;
    for (int a = 0; a < 3; ++a)
    {
      // Minimum image of (x_i - x0) into [-L/2, L/2).
      double d = (double) pos[i * posStride + a] - x0[a];
      while (d >= halfBox)  d -= boxSize;
      while (d < -halfBox)  d += boxSize;
      if (d < dmin[a]) dmin[a] = d;
      if (d > dmax[a]) dmax[a] = d;
    }
  }

  for (int a = 0; a < 3; ++a)
  {
    const double extent = dmax[a] - dmin[a];
    if (extent >= halfBox)
    {
      // Not a failure of the arithmetic -- a statement that the premise does not hold. Reported
      // rather than silently returning a centre that depends on which particle happened to be first.
      if (errOut) *errOut = "high-res species spans >= half the box on some axis: the "
                            "minimum-image centre is not well defined, so this is not a zoom "
                            "configuration (check --zoom-mask)";
      return false;
    }
    extentOut[a] = extent;
    double c = x0[a] + 0.5 * (dmin[a] + dmax[a]);
    while (c >= boxSize) c -= boxSize;
    while (c < 0.0)      c += boxSize;
    centerOut[a] = c;
  }
  return true;
}

void pm_zoom_shift_positions(float *pos, size_t posStride, size_t n,
                             const double shift[3], double boxSize)
{
  if (boxSize <= 0.0) return;
  for (size_t i = 0; i < n; ++i)
    for (int a = 0; a < 3; ++a)
    {
      double x = (double) pos[i * posStride + a] + shift[a];
      // shift is already wrapped into [0, boxSize) by the caller and x >= 0, so one conditional
      // subtraction suffices; the loop is belt-and-braces for a float position sitting exactly at
      // boxSize after rounding.
      while (x >= boxSize) x -= boxSize;
      while (x < 0.0)      x += boxSize;
      pos[i * posStride + a] = (float) x;
    }
}

bool pm_zoom_region_fits_in_box(const ZoomRegion &region, float boxSize)
{
  if (!region.valid || boxSize <= 0.0f)
    return true; // non-periodic zoom (or an invalid region, caught elsewhere) -- no box to cross
  // The DATA span is what must stay inside the box: a high-res particle outside it is silently
  // dropped by the clipped CIC. The padding half beyond it may hang past the box edge harmlessly --
  // every particle coordinate is wrapped into [0, boxSize), so those cells simply receive nothing,
  // which is exactly the zero padding the isolated solve wants.
  for (int a = 0; a < 3; ++a)
  {
    const float dataLo = region.corner[a] + region.cornerInset;
    if (dataLo < 0.0f || dataLo + region.totalMeshSize > boxSize)
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
    {
      const float dataLo = region.corner[a] + region.cornerInset;
      if (c[a] < dataLo || c[a] > dataLo + region.totalMeshSize)
        return true;
    }
  }
  return false;
}
#endif // GADGET_HIP_HIGHRES
