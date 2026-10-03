#ifdef GADGET_HIP_HIGHRES
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include "pm_zoom.h"

// Phase 5 ticket 06 (PLAN.md): the fine ("grid 1") zoom PM solver. Mirrors
// pm_solve_isolated.cpp's own PMIsolatedSolver create/compute structure almost exactly -- the
// only real differences are (1) the cached kernel is built with pm_build_finegrid_kernel()'s
// differential formula instead of pm_build_isolated_kernel()'s standalone vacuum one, and (2) mass
// assignment/force readback use the CLIPPED (not periodic-wrap) CIC variants, shifted by the
// region's own `corner`, since this grid covers only a small sub-volume of the full box.

// T50 section 9: the same synced sub-phase timing pm_solve.cpp applies to the COARSE solver, so the
// fine grid's cost can be attributed instead of inferred. Identical semantics on purpose --
// hipStreamSynchronize BEFORE each timestamp, so these are kernel-completion times and not launch
// times. Without this, the fine grid's 18-32 s was a single opaque number and I attributed it to CIC
// atomic contention by analogy with the coarse grid, which is exactly the kind of inference that
// needs a measurement: the fine grid's cell is SMALLER than the coarse grid's, so its scatter should
// contend LESS, not more.
static double pm_zoom_sub_now()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#define PMZSUB(label) do { if (pmSubTime) { hipStreamSynchronize(stream); \
    const double t_ = pm_zoom_sub_now(); \
    fprintf(stderr, "[PMZSUB] %-24s %8.4f s\n", label, t_ - pmSubT); pmSubT = t_; } } while (0)

static void pm_solve_zoom_check_hip(hipError_t err, const char *what)
{
  if (err != hipSuccess)
  {
    fprintf(stderr, "HIP error in %s: %s\n", what, hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

PMIsolatedSolver pm_solver_create_finegrid(int gridSize, float meshSize, float asmthRatio)
{
  PMIsolatedSolver s;
  s.gridSize    = gridSize;
  s.fftGridSize = 2 * gridSize;
  s.gridPitch   = 2 * (s.fftGridSize / 2 + 1);
  s.meshSize    = meshSize;
  s.plans       = pm_fft_create_plans(s.fftGridSize);

  const size_t gridElems = (size_t)s.fftGridSize * s.fftGridSize * s.gridPitch;
  pm_solve_zoom_check_hip(hipMalloc((void**)&s.d_kernelHat, gridElems * sizeof(float)),
                           "hipMalloc(d_kernelHat)");
  pm_solve_zoom_check_hip(hipMalloc((void**)&s.d_potential, gridElems * sizeof(float)),
                           "hipMalloc(d_potential)");
  pm_solve_zoom_check_hip(hipMalloc((void**)&s.d_forceScratch, gridElems * sizeof(float)),
                           "hipMalloc(d_forceScratch)");

  // T50 item B: pm_build_finegrid_kernel writes k in [0, fftGridSize) but the row stride is
  // gridPitch = 2*(fftGridSize/2+1), so the 2 padding floats per row of a fresh hipMalloc are never
  // written. Benign today -- the in-place R2C forward FFT reads only the first fftGridSize reals --
  // but it is an uninitialised read waiting for anyone who changes the FFT layout or adds a pass
  // over the raw buffer. Zero it once at create; it costs nothing on a path that runs per region.
  pm_solve_zoom_check_hip(hipMemset(s.d_kernelHat, 0, gridElems * sizeof(float)),
                           "hipMemset(d_kernelHat)");
  pm_build_finegrid_kernel(s.d_kernelHat, s.fftGridSize, s.gridPitch, asmthRatio, 0);
  pm_fft_forward(s.plans, s.d_kernelHat, 0);
  pm_isolated_kernel_deconvolve(s.d_kernelHat, s.fftGridSize, 0);
  pm_solve_zoom_check_hip(hipDeviceSynchronize(), "kernel build+FFT+deconvolve");

  return s;
}

void pm_compute_forces_finegrid(PMIsolatedSolver &solver, const ZoomRegion &region,
                                 const float4 *d_bodies_pos, int n, float *d_force_x,
                                 float *d_force_y, float *d_force_z, hipStream_t stream,
                                 float gravityConstant, float *d_potential_out)
{
  static const bool pmSubTime = (getenv("GADGET_HIP_PHASE_TIME") != NULL);
  double pmSubT = 0.0;
  if (pmSubTime) { hipStreamSynchronize(stream); pmSubT = pm_zoom_sub_now(); }

  const int    G          = solver.fftGridSize;
  const size_t gridElems  = (size_t)G * G * solver.gridPitch;
  const float3 corner     = make_float3(region.corner[0], region.corner[1], region.corner[2]);

  pm_solve_zoom_check_hip(hipMemsetAsync(solver.d_potential, 0, gridElems * sizeof(float), stream),
                           "hipMemsetAsync(d_potential)");
  PMZSUB("zoom memset");
  // Unfiltered by type (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.1's first effect) -- any particle
  // physically inside the fine grid's box deposits mass, regardless of type; the CLIPPED CIC
  // variant itself is what confines this to particles actually inside the box (silently skipping
  // everyone else), matching Gadget-2's own explicit bounds check.
  // T50 item G: Gadget's own deposit window, (GRID/2 - 1) cells from the corner
  // (pm_nonperiodic.c:120). Everything beyond is zero padding and must stay empty, or the doubled
  // grid stops computing an isolated convolution.
  const int depositCells = G / 2 - 1;
  pm_cic_assign_mass_clipped(d_bodies_pos, n, solver.d_potential, G, solver.gridPitch,
                              solver.meshSize, corner, depositCells, stream);
  PMZSUB("zoom cic_scatter");

  pm_fft_forward(solver.plans, solver.d_potential, stream);
  PMZSUB("zoom fft_forward");
  pm_complex_multiply(solver.d_potential, solver.d_kernelHat, G, stream);
  PMZSUB("zoom kernel_multiply");
  pm_fft_inverse(solver.plans, solver.d_potential, stream);
  PMZSUB("zoom fft_inverse");

  // Same physicalScale derivation as pm_compute_forces_isolated (pm_solve_isolated.cpp) --
  // independent of the kernel's own real-space VALUES (the differential formula here vs. the
  // standalone vacuum one there), since it only corrects for this port's FFT-normalization and
  // CIC-window conventions, which are identical for both grids.
  const float physicalScale = gravityConstant / solver.meshSize;

  if (d_potential_out)
  {
    pm_cic_interpolate_clipped(d_bodies_pos, n, solver.d_potential, d_potential_out, G,
                                solver.gridPitch, solver.meshSize, corner, depositCells, stream);
    pm_scale_buffer(d_potential_out, n, physicalScale, stream);
    PMZSUB("zoom cic_gather_pot");
  }

  float *const outs[3] = { d_force_x, d_force_y, d_force_z };
  for (int axis = 0; axis < 3; axis++)
  {
    pm_finite_diff_force(solver.d_potential, solver.d_forceScratch, G, solver.gridPitch, axis,
                          solver.meshSize, stream, physicalScale);
    PMZSUB("zoom finite_diff/axis");
    pm_cic_interpolate_clipped(d_bodies_pos, n, solver.d_forceScratch, outs[axis], G,
                                solver.gridPitch, solver.meshSize, corner, depositCells, stream);
    PMZSUB("zoom cic_gather/axis");
  }
}
#endif // GADGET_HIP_HIGHRES
