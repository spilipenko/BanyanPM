#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "pm.h"

// Non-periodic (isolated/vacuum-boundary) PM long-range solver -- mirrors pm_solve.cpp's own
// periodic solver structure closely (create once, reuse every call), the only real differences
// being (1) the FFT grid is doubled (fftGridSize = 2*gridSize, PLAN.md's zero-padding
// requirement for a correct isolated boundary out of an inherently-periodic FFT) and (2) the
// Green's function is an ONCE-CACHED real-space kernel (pm_isolated.cu) rather than an
// analytic per-call k-space formula.

static void pm_solve_isolated_check_hip(hipError_t err, const char *what)
{
  if (err != hipSuccess)
  {
    fprintf(stderr, "HIP error in %s: %s\n", what, hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

PMIsolatedSolver pm_solver_create_isolated(int gridSize, float meshSize)
{
  PMIsolatedSolver s;
  s.gridSize    = gridSize;
  s.fftGridSize = 2 * gridSize;
  s.gridPitch   = 2 * (s.fftGridSize / 2 + 1);
  s.meshSize    = meshSize;
  s.plans       = pm_fft_create_plans(s.fftGridSize);

  const size_t gridElems = (size_t)s.fftGridSize * s.fftGridSize * s.gridPitch;
  pm_solve_isolated_check_hip(hipMalloc((void**)&s.d_kernelHat, gridElems * sizeof(float)),
                              "hipMalloc(d_kernelHat)");
  pm_solve_isolated_check_hip(hipMalloc((void**)&s.d_potential, gridElems * sizeof(float)),
                              "hipMalloc(d_potential)");
  pm_solve_isolated_check_hip(hipMalloc((void**)&s.d_forceScratch, gridElems * sizeof(float)),
                              "hipMalloc(d_forceScratch)");

  // Build the real-space kernel directly into d_kernelHat's buffer (reused as scratch before it
  // holds the FFT'd/deconvolved result -- same "one buffer, transformed in place" pattern the
  // periodic solver's own d_potential uses for density->potential).
  pm_build_isolated_kernel(s.d_kernelHat, s.fftGridSize, s.gridPitch, 0);
  pm_fft_forward(s.plans, s.d_kernelHat, 0);
  pm_isolated_kernel_deconvolve(s.d_kernelHat, s.fftGridSize, 0);
  pm_solve_isolated_check_hip(hipDeviceSynchronize(), "kernel build+FFT+deconvolve");

  return s;
}

void pm_solver_destroy_isolated(PMIsolatedSolver &s)
{
  pm_fft_destroy_plans(s.plans);
  if (s.d_kernelHat) hipFree(s.d_kernelHat);
  if (s.d_potential) hipFree(s.d_potential);
  if (s.d_forceScratch) hipFree(s.d_forceScratch);
  s.d_kernelHat = s.d_potential = s.d_forceScratch = nullptr;
}

void pm_compute_forces_isolated(PMIsolatedSolver &solver, const float4 *d_bodies_pos, int n,
                                 float *d_force_x, float *d_force_y, float *d_force_z,
                                 hipStream_t stream, float gravityConstant, float *d_potential_out)
{
  const int    G          = solver.fftGridSize;
  const size_t gridElems  = (size_t)G * G * solver.gridPitch;

  pm_solve_isolated_check_hip(hipMemsetAsync(solver.d_potential, 0, gridElems * sizeof(float), stream),
                              "hipMemsetAsync(d_potential)");
  // meshSize (not meshSize/2!) is the physical size CIC assignment needs -- it's the size of the
  // WHOLE doubled grid the caller's shifted/placed coordinates span; the "occupied first half"
  // convention is a caller contract (pm.h's PMIsolatedSolver doc comment), not something this
  // function enforces or needs to know about -- pm_cic_assign_mass just deposits wherever the
  // given positions land within [0, meshSize).
  pm_cic_assign_mass(d_bodies_pos, n, solver.d_potential, G, solver.gridPitch, solver.meshSize, stream);

  pm_fft_forward(solver.plans, solver.d_potential, stream);
  pm_complex_multiply(solver.d_potential, solver.d_kernelHat, G, stream);
  pm_fft_inverse(solver.plans, solver.d_potential, stream);

  // Physical scale: NOT Gadget-2's own non-periodic "fac" formula copied verbatim (see pm.h's
  // doc comment on this function, and LOG.md, for why that would repeat the exact
  // FFT-normalization mismatch already found and fixed for the periodic case). Gadget-2's own fac
  // (pm_nonperiodic.c:493-494) is itself a TWO-FACTOR product: a potential-conversion factor
  // (G/L^4 * (L/GRID)^3 = G/(L*GRID^3), L=meshSize) times a grid-derivative-conversion factor
  // (1/(2*L/GRID)) that pm_finite_diff_force below ALREADY applies internally via its own
  // cellSize=boxSize/gridSize computation -- matching exactly how the periodic solver's own
  // physicalScale (§28) only ever carried Gadget-2's FIRST factor, never their second, to avoid
  // double-applying the grid-derivative conversion. Multiplying by GRID^3 to compensate for this
  // port's different (properly-normalized) FFT convention, same reasoning as §28: physicalScale =
  // [G/(L*GRID^3)] * GRID^3 = G/L. (An initial attempt kept Gadget-2's second factor too, giving
  // G*GRID/(2*L^2) -- validated against --pm-test-isolated-force and found high by a factor of
  // exactly GRID/(2*L), confirming the double-counting before this fix.)
  const float physicalScale = gravityConstant / solver.meshSize;

  if (d_potential_out)
  {
    pm_cic_interpolate(d_bodies_pos, n, solver.d_potential, d_potential_out, G,
                        solver.gridPitch, solver.meshSize, stream);
    pm_scale_buffer(d_potential_out, n, physicalScale, stream);
  }

  float *const outs[3] = { d_force_x, d_force_y, d_force_z };
  for (int axis = 0; axis < 3; axis++)
  {
    pm_finite_diff_force(solver.d_potential, solver.d_forceScratch, G, solver.gridPitch, axis,
                          solver.meshSize, stream, physicalScale);
    pm_cic_interpolate(d_bodies_pos, n, solver.d_forceScratch, outs[axis], G, solver.gridPitch,
                        solver.meshSize, stream);
  }
}
