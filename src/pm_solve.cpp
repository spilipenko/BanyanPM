#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "pm.h"
#include <ctime>

// T36 sub-phase timing, gated on the same GADGET_HIP_PHASE_TIME as the step-level timers in
// gpu_iterate.cpp and using the same convention: sync the stream on both sides so a sub-phase
// measures device work rather than kernel-launch latency. Zero cost when the variable is unset.
static double pm_sub_now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#define PMSUB(label) do { if (pmSubTime) { hipStreamSynchronize(stream); \
    const double t_ = pm_sub_now(); \
    fprintf(stderr, "[PMSUB] %-24s %8.4f s\n", label, t_ - pmSubT); pmSubT = t_; } } while (0)


static void pm_solve_check_hip(hipError_t err, const char *what)
{
  if (err != hipSuccess)
  {
    fprintf(stderr, "HIP error in %s: %s\n", what, hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

PMPeriodicSolver pm_solver_create_periodic(int gridSize, float boxSize)
{
  PMPeriodicSolver s;
  s.gridSize  = gridSize;
  s.gridPitch = 2 * (gridSize / 2 + 1);
  s.boxSize   = boxSize;
  s.plans     = pm_fft_create_plans(gridSize);

  const size_t gridElems = (size_t)gridSize * gridSize * s.gridPitch;
  pm_solve_check_hip(hipMalloc((void**)&s.d_potential, gridElems * sizeof(float)),
                      "hipMalloc(d_potential)");
  pm_solve_check_hip(hipMalloc((void**)&s.d_forceScratch, gridElems * sizeof(float)),
                      "hipMalloc(d_forceScratch)");
  return s;
}

void pm_solver_destroy_periodic(PMPeriodicSolver &s)
{
  pm_fft_destroy_plans(s.plans);
  if (s.d_potential) hipFree(s.d_potential);
  if (s.d_forceScratch) hipFree(s.d_forceScratch);
  s.d_potential = s.d_forceScratch = nullptr;
}

void pm_compute_forces_periodic(PMPeriodicSolver &solver, const float4 *d_bodies_pos, int n,
                                 float *d_force_x, float *d_force_y, float *d_force_z,
                                 hipStream_t stream, float gravityConstant, float *d_potential_out)
{
  static const bool pmSubTime = (getenv("GADGET_HIP_PHASE_TIME") != NULL);
  double pmSubT = 0.0;
  if (pmSubTime) { hipStreamSynchronize(stream); pmSubT = pm_sub_now(); }

  const size_t gridElems = (size_t)solver.gridSize * solver.gridSize * solver.gridPitch;

  pm_solve_check_hip(hipMemsetAsync(solver.d_potential, 0, gridElems * sizeof(float), stream),
                      "hipMemsetAsync(d_potential)");
  pm_cic_assign_mass(d_bodies_pos, n, solver.d_potential, solver.gridSize, solver.gridPitch,
                      solver.boxSize, stream);

  PMSUB("cic_scatter+memset");
  pm_fft_forward(solver.plans, solver.d_potential, stream);
  PMSUB("fft_forward");
  // applyCicDeconvolution=true: unlike --pm-test-poisson's analytically-seeded density, real
  // particles here went through pm_cic_assign_mass, so the CIC window's smoothing must be
  // compensated for (GADGET2_NOTES.md).
  pm_greens_multiply_periodic(solver.d_potential, solver.gridSize, solver.boxSize,
                               /*applyCicDeconvolution=*/true, stream);
  PMSUB("greens_multiply");
  pm_fft_inverse(solver.plans, solver.d_potential, stream);
  PMSUB("fft_inverse");

  // Gadget-2's own physical prefactor (pm_periodic.c:206-210), kept separate from the stencil
  // kernel itself (pm.h's doc comment on pm_finite_diff_force has the reasoning). Gadget-2's own
  // fac=G/(pi*BoxSize) is calibrated for its FFTW backend's convention of UNNORMALIZED forward
  // AND inverse transforms (verified by grep: no 1/PMGRID^3 division appears anywhere in
  // pm_periodic.c) -- a full forward+inverse round trip in Gadget-2's own code reproduces
  // PMGRID^3 times the original, not the original itself. This port's rocFFT-based
  // pm_fft_create_plans() instead bakes an explicit 1/gridSize^3 scale into the INVERSE plan (a
  // deliberate choice so a bare round-trip reproduces the input exactly -- validated by
  // --pm-test-fft, LOG.md §20), which is a different but self-consistent convention. Copying
  // Gadget-2's fac verbatim without compensating for that difference silently produced a PM force
  // too small by exactly gridSize^3 -- found via an independent Ewald-summation cross-check
  // (LOG.md §27), root-caused by comparing this port's own already-normalized potential grid
  // magnitude against a from-scratch discrete Poisson-equation re-derivation, and confirmed
  // numerically: multiplying by gridSize^3 closes the gap from ~100% relative error (PM
  // contributing essentially nothing) to a few percent across the erfc short/long-range crossover
  // (LOG.md §28). The extra gridSize^3 factor exactly compensates for the different FFT
  // normalization convention chosen here -- it is not an independent physical constant.
  const float physicalScale = gravityConstant * (float)solver.gridSize * (float)solver.gridSize *
                              (float)solver.gridSize / ((float)M_PI * solver.boxSize);

  // Optional potential output (LOG.md §30): solver.d_potential still holds the raw (unscaled)
  // Green's-function-filtered grid at this point -- pm_finite_diff_force below only READS it, so
  // interpolating it here (before or after the force loop, doesn't matter) is safe. The SAME
  // physicalScale used for force is the correct constant here too: physicalScale is exactly
  // Phi_true/Phi_grid (LOG.md §28's derivation), independent of the finite-difference gridScale,
  // which is a force-specific unit conversion (grid-index derivative -> physical derivative).
  if (d_potential_out)
  {
    pm_cic_interpolate(d_bodies_pos, n, solver.d_potential, d_potential_out, solver.gridSize,
                        solver.gridPitch, solver.boxSize, stream);
    pm_scale_buffer(d_potential_out, n, physicalScale, stream);
    PMSUB("cic_gather_potential");
  }

  float *const outs[3] = { d_force_x, d_force_y, d_force_z };
  for (int axis = 0; axis < 3; axis++)
  {
    pm_finite_diff_force(solver.d_potential, solver.d_forceScratch, solver.gridSize,
                          solver.gridPitch, axis, solver.boxSize, stream, physicalScale);
    pm_cic_interpolate(d_bodies_pos, n, solver.d_forceScratch, outs[axis], solver.gridSize,
                        solver.gridPitch, solver.boxSize, stream);
    PMSUB("finite_diff+gather/axis");
  }
}

static float g_dumpContextBoxSize = 0.0f;
static int   g_dumpContextGridSize = 0;
static bool  g_dumpContextSet = false;

void pm_set_dump_context(float boxSize, int gridSize)
{
  g_dumpContextBoxSize = boxSize;
  g_dumpContextGridSize = gridSize;
  g_dumpContextSet = true;
}

bool pm_get_dump_context(float &boxSize, int &gridSize)
{
  boxSize = g_dumpContextBoxSize;
  gridSize = g_dumpContextGridSize;
  return g_dumpContextSet;
}
