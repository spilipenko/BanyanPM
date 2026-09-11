#include "hip/hip_runtime.h"
#include "pm.h"

// Local PI constant rather than relying on M_PI being available in device code under this
// toolchain -- avoids a host/device math-macro portability assumption.
#define PM_PI_F 3.14159265358979323846f

// Gadget-2's own compile-time short/long-range split constant (allvars.h:83), not a runtime
// parameter -- see GADGET2_NOTES.md.
#define PM_ASMTH 1.25f

__global__ void pm_greens_multiply_periodic_kernel(
    float2 *grid, int gridSize, int complexPitch, float asmth2, int applyCicDeconvolution)
{
  // complexPitch == gridSize/2+1 exactly (pm.h's gridPitch=2*(gridSize/2+1) convention), so the
  // hermitian array is tightly packed in this (x,y,z) decode space -- no separate stride needed
  // here beyond complexPitch itself.
  const size_t nz    = (size_t)complexPitch;
  const size_t total = (size_t)gridSize * gridSize * nz;

  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total)
    return;

  const int z = (int)(i % nz);
  const size_t tmp = i / nz;
  const int y = (int)(tmp % gridSize);
  const int x = (int)(tmp / gridSize);

  const int halfN = gridSize / 2;
  const int kx = (x > halfN) ? x - gridSize : x;
  const int ky = (y > halfN) ? y - gridSize : y;
  const int kz = z; // stored half already excludes the redundant negative-kz mirror

  const int k2 = kx * kx + ky * ky + kz * kz;

  const size_t idx = ((size_t)x * gridSize + y) * complexPitch + z;

  if (k2 == 0)
  {
    // Removes the mean/DC density component -- periodic self-gravity of a nonzero mean density
    // has no solution otherwise (the standard "Jeans swindle"), matching pm_periodic.c:431's
    // fft_of_rhogrid[0]=0.
    grid[idx] = make_float2(0.f, 0.f);
    return;
  }

  // Gadget-2's own Green's function (pm_periodic.c:396-421): the usual 1/k^2 Poisson solve,
  // low-pass filtered by a Gaussian of scale Asmth -- this is the long-range/short-range split,
  // PM handles scales >~Asmth, the tree's shortrange_table complementary erfc factor handles the
  // rest (GADGET2_NOTES.md).
  float smth = -expf(-(float)k2 * asmth2) / (float)k2;

  if (applyCicDeconvolution)
  {
    // Compensates for the CIC window applied at both mass-assignment and (later) force-
    // interpolation stages -- sin(x)/x per axis, raised to the 4th power total. Only correct when
    // the density grid actually came from CIC assignment; deliberately skippable for the
    // analytically-seeded Poisson-solve correctness test (pm.h's applyCicDeconvolution doc).
    float fx = 1.f, fy = 1.f, fz = 1.f;
    if (kx != 0) { const float t = (PM_PI_F * kx) / gridSize; fx = sinf(t) / t; }
    if (ky != 0) { const float t = (PM_PI_F * ky) / gridSize; fy = sinf(t) / t; }
    if (kz != 0) { const float t = (PM_PI_F * kz) / gridSize; fz = sinf(t) / t; }
    const float ff = 1.f / (fx * fy * fz);
    smth *= ff * ff * ff * ff;
  }

  grid[idx].x *= smth;
  grid[idx].y *= smth;
}

void pm_greens_multiply_periodic(float *d_grid, int gridSize, float boxSize,
                                  bool applyCicDeconvolution, hipStream_t stream)
{
  const int complexPitch = gridSize / 2 + 1;
  const float asmth = PM_ASMTH * boxSize / gridSize;
  const float asmth2 = (2.0f * PM_PI_F * asmth / boxSize) * (2.0f * PM_PI_F * asmth / boxSize);

  const size_t total = (size_t)gridSize * gridSize * complexPitch;
  const int threads = 256;
  const size_t blocks = (total + threads - 1) / threads;

  hipLaunchKernelGGL(pm_greens_multiply_periodic_kernel, dim3(blocks), dim3(threads), 0, stream,
                      (float2*)d_grid, gridSize, complexPitch, asmth2, applyCicDeconvolution ? 1 : 0);
}
