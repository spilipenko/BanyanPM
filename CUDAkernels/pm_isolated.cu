#include "hip/hip_runtime.h"
#include "pm.h"

// Non-periodic (isolated/vacuum-boundary) PM long-range solver (PLAN.md Phase 4's deferred
// item, LOG.md's own §-numbered writeup). Unlike the periodic path (pm_greens.cu, analytic
// 1/k^2 Poisson solve), an isolated Green's function has no convenient closed k-space form, so
// Gadget-2 builds it directly in REAL space, once, at startup (pm_nonperiodic.c:321-362), FFTs
// it once, and caches the result -- every subsequent step is then just an ordinary
// convolution-theorem multiply (density_hat * kernel_hat), not a per-step analytic formula.

#define PM_PI_F 3.14159265358979323846f

// Gadget-2's own compile-time short/long-range split constant (allvars.h:83) -- same value the
// periodic path uses (pm_greens.cu's PM_ASMTH), duplicated here rather than shared across
// translation units for the same reason pm_greens.cu gives: it's a trivial constant, not worth a
// header dependency.
#define PM_ASMTH 1.25f

// Builds the real-space kernel on a GRID^3 grid (GRID = 2*PMGRID, the caller's "gridSize" here),
// in DIMENSIONLESS grid-relative units -- matching Gadget-2's own convention exactly
// (pm_nonperiodic.c:321-362): position expressed as a fraction of the grid (x=i/GRID), wrapped
// into (-0.5,0.5] so the kernel is correctly centered for what will become a circular (but,
// because of the doubled/zero-padded grid, effectively linear within the occupied region)
// convolution. kernel(r) = -(1-erfc(u))/r, u = 0.5*r/(ASMTH/GRID) -- the same short-range
// Gaussian-type suppression the periodic Green's function applies, just constructed directly in
// real space here instead of analytically in k-space. The r=0 singularity is replaced by its
// analytic limit (GADGET2_NOTES.md), not a runtime branch-free hack.
__global__ void pm_build_isolated_kernel_kernel(float *kernel, int gridSize, int gridPitch)
{
  const size_t total = (size_t)gridSize * gridSize * gridSize;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  const int k = (int)(idx % gridSize);
  const size_t tmp = idx / gridSize;
  const int j = (int)(tmp % gridSize);
  const int i = (int)(tmp / gridSize);

  float x = (float)i / gridSize;
  float y = (float)j / gridSize;
  float z = (float)k / gridSize;
  if (x >= 0.5f) x -= 1.0f;
  if (y >= 0.5f) y -= 1.0f;
  if (z >= 0.5f) z -= 1.0f;

  const float r = sqrtf(x*x + y*y + z*z);
  const float asmthGrid = PM_ASMTH / gridSize;
  const float u = 0.5f * r / asmthGrid;

  float value;
  if (r > 0.0f)
  {
    const float fac = 1.0f - erfcf(u);
    value = -fac / r;
  }
  else
  {
    value = -1.0f / (sqrtf(PM_PI_F) * asmthGrid);
  }

  kernel[((size_t)i * gridSize + j) * gridPitch + k] = value;
}

void pm_build_isolated_kernel(float *d_kernel, int gridSize, int gridPitch, hipStream_t stream)
{
  const size_t total = (size_t)gridSize * gridSize * gridSize;
  const int threads = 256;
  const size_t blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(pm_build_isolated_kernel_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_kernel, gridSize, gridPitch);
}

// Applies ONLY the CIC deconvolution factor (sin(x)/x per axis, ^4 total -- same formula as the
// periodic path's applyCicDeconvolution, compensating for both the mass-assignment and later
// force-readout CIC windows in one shot) to the kernel's own FFT, done ONCE at solver-creation
// time. Unlike the periodic Green's-function multiply, there is no analytic 1/k^2 factor here
// (the kernel's real-space VALUES already encode the full Green's function -- this step only
// undoes the CIC smoothing) and no k=0 special-casing (no Jeans-swindle DC removal needed: an
// isolated, decaying Green's function has no "mean density" pathology the periodic case has).
__global__ void pm_isolated_kernel_deconvolve_kernel(float2 *kernelHat, int gridSize, int complexPitch)
{
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
  const int kz = z;

  float fx = 1.f, fy = 1.f, fz = 1.f;
  if (kx != 0) { const float t = (PM_PI_F * kx) / gridSize; fx = sinf(t) / t; }
  if (ky != 0) { const float t = (PM_PI_F * ky) / gridSize; fy = sinf(t) / t; }
  if (kz != 0) { const float t = (PM_PI_F * kz) / gridSize; fz = sinf(t) / t; }
  const float ff = 1.f / (fx * fy * fz);
  const float deconv = ff * ff * ff * ff;

  const size_t idx = ((size_t)x * gridSize + y) * nz + z;
  kernelHat[idx].x *= deconv;
  kernelHat[idx].y *= deconv;
}

void pm_isolated_kernel_deconvolve(float *d_kernelHat, int gridSize, hipStream_t stream)
{
  const int complexPitch = gridSize / 2 + 1;
  const size_t total = (size_t)gridSize * gridSize * complexPitch;
  const int threads = 256;
  const size_t blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(pm_isolated_kernel_deconvolve_kernel, dim3(blocks), dim3(threads), 0, stream,
                      (float2*)d_kernelHat, gridSize, complexPitch);
}

// Ordinary complex multiply (convolution theorem): density_hat *= kernel_hat, in place on the
// density buffer. This is the entire per-step "Green's function application" for the isolated
// case -- no analytic formula, just the cached kernel's own FFT.
__global__ void pm_complex_multiply_kernel(float2 *density, const float2 *kernel, int gridSize, int complexPitch)
{
  const size_t nz    = (size_t)complexPitch;
  const size_t total = (size_t)gridSize * gridSize * nz;

  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total)
    return;

  const float2 d = density[i];
  const float2 k = kernel[i];
  density[i] = make_float2(d.x * k.x - d.y * k.y, d.x * k.y + d.y * k.x);
}

void pm_complex_multiply(float *d_density, const float *d_kernelHat, int gridSize, hipStream_t stream)
{
  const int complexPitch = gridSize / 2 + 1;
  const size_t total = (size_t)gridSize * gridSize * complexPitch;
  const int threads = 256;
  const size_t blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(pm_complex_multiply_kernel, dim3(blocks), dim3(threads), 0, stream,
                      (float2*)d_density, (const float2*)d_kernelHat, gridSize, complexPitch);
}
