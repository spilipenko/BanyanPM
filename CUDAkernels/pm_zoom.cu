#include "hip/hip_runtime.h"
#include "pm_zoom.h"
#include <cstdlib>

#ifdef GADGET_HIP_HIGHRES

// Phase 5 ticket 06 (PLAN.md): zoom (PLACEHIGHRESREGION) fine-grid device kernels. See pm_zoom.h
// for the full rationale; formulas cited against PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.4.

#define PM_ZOOM_PI_F 3.14159265358979323846f

// Same grid-relative `u` convention as pm_build_isolated_kernel_kernel (pm_isolated.cu) -- see
// that file's own comment for the coordinate-wrapping rationale, unchanged here.
__global__ void pm_build_finegrid_kernel_kernel(float *kernel, int gridSize, int gridPitch,
                                                 float asmthRatio)
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
  // Fine grid's OWN ASMTH=1.25/gridSize scale, matching pm_nonperiodic.c:391-402's `u` exactly
  // (the same dimensionless constant every other kernel in this project uses -- PM_ASMTH in
  // pm_isolated.cu, duplicated here for the same "not worth a header dependency" reason).
  const float asmthGrid = 1.25f / gridSize;
  const float u = 0.5f * r / asmthGrid;

  float value;
  if (r > 0.0f)
  {
    const float fac = erfcf(u * asmthRatio) - erfcf(u);
    value = -fac / r;
  }
  else
  {
    // Analytic r=0 limit (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.4): fac -> 1-asmthRatio as r->0.
    const float fac0 = 1.0f - asmthRatio;
    value = -fac0 / (sqrtf(PM_ZOOM_PI_F) * asmthGrid);
  }

  kernel[((size_t)i * gridSize + j) * gridPitch + k] = value;
}

void pm_build_finegrid_kernel(float *d_kernel, int gridSize, int gridPitch, float asmthRatio,
                               hipStream_t stream)
{
  const size_t total = (size_t)gridSize * gridSize * gridSize;
  const int threads = 256;
  const size_t blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(pm_build_finegrid_kernel_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_kernel, gridSize, gridPitch, asmthRatio);
}

// Phase 5 ticket 06: CLIPPED CIC assign/interpolate -- see pm_zoom.h's own doc comment for why
// this differs from pm_cic_assign_mass/pm_cic_interpolate's periodic-wrap convention. Shares the
// exact same trilinear-weight math, just a different (bounds-check-and-skip, not modulo-wrap)
// treatment of the 8 surrounding cell indices.
__device__ __forceinline__ int pm_zoom_grid_index(int x, int y, int z, int gridSize, int gridPitch)
{
  return (x * gridSize + y) * gridPitch + z;
}

__global__ void pm_cic_assign_mass_clipped_kernel(
    const float4 *bodies_pos, int n, float *density, int gridSize, int gridPitch,
    float invCellSize, float3 corner, int depositCells, int mode)
{
  // T50: the same transposed lane->particle mapping T36 added to the COARSE scatter
  // (pm_cic.cu's pm_cic_particle_index) -- this kernel had been left on the legacy mapping, which
  // T36 measured at 9.48 s of a 9.70 s PM phase on the coarse grid before fixing it. Duplicated
  // rather than shared because pm_cic.cu's helper is static to that translation unit; keep the two
  // in step. mode 1 = transposed (default), mode 0 = legacy, via GADGET_HIP_PM_CIC.
  const int i = (mode == 0) ? (blockIdx.x * blockDim.x + threadIdx.x)
                            : (threadIdx.x * gridDim.x + blockIdx.x);
  if (i >= n) return;

  const float4 p = bodies_pos[i];
  const float mass = p.w;

  const float gx = (p.x - corner.x) * invCellSize;
  const float gy = (p.y - corner.y) * invCellSize;
  const float gz = (p.z - corner.z) * invCellSize;

  const int ix = (int)floorf(gx);
  const int iy = (int)floorf(gy);
  const int iz = (int)floorf(gz);

  // Clip: skip entirely unless all 8 surrounding cells (ix/ix+1 etc.) are in range -- matches
  // Gadget-2's own explicit bounds check before depositing (pm_nonperiodic.c:579-584), not a
  // periodic wrap (this grid is a small sub-region of a much larger box; wrapping would alias a
  // far-away particle's mass onto an essentially random cell).
  // T50 item G: bound by the DEPOSIT WINDOW (the inner half), not by the array extent.
  if (ix < 0 || ix + 1 >= depositCells ||
      iy < 0 || iy + 1 >= depositCells ||
      iz < 0 || iz + 1 >= depositCells)
    return;

  const float dx = gx - ix, dy = gy - iy, dz = gz - iz;

  atomicAdd(&density[pm_zoom_grid_index(ix,   iy,   iz,   gridSize, gridPitch)], mass * (1 - dx) * (1 - dy) * (1 - dz));
  atomicAdd(&density[pm_zoom_grid_index(ix,   iy,   iz+1, gridSize, gridPitch)], mass * (1 - dx) * (1 - dy) * dz);
  atomicAdd(&density[pm_zoom_grid_index(ix,   iy+1, iz,   gridSize, gridPitch)], mass * (1 - dx) * dy * (1 - dz));
  atomicAdd(&density[pm_zoom_grid_index(ix,   iy+1, iz+1, gridSize, gridPitch)], mass * (1 - dx) * dy * dz);
  atomicAdd(&density[pm_zoom_grid_index(ix+1, iy,   iz,   gridSize, gridPitch)], mass * dx * (1 - dy) * (1 - dz));
  atomicAdd(&density[pm_zoom_grid_index(ix+1, iy,   iz+1, gridSize, gridPitch)], mass * dx * (1 - dy) * dz);
  atomicAdd(&density[pm_zoom_grid_index(ix+1, iy+1, iz,   gridSize, gridPitch)], mass * dx * dy * (1 - dz));
  atomicAdd(&density[pm_zoom_grid_index(ix+1, iy+1, iz+1, gridSize, gridPitch)], mass * dx * dy * dz);
}

void pm_cic_assign_mass_clipped(const float4 *d_bodies_pos, int n, float *d_density, int gridSize,
                                 int gridPitch, float cellSize, float3 corner, int depositCells,
                                 hipStream_t stream)
{
  // Bug fix (Phase 5 ticket 07, LOG.md): this was `1.0f / cellSize`, missing the `gridSize`
  // factor pm_cic.cu's own (already-validated) `pm_cic_assign_mass`/`pm_cic_interpolate` use
  // (`invCellSize = gridSize / boxSize`, CUDAkernels/pm_cic.cu) -- `cellSize` here is actually the
  // grid's FULL physical box size (matching pm_solve_isolated.cpp's own doc comment: "meshSize...
  // is the size of the [box]... deposits wherever positions land within [0, meshSize)"), not a
  // single cell's own physical size despite the parameter's name, so converting a physical
  // coordinate to a GRID INDEX needs the grid cell COUNT folded in too. This silently shipped with
  // Ticket 06 (undetected there: the kernel real-space TABLE values were validated directly via
  // --zoom-kernel-test, and the real-particle end-to-end test only checked qualitative masking
  // behavior, never absolute force accuracy against a ground truth) -- caught by Ticket 07's own
  // --zoom-treepm-force-test showing systematic ~20-60% errors against exact Newtonian instead of
  // the ~2-5% the equivalent coarse-only test achieves.
  const float invCellSize = (float)gridSize / cellSize;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  // Same env knob and same default as the coarse scatter, so one binary can A/B both grids.
  static const int cicMode = (getenv("GADGET_HIP_PM_CIC") != NULL)
                               ? atoi(getenv("GADGET_HIP_PM_CIC")) : 1;
  hipLaunchKernelGGL(pm_cic_assign_mass_clipped_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_bodies_pos, n, d_density, gridSize, gridPitch, invCellSize, corner,
                      depositCells, cicMode);
}

__global__ void pm_cic_interpolate_clipped_kernel(
    const float4 *bodies_pos, int n, const float *grid, float *out, int gridSize, int gridPitch,
    float invCellSize, float3 corner, int depositCells)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float4 p = bodies_pos[i];

  const float gx = (p.x - corner.x) * invCellSize;
  const float gy = (p.y - corner.y) * invCellSize;
  const float gz = (p.z - corner.z) * invCellSize;

  const int ix = (int)floorf(gx);
  const int iy = (int)floorf(gy);
  const int iz = (int)floorf(gz);

  // T50 item G: same inner-half window as the deposit; outside it the field is contaminated.
  if (ix < 0 || ix + 1 >= depositCells ||
      iy < 0 || iy + 1 >= depositCells ||
      iz < 0 || iz + 1 >= depositCells)
  {
    out[i] = 0.0f; // outside the fine grid's valid region -- no contribution from this grid
    return;
  }

  const float dx = gx - ix, dy = gy - iy, dz = gz - iz;

  float v = 0.0f;
  v += grid[pm_zoom_grid_index(ix,   iy,   iz,   gridSize, gridPitch)] * (1 - dx) * (1 - dy) * (1 - dz);
  v += grid[pm_zoom_grid_index(ix,   iy,   iz+1, gridSize, gridPitch)] * (1 - dx) * (1 - dy) * dz;
  v += grid[pm_zoom_grid_index(ix,   iy+1, iz,   gridSize, gridPitch)] * (1 - dx) * dy * (1 - dz);
  v += grid[pm_zoom_grid_index(ix,   iy+1, iz+1, gridSize, gridPitch)] * (1 - dx) * dy * dz;
  v += grid[pm_zoom_grid_index(ix+1, iy,   iz,   gridSize, gridPitch)] * dx * (1 - dy) * (1 - dz);
  v += grid[pm_zoom_grid_index(ix+1, iy,   iz+1, gridSize, gridPitch)] * dx * (1 - dy) * dz;
  v += grid[pm_zoom_grid_index(ix+1, iy+1, iz,   gridSize, gridPitch)] * dx * dy * (1 - dz);
  v += grid[pm_zoom_grid_index(ix+1, iy+1, iz+1, gridSize, gridPitch)] * dx * dy * dz;

  out[i] = v;
}

void pm_cic_interpolate_clipped(const float4 *d_bodies_pos, int n, const float *d_grid,
                                 float *d_out, int gridSize, int gridPitch, float cellSize,
                                 float3 corner, int depositCells, hipStream_t stream)
{
  // Bug fix (Phase 5 ticket 07, LOG.md): see pm_cic_assign_mass_clipped's own comment above for
  // the full rationale -- same missing `gridSize` factor.
  const float invCellSize = (float)gridSize / cellSize;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_cic_interpolate_clipped_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_bodies_pos, n, d_grid, d_out, gridSize, gridPitch, invCellSize, corner,
                      depositCells);
}

// Type-filtered accumulation (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.1's second effect): only
// particles whose type bit is set in `highResMask` receive the fine grid's force/potential.
// T58: accumulate the fine mesh's force (and potential) into the SAME arrays the coarse mesh wrote,
// for the zoom-mask types only. This is what makes the fine mesh share the coarse one's cadence,
// its id-space store, its once-per-interval kick, its drift-corrected energy diagnostic and its
// contribution to the Springel `aold` -- all four without touching any of them, because Gadget
// achieves the same thing the same way: long_range_force() (longrange.c:54-82) zeroes GravPM, then
// pmforce_periodic() and pmforce_nonperiodic(1) both `+=` into it, and every consumer downstream
// reads the SUM and never asks which mesh it came from.
__global__ void pm_accumulate_masked_longrange_kernel(float *fx, float *fy, float *fz, float *pot,
                                                       const float *zx, const float *zy,
                                                       const float *zz, const float *zpot,
                                                       const int *types,
                                                       unsigned int highResMask, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // The mask is the readout filter, not a deposit filter: Gadget skips the force READOUT for
  // non-PLACEHIGHRESREGION types (pm_nonperiodic.c:930-932) while depositing every particle inside
  // the mesh regardless of type (:575-600). The deposit side is pm_cic_assign_mass_clipped, which
  // takes no type array at all, so this is the only place the type enters.
  if (!((1u << types[i]) & highResMask)) return;
  fx[i] += zx[i];
  fy[i] += zy[i];
  fz[i] += zz[i];
  if (pot && zpot) pot[i] += zpot[i];
}

__global__ void pm_add_masked_force_to_acc_kernel(float4 *acc, const float *fx, const float *fy,
                                                    const float *fz, const int *types,
                                                    unsigned int highResMask, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!((1u << types[i]) & highResMask)) return;
  acc[i].x += fx[i];
  acc[i].y += fy[i];
  acc[i].z += fz[i];
}

void pm_add_masked_force_to_acc(float4 *d_acc, const float *d_fx, const float *d_fy,
                                 const float *d_fz, const int *d_types, unsigned int highResMask,
                                 int n, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_add_masked_force_to_acc_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_acc, d_fx, d_fy, d_fz, d_types, highResMask, n);
}

__global__ void pm_add_masked_potential_to_acc_kernel(float4 *acc, const float *pot,
                                                        const int *types, unsigned int highResMask,
                                                        int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!((1u << types[i]) & highResMask)) return;
  acc[i].w += pot[i];
}

void pm_add_masked_potential_to_acc(float4 *d_acc, const float *d_potential, const int *d_types,
                                     unsigned int highResMask, int n, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_add_masked_potential_to_acc_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_acc, d_potential, d_types, highResMask, n);
}

// T50 section 3A: the device-side replacement for the per-step host containment scan. One flag, set
// by any offending particle; no position transfer, only a 4-byte readback.
__device__ int g_zoom_out_of_range = 0;

__global__ void pm_zoom_any_out_of_range_kernel(const float4 *pos, const int *types, int n,
                                                 unsigned int highResMask, float3 corner,
                                                 float totalMeshSize)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!((1u << types[i]) & highResMask)) return;
  // Inclusive upper bound, identical to pm_zoom_region_out_of_range()'s host test: a particle exactly
  // on the far face counts as INSIDE. Keep the two in step -- this one is what runs in production and
  // the host one is what the standalone region unit test exercises.
  const float4 p = pos[i];
  const float hi = totalMeshSize;
  if (p.x - corner.x < 0.0f || p.x - corner.x > hi ||
      p.y - corner.y < 0.0f || p.y - corner.y > hi ||
      p.z - corner.z < 0.0f || p.z - corner.z > hi)
    g_zoom_out_of_range = 1;   // benign race: every writer writes the same value
}

bool pm_zoom_any_out_of_range(const float4 *d_bodies_pos, const int *d_types,
                              unsigned int highResMask, float3 corner, float totalMeshSize,
                              int n, hipStream_t stream)
{
  if (n <= 0 || highResMask == 0u) return false;
  const int zero = 0;
  hipMemcpyToSymbolAsync(HIP_SYMBOL(g_zoom_out_of_range), &zero, sizeof(zero), 0,
                          hipMemcpyHostToDevice, stream);
  const int threads = 256;
  const int blocks  = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_zoom_any_out_of_range_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_bodies_pos, d_types, n, highResMask, corner, totalMeshSize);
  int flag = 0;
  hipMemcpyFromSymbolAsync(&flag, HIP_SYMBOL(g_zoom_out_of_range), sizeof(flag), 0,
                            hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
  return flag != 0;
}

#endif // GADGET_HIP_HIGHRES

// T58: see the kernel's own comment. `d_pot`/`d_zpot` may both be null when no potential is wanted.
void pm_accumulate_masked_longrange(float *d_fx, float *d_fy, float *d_fz, float *d_pot,
                                     const float *d_zx, const float *d_zy, const float *d_zz,
                                     const float *d_zpot, const int *d_types,
                                     unsigned int highResMask, int n, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_accumulate_masked_longrange_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_fx, d_fy, d_fz, d_pot, d_zx, d_zy, d_zz, d_zpot, d_types, highResMask, n);
}
