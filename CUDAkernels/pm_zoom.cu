#include "hip/hip_runtime.h"
#include "pm_zoom.h"

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
    float invCellSize, float3 corner)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
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
  if (ix < 0 || ix + 1 >= gridSize || iy < 0 || iy + 1 >= gridSize || iz < 0 || iz + 1 >= gridSize)
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
                                 int gridPitch, float cellSize, float3 corner, hipStream_t stream)
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
  hipLaunchKernelGGL(pm_cic_assign_mass_clipped_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_bodies_pos, n, d_density, gridSize, gridPitch, invCellSize, corner);
}

__global__ void pm_cic_interpolate_clipped_kernel(
    const float4 *bodies_pos, int n, const float *grid, float *out, int gridSize, int gridPitch,
    float invCellSize, float3 corner)
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

  if (ix < 0 || ix + 1 >= gridSize || iy < 0 || iy + 1 >= gridSize || iz < 0 || iz + 1 >= gridSize)
  {
    out[i] = 0.0f; // outside the fine grid entirely -- no contribution from this grid
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
                                 float3 corner, hipStream_t stream)
{
  // Bug fix (Phase 5 ticket 07, LOG.md): see pm_cic_assign_mass_clipped's own comment above for
  // the full rationale -- same missing `gridSize` factor.
  const float invCellSize = (float)gridSize / cellSize;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_cic_interpolate_clipped_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_bodies_pos, n, d_grid, d_out, gridSize, gridPitch, invCellSize, corner);
}

// Type-filtered accumulation (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.1's second effect): only
// particles whose type bit is set in `highResMask` receive the fine grid's force/potential.
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

#endif // GADGET_HIP_HIGHRES
