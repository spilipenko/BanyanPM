#include "hip/hip_runtime.h"
#include "pm.h"

__device__ __forceinline__ int pm_wrap(int i, int gridSize)
{
  return ((i % gridSize) + gridSize) % gridSize;
}

__device__ __forceinline__ int pm_real_index(int x, int y, int z, int gridSize, int gridPitch)
{
  return (x * gridSize + y) * gridPitch + z;
}

// One thread per real-space cell. `axis` selects which of x/y/z the 4-point stencil differences
// along -- matching Gadget-2's own for(dim=0;dim<3;dim++) loop (pm_periodic.c:590-627), except
// this port holds the whole grid on one GPU, so periodic neighbor lookups are a plain wrap here
// instead of Gadget-2's ghost-cell/patch machinery (no domain decomposition to route around --
// LOG.md's Phase 4 architecture note on single-GPU vs. Gadget-2's own MPI design).
__global__ void pm_finite_diff_force_kernel(
    const float *potential, float *force, int gridSize, int gridPitch, int axis, float gridScale)
{
  const size_t total = (size_t)gridSize * gridSize * gridSize;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total)
    return;

  const int z = (int)(i % gridSize);
  const size_t tmp = i / gridSize;
  const int y = (int)(tmp % gridSize);
  const int x = (int)(tmp / gridSize);

  int xm1 = x, xp1 = x, xm2 = x, xp2 = x;
  int ym1 = y, yp1 = y, ym2 = y, yp2 = y;
  int zm1 = z, zp1 = z, zm2 = z, zp2 = z;

  switch (axis)
  {
    case 0: xm1 = pm_wrap(x - 1, gridSize); xp1 = pm_wrap(x + 1, gridSize);
            xm2 = pm_wrap(x - 2, gridSize); xp2 = pm_wrap(x + 2, gridSize); break;
    case 1: ym1 = pm_wrap(y - 1, gridSize); yp1 = pm_wrap(y + 1, gridSize);
            ym2 = pm_wrap(y - 2, gridSize); yp2 = pm_wrap(y + 2, gridSize); break;
    case 2: zm1 = pm_wrap(z - 1, gridSize); zp1 = pm_wrap(z + 1, gridSize);
            zm2 = pm_wrap(z - 2, gridSize); zp2 = pm_wrap(z + 2, gridSize); break;
  }

  const float phi_m1 = potential[pm_real_index(xm1, ym1, zm1, gridSize, gridPitch)];
  const float phi_p1 = potential[pm_real_index(xp1, yp1, zp1, gridSize, gridPitch)];
  const float phi_m2 = potential[pm_real_index(xm2, ym2, zm2, gridSize, gridPitch)];
  const float phi_p2 = potential[pm_real_index(xp2, yp2, zp2, gridSize, gridPitch)];

  // Gadget-2's own 4th-order (5-point) stencil, force sign already included (this is -dphi/dx,
  // not a raw derivative to be negated later) -- GADGET2_NOTES.md, pm_periodic.c:625-629.
  const float stencil = (4.0f / 3.0f) * (phi_m1 - phi_p1) - (1.0f / 6.0f) * (phi_m2 - phi_p2);

  force[pm_real_index(x, y, z, gridSize, gridPitch)] = gridScale * stencil;
}

void pm_finite_diff_force(const float *d_potential, float *d_force, int gridSize, int gridPitch,
                           int axis, float boxSize, hipStream_t stream, float extraScale)
{
  // gridScale is the grid-differencing normalization (1/(2*cellSize)); extraScale defaults to 1
  // for callers checking the stencil in isolation, and carries Gadget-2's own additional physical
  // prefactor (G/(pi*BoxSize)) when called from pm_compute_forces_periodic -- kept as a separate
  // multiplicative parameter rather than folded in unconditionally, so this kernel's correctness
  // (the error-prone numerical part) stays independently testable from that trivial scalar
  // (pm.h's doc comment on this function has the full reasoning).
  const float cellSize = boxSize / gridSize;
  const float gridScale = extraScale / (2.0f * cellSize);

  const size_t total = (size_t)gridSize * gridSize * gridSize;
  const int threads = 256;
  const size_t blocks = (total + threads - 1) / threads;

  hipLaunchKernelGGL(pm_finite_diff_force_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_potential, d_force, gridSize, gridPitch, axis, gridScale);
}
