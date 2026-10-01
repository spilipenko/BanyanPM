#include "hip/hip_runtime.h"
#include "pm.h"
#include "gadget_units.h"
#include "gadget_index_spaces.h"   // CurrentOrder / IdSpace (T44 item 1)   // GFreeAcc: PM output carries no G until someone applies it
#include <cstdlib>

// Flattened index into the padded density grid (row-major x,y,z with z the fastest/padded axis).
__device__ __forceinline__ int pm_grid_index(int x, int y, int z, int gridSize, int gridPitch)
{
  return (x * gridSize + y) * gridPitch + z;
}

// T36: which particle a lane takes, and why it is not the obvious choice.
//
// The tree hands this kernel Peano-Hilbert-SORTED positions, so consecutive particles are spatial
// neighbours and usually share a grid cell. For CIC, particles in the same cell share all EIGHT
// target addresses exactly -- only the weights differ. With the natural mapping (lane L takes
// particle base+L) an entire wavefront therefore issues its eight atomicAdds to the same eight
// addresses, and the hardware serialises them. Measured at z=0 on 512^3: 9.48 s of a 9.70 s PM
// phase, ~14 M particles/s, while the FFTs either side cost 0.03 s each. It is not the mesh solve
// that is expensive, it is putting the mass on the mesh.
//
// Mode 1 transposes the mapping: lane L takes particle L*gridDim.x + blockIdx.x, so lanes in a
// wavefront are ~gridDim.x particles apart -- different regions of the box, hence different cells
// and no same-address serialisation. The cost is that the float4 loads are no longer coalesced;
// at this size that is a few hundred MB of extra traffic, against atomics that were costing
// whole seconds.
//
// Mode 0 restores the legacy mapping (GADGET_HIP_PM_CIC=0), which is what makes the two
// measurable against each other on one binary.
__device__ __forceinline__ int pm_cic_particle_index(int mode)
{
  if (mode == 0)
    return blockIdx.x * blockDim.x + threadIdx.x;
  return threadIdx.x * gridDim.x + blockIdx.x;
}

__global__ void pm_cic_assign_mass_kernel(
    const float4 *bodies_pos, int n,
    float *density, int gridSize, int gridPitch, float invCellSize, int mode)
{
  int i = pm_cic_particle_index(mode);
  if (i >= n)
    return;

  const float4 p = bodies_pos[i];
  const float mass = p.w;

  const float gx = p.x * invCellSize;
  const float gy = p.y * invCellSize;
  const float gz = p.z * invCellSize;

  const int ix = (int)floorf(gx);
  const int iy = (int)floorf(gy);
  const int iz = (int)floorf(gz);

  const float dx = gx - ix;
  const float dy = gy - iy;
  const float dz = gz - iz;

  // Periodic wrap of the two bracketing cells per axis (Gadget-2's own CIC convention --
  // wrap the bin index, not the particle position).
  const int ix0 = ((ix % gridSize) + gridSize) % gridSize;
  const int iy0 = ((iy % gridSize) + gridSize) % gridSize;
  const int iz0 = ((iz % gridSize) + gridSize) % gridSize;
  const int ix1 = (ix0 + 1) % gridSize;
  const int iy1 = (iy0 + 1) % gridSize;
  const int iz1 = (iz0 + 1) % gridSize;

  atomicAdd(&density[pm_grid_index(ix0, iy0, iz0, gridSize, gridPitch)], mass * (1 - dx) * (1 - dy) * (1 - dz));
  atomicAdd(&density[pm_grid_index(ix0, iy0, iz1, gridSize, gridPitch)], mass * (1 - dx) * (1 - dy) * dz);
  atomicAdd(&density[pm_grid_index(ix0, iy1, iz0, gridSize, gridPitch)], mass * (1 - dx) * dy * (1 - dz));
  atomicAdd(&density[pm_grid_index(ix0, iy1, iz1, gridSize, gridPitch)], mass * (1 - dx) * dy * dz);
  atomicAdd(&density[pm_grid_index(ix1, iy0, iz0, gridSize, gridPitch)], mass * dx * (1 - dy) * (1 - dz));
  atomicAdd(&density[pm_grid_index(ix1, iy0, iz1, gridSize, gridPitch)], mass * dx * (1 - dy) * dz);
  atomicAdd(&density[pm_grid_index(ix1, iy1, iz0, gridSize, gridPitch)], mass * dx * dy * (1 - dz));
  atomicAdd(&density[pm_grid_index(ix1, iy1, iz1, gridSize, gridPitch)], mass * dx * dy * dz);
}

void pm_cic_assign_mass(const float4 *d_bodies_pos, int n, float *d_density,
                         int gridSize, int gridPitch, float boxSize, hipStream_t stream)
{
  const float invCellSize = gridSize / boxSize;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  // Mode 1 (the transposed mapping) is the default; GADGET_HIP_PM_CIC=0 restores the legacy one.
  // The index mapping only permutes which lane handles which particle, so both modes sum exactly
  // the same set of contributions -- they differ in float summation ORDER, which is the same class
  // of difference the atomicAdd nondeterminism already introduces (T3).
  static const int cicMode = (getenv("GADGET_HIP_PM_CIC") != NULL)
                               ? atoi(getenv("GADGET_HIP_PM_CIC")) : 1;
  hipLaunchKernelGGL(pm_cic_assign_mass_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_bodies_pos, n, d_density, gridSize, gridPitch, invCellSize, cicMode);
}

// Gather counterpart of pm_cic_assign_mass: trilinearly samples d_grid at each particle's
// position using the identical 8-cell weights/periodic-wrap geometry, matching Gadget-2's own
// force read-out (pm_periodic.c:640-671). Writes (does not accumulate) d_out[i] -- one axis's
// force component per call, matching pm_finite_diff_force's per-axis convention.
__global__ void pm_cic_interpolate_kernel(
    const float4 *bodies_pos, int n,
    const float *grid, float *out, int gridSize, int gridPitch, float invCellSize)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  const float4 p = bodies_pos[i];

  const float gx = p.x * invCellSize;
  const float gy = p.y * invCellSize;
  const float gz = p.z * invCellSize;

  const int ix = (int)floorf(gx);
  const int iy = (int)floorf(gy);
  const int iz = (int)floorf(gz);

  const float dx = gx - ix;
  const float dy = gy - iy;
  const float dz = gz - iz;

  const int ix0 = ((ix % gridSize) + gridSize) % gridSize;
  const int iy0 = ((iy % gridSize) + gridSize) % gridSize;
  const int iz0 = ((iz % gridSize) + gridSize) % gridSize;
  const int ix1 = (ix0 + 1) % gridSize;
  const int iy1 = (iy0 + 1) % gridSize;
  const int iz1 = (iz0 + 1) % gridSize;

  float v = 0.0f;
  v += grid[pm_grid_index(ix0, iy0, iz0, gridSize, gridPitch)] * (1 - dx) * (1 - dy) * (1 - dz);
  v += grid[pm_grid_index(ix0, iy0, iz1, gridSize, gridPitch)] * (1 - dx) * (1 - dy) * dz;
  v += grid[pm_grid_index(ix0, iy1, iz0, gridSize, gridPitch)] * (1 - dx) * dy * (1 - dz);
  v += grid[pm_grid_index(ix0, iy1, iz1, gridSize, gridPitch)] * (1 - dx) * dy * dz;
  v += grid[pm_grid_index(ix1, iy0, iz0, gridSize, gridPitch)] * dx * (1 - dy) * (1 - dz);
  v += grid[pm_grid_index(ix1, iy0, iz1, gridSize, gridPitch)] * dx * (1 - dy) * dz;
  v += grid[pm_grid_index(ix1, iy1, iz0, gridSize, gridPitch)] * dx * dy * (1 - dz);
  v += grid[pm_grid_index(ix1, iy1, iz1, gridSize, gridPitch)] * dx * dy * dz;

  out[i] = v;
}

void pm_cic_interpolate(const float4 *d_bodies_pos, int n, const float *d_grid, float *d_out,
                         int gridSize, int gridPitch, float boxSize, hipStream_t stream)
{
  const float invCellSize = gridSize / boxSize;
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_cic_interpolate_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_bodies_pos, n, d_grid, d_out, gridSize, gridPitch, invCellSize);
}

// Phase 4 (PLAN.md): accumulates PM's per-axis long-range force (d_fx/fy/fz, from
// pm_compute_forces_periodic) directly into the tree's own float4 acceleration buffer (.xyz),
// so the real per-step simulation loop can add PM on top of the tree's already-computed
// short-range force with one small kernel instead of a host round-trip. Deliberately leaves
// .w (potential) untouched -- PM's own potential isn't computed by this pipeline yet (only the
// finite-differenced force), so reported total energy is short-range-only until that's added;
// this doesn't affect the dynamics (integration only uses .xyz), only energy-conservation
// bookkeeping, which is not the real risk this port already flagged elsewhere as the current gap.
__global__ void pm_add_force_to_acc_kernel(float4 *acc, const float *fx, const float *fy,
                                            const float *fz, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  acc[i].x += fx[i];
  acc[i].y += fy[i];
  acc[i].z += fz[i];
}

void pm_add_force_to_acc(float4 *d_acc, const float *d_fx, const float *d_fy, const float *d_fz,
                          int n, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_add_force_to_acc_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_acc, d_fx, d_fy, d_fz, n);
}

// T23 counterpart: add the PM force/potential only where the walk refreshed acc1. Adding it to a
// stale entry double-counts the long-range force even when the gravitational constant is 1.
__global__ void pm_add_force_masked_kernel(float4 *acc, const float *fx, const float *fy,
                                           const float *fz, const float *pot, int n,
                                           const int *active)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || active[i] == 0) return;
  // Both halves are optional: the force arrays are null on the potential-only call and vice
  // versa. Dereferencing them unconditionally faults (caught immediately at 512^3 on a null
  // read), so each group is guarded on its own pointer.
  if (fx)  { acc[i].x += fx[i]; acc[i].y += fy[i]; acc[i].z += fz[i]; }
  if (pot) { acc[i].w += pot[i]; }
}

void pm_add_force_to_acc_masked(float4 *d_acc, const float *d_fx, const float *d_fy,
                                const float *d_fz, const float *d_pot, int n,
                                const int *d_active, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_add_force_masked_kernel, dim3(blocks), dim3(threads), 0, stream,
                     d_acc, d_fx, d_fy, d_fz, d_pot, n, d_active);
}

// Elementwise in-place scale, used by pm_compute_forces_periodic to convert its raw
// CIC-interpolated grid potential into physical units (LOG.md §30).
__global__ void pm_scale_buffer_kernel(float *buf, int n, float scale)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  buf[i] *= scale;
}

void pm_scale_buffer(float *d_buf, int n, float scale, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_scale_buffer_kernel, dim3(blocks), dim3(threads), 0, stream, d_buf, n, scale);
}

// T23: masked variants. bodies_acc1 is only PARTIALLY refreshed each iteration -- the tree walk
// writes it for particles in ACTIVE groups only -- so a whole-buffer post-pass over it re-processes
// stale entries. With individual timesteps the active set can collapse (measured: 3 of 4,194,801
// groups at 512^3), and the inactive particles then had the PM force added to, and the gravitational
// constant applied to, values that ALREADY carried both from a previous iteration. Two such steps
// compounded to G^2 ~ 1850x and blew the energy up by 1774 in relative terms.
//
// `active` is the walk's own output mask (active_inout / tree.activePartlist, zeroed before the
// walk and set to 1 by it for every particle it writes), so it marks exactly the entries whose
// acc1 is fresh this iteration. One float4 per thread here, so the mask is indexed per PARTICLE.
__global__ void pm_scale_buffer4_masked_kernel(float4 *acc, int n, float scale, const int *active)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || active[i] == 0) return;
  acc[i].x *= scale; acc[i].y *= scale; acc[i].z *= scale; acc[i].w *= scale;
}

void pm_scale_acc_masked(float4 *d_acc, int n, float scale, const int *d_active, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_scale_buffer4_masked_kernel, dim3(blocks), dim3(threads), 0, stream,
                     d_acc, n, scale, d_active);
}

// Counterpart to pm_add_force_to_acc for the .w (potential) component (LOG.md §30) -- pot[i] is
// already in physical per-unit-mass units (pm_compute_forces_periodic applies physicalScale
// before this would ever be called), matching bodies_acc.w's own convention exactly.
__global__ void pm_add_potential_to_acc_kernel(float4 *acc, const float *pot, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  acc[i].w += pot[i];
}

void pm_add_potential_to_acc(float4 *d_acc, const float *d_potential, int n, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_add_potential_to_acc_kernel, dim3(blocks), dim3(threads), 0, stream,
                      d_acc, d_potential, n);
}

// ---------------------------------------------------------------------------------------------
// T36: PM force stored in ID SPACE, so it can survive between PM steps.
//
// Gadget-2 computes the long-range force only on PM steps and keeps it in P[i].GravPM, applying
// the stored value on every intervening step (accel.c: long_range_force() is called only when
// All.PM_Ti_endstep == All.Ti_Current; the kick always uses GravAccel + GravPM). This port instead
// recomputed PM every single step -- 0.73 s of a 2.71 s sparse step at 512^3.
//
// The arrays cannot simply be kept, because they are indexed by CURRENT sort order and the tree
// re-sorts every step: a stored entry would be handed to whichever particle later occupies that
// index. That is the C-B-13 / C-C-18 / C-D-08 defect class, and it would be invisible in the
// energies while quietly scrambling the long-range force. Storing by particle ID instead makes the
// association permanent: ids are dense here (the same property T35's softening table relies on,
// and checked the same way), so the id IS the index.
__global__ void pm_scatter_by_id_kernel(const int n,
                                        CurrentOrder<const unsigned long long> ids,
                                        CurrentOrder<const float> fx,
                                        CurrentOrder<const float> fy,
                                        CurrentOrder<const float> fz,
                                        CurrentOrder<const float> pot,
                                        IdSpace<GFreeAcc> outX, IdSpace<GFreeAcc> outY,
                                        IdSpace<GFreeAcc> outZ, IdSpace<float> outPot)
{
  // This kernel IS the space change: it reads the PM solve's output in post-sort order and writes it
  // into id space, which is the whole reason the stored long-range force survives the re-sort
  // between PM steps.
  const CurrentIdx i = { blockIdx.x * blockDim.x + threadIdx.x };
  if (i.v >= (unsigned int) n) return;
  const ParticleId id = id_at(i, ids);
  if (id.v >= (unsigned long long) n) return;   // guarded by the same dense-id check as T35
  // The PM solve is called with gravityConstant = 1.0f, so what it produces is G-free. Tagging
  // it at the point of STORAGE is what makes every later consumer declare which G it applies.
  // (outPot stays a plain float: the potential's G travels with the t30 coefficients in
  // compute_energies, a different and more tangled path -- see the note in pm.h.)
  outX[id].v = fx[i]; outY[id].v = fy[i]; outZ[id].v = fz[i]; outPot[id] = pot[i];
}

void pm_scatter_by_id(const unsigned long long *d_ids, const float *d_fx, const float *d_fy,
                      const float *d_fz, const float *d_pot,
                      GFreeAcc *d_outX, GFreeAcc *d_outY, GFreeAcc *d_outZ, float *d_outPot,
                      int n, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  // The wrappers are the boundary: raw device pointers in, space-tagged views out. Constructing
  // them here rather than at every caller keeps include/pm.h's interface unchanged.
  hipLaunchKernelGGL(pm_scatter_by_id_kernel, dim3(blocks), dim3(threads), 0, stream,
                     n, CurrentOrder<const unsigned long long>{d_ids},
                     CurrentOrder<const float>{d_fx}, CurrentOrder<const float>{d_fy},
                     CurrentOrder<const float>{d_fz}, CurrentOrder<const float>{d_pot},
                     IdSpace<GFreeAcc>{d_outX}, IdSpace<GFreeAcc>{d_outY},
                     IdSpace<GFreeAcc>{d_outZ}, IdSpace<float>{d_outPot});
}

// Gather-and-add, masked. The gather is the price of id-space storage, but it is paid only for
// ACTIVE particles -- 1.6% of them on the sparse steps this exists to speed up -- because the mask
// is tested before the indirection.
__global__ void pm_add_by_id_masked_kernel(CurrentOrder<float4> acc,
                                           CurrentOrder<const unsigned long long> ids,
                                           IdSpace<const GFreeAcc> byIdX,
                                           IdSpace<const GFreeAcc> byIdY,
                                           IdSpace<const GFreeAcc> byIdZ,
                                           IdSpace<const float> byIdPot,
                                           int n, CurrentOrder<const int> active)
{
  const CurrentIdx i = { blockIdx.x * blockDim.x + threadIdx.x };
  if (i.v >= (unsigned int) n || active[i] == 0) return;
  const ParticleId id = id_at(i, ids);
  if (id.v >= (unsigned long long) n) return;
  // Phase 3 wants the POTENTIAL without the force: under the PM cadence the long-range force is
  // applied as a separate kick, so folding it into acc here would double-count it, but the
  // potential is a diagnostic and still belongs in acc.w. Each group is therefore optional --
  // previously all four were dereferenced unconditionally, and passing NULL for the force faulted
  // the GPU at NULL + id*4.
  // This is the one path where staying G-free is correct: acc is bodies_acc1, and
  // pm_scale_acc_masked multiplies the whole of it by G a few lines later in the caller. The escape
  // is named at length so that using it is a decision rather than a reflex.
  if (!byIdX.is_null())
  {
    acc[i].x += byIdX[id].raw_because_acc1_is_G_scaled_downstream();
    acc[i].y += byIdY[id].raw_because_acc1_is_G_scaled_downstream();
    acc[i].z += byIdZ[id].raw_because_acc1_is_G_scaled_downstream();
  }
  if (!byIdPot.is_null()) acc[i].w += byIdPot[id];
}

void pm_add_by_id_masked(float4 *d_acc, const unsigned long long *d_ids,
                         const GFreeAcc *d_byIdX, const GFreeAcc *d_byIdY,
                         const GFreeAcc *d_byIdZ,
                         const float *d_byIdPot, int n, const int *d_active, hipStream_t stream)
{
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(pm_add_by_id_masked_kernel, dim3(blocks), dim3(threads), 0, stream,
                     CurrentOrder<float4>{d_acc}, CurrentOrder<const unsigned long long>{d_ids},
                     IdSpace<const GFreeAcc>{d_byIdX}, IdSpace<const GFreeAcc>{d_byIdY},
                     IdSpace<const GFreeAcc>{d_byIdZ}, IdSpace<const float>{d_byIdPot},
                     n, CurrentOrder<const int>{d_active});
}
