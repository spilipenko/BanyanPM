#include "hip/hip_runtime.h"
#include "bonsai.h"
#include "gadget_spline_force.cuh"   // C-C-03: the SAME spline the tree path uses

// C-C-03: this used to be a flat Plummer kernel (`distSqr += eps2`) with the CLI `eps` as the
// length scale. Three differences from the tree path, all silent:
//   (1) kernel SHAPE     -- Plummer, not Gadget-2's cubic spline;
//   (2) length SCALE     -- `eps`, where the tree uses ForceSoftening = 2.8*Softening;
//   (3) PERIODICITY      -- no minimum image, while the tree applies pm_nearest under PERIODIC.
// It also returned no potential. `--direct` is what one reaches for as ground truth when the tree
// looks wrong, so all three corrupted exactly the comparison it exists to serve; this already
// produced one false result in this audit (the quadrupole test).
//
// `boxSize <= 0` selects the non-periodic branch, so a single kernel serves both builds.
static __device__ __forceinline__ float nearest_img(float x, float boxSize, float boxHalf)
{
    if (boxSize <= 0.0f) return x;
    if (x >  boxHalf) return x - boxSize;
    if (x < -boxHalf) return x + boxSize;
    return x;
}

__device__ float4
bodyBodyInteraction(float4 ai,          // .xyz accel, .w potential
                    float4 bi,
                    float4 bj,
                    float  hi,          // target ForceSoftening
                    float  hj,          // source ForceSoftening
                    float  boxSize,
                    float  boxHalf)
{
    const float3 r = make_float3(nearest_img(bj.x - bi.x, boxSize, boxHalf),
                                 nearest_img(bj.y - bi.y, boxSize, boxHalf),
                                 nearest_img(bj.z - bi.z, boxSize, boxHalf));
    const float r2 = r.x*r.x + r.y*r.y + r.z*r.z;
    const float rr = sqrtf(r2);

    // Gadget-2 softens a pair with max(h_target, h_source) -- forcetree.c:1537-1539 (C-C-17).
    float mrinv, mrinv3;
    gadget_spline_pair(rr, r2, bj.w, fmaxf(hi, hj), &mrinv, &mrinv3);

    ai.x += r.x * mrinv3;
    ai.y += r.y * mrinv3;
    ai.z += r.z * mrinv3;
    ai.w -= mrinv;
    return ai;
}


// This is the "tile_calculation" function from the GPUG3 article.

__device__ float4
gravitation(float4 iPos,
            float  iSoft,
            float4 accel,
            float4 *sharedPos,
            float  *sharedSoft,
            float  boxSize,
            float  boxHalf)
{
#pragma unroll 32
    for (unsigned int counter = 0; counter < blockDim.x; counter++)
    {
        accel = bodyBodyInteraction(accel, iPos, sharedPos[counter], iSoft,
                                    sharedSoft[counter], boxSize, boxHalf);
    }
    return accel;
}

// WRAP is used to force each block to start working on a different
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at
// once.
#define WRAP(x,m) (((x)<(m))?(x):((x)-(m)))  // Mod without divide, works on values from 0 up to 2m

//JB, different numbers of i-particles and j-particles incase tree.n_dust and tree.n are unequal
KERNEL_DECLARE(dev_direct_gravity)(float4 *accel, float4 *i_positions, float4 *j_positions,
                                    int numBodies_i, int numBodies_j,
                                    const float *i_soft, const float *j_soft, float boxSize)
{
    // Shared tile now carries the source softening alongside the position.
    extern __shared__ float4 shMem[];
    float4 *sharedPos  = shMem;
    float  *sharedSoft = (float *)&shMem[blockDim.x];
    const float boxHalf = 0.5f * boxSize;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    sharedPos[threadIdx.x]  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    sharedSoft[threadIdx.x] = 0.0f;

    if (index >= numBodies_i)
    {
        return;
    }

    float4 iPos  = i_positions[index];
    const float iSoft = i_soft ? i_soft[index] : 0.0f;

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    int p        = blockDim.x;
    int n        = numBodies_j;
    int numTiles = (n + p - 1) / p;


    for (int tile = blockIdx.y; tile < numTiles + blockIdx.y; tile++)
    {
        int jindex = WRAP(blockIdx.x + tile, gridDim.x) * p + threadIdx.x;
        if (jindex < numBodies_j)
        {
          sharedPos[threadIdx.x]  = j_positions[jindex];
          sharedSoft[threadIdx.x] = j_soft ? j_soft[jindex] : 0.0f;
        }
        else
        {
          sharedPos[threadIdx.x]  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);   // zero mass: inert
          sharedSoft[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation(iPos, iSoft, acc, sharedPos, sharedSoft, boxSize, boxHalf);

        __syncthreads();
    }

    accel[index] = acc;   // .w now carries the potential, as the tree path does
}
