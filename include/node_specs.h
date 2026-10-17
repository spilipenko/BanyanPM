#ifndef _NODE_SPECS_H_
#define _NODE_SPECS_H_

typedef unsigned int uint;

typedef float real;
typedef float4 real4;

// Moved here from octree.h (a host-only header) so that the kernel declarations in
// devFunctionDefinitions.h can name the SAME type the kernels are defined with. That header
// previously carried a duplicate called setupParams2, which made three scanKernels declarations
// disagree with their definitions by type while being layout-compatible.
typedef struct setupParams {
  int jobs;                     //Minimal number of jobs for each 'processor'
  int blocksWithExtraJobs;      //Some ' processors'  do one extra job all with bid < bWEJ
  int extraElements;            //The elements that didn't fit completely
  int extraOffset;              //Start of the extra elements
} setupParams;

//Dont uncomment this yet
#define DO_BLOCK_TIMESTEP

//Enabling the following increases the number of particle properties
//exchanged during mpi particle exchange. Only required if you run
//block time steps. Not needed in the default shared time-step mode.
//#define DO_BLOCK_TIMESTEP_EXCHANGE_MPI

//Uncomment the next line to use thrust radix sort instead of built in one
// #define USE_THRUST

#if USE_DUST
  #if USE_MPI
    #error "Fatal, USE DUST does not work when using MPI. Its for demo only"
  #endif
#endif


typedef struct bodyStruct
{
  real4  pos;
  real4  vel;
  real4  acc0;
  real4  Ppos;
  real4  Pvel;
  float2 time;
  unsigned long long id;

#ifdef DO_BLOCK_TIMESTEP_EXCHANGE_MPI

  uint4 key;
  real4 acc1;
#endif
} bodyStruct;




#define IMPBH   //Improved barnes hut opening method
//#define INDSOFT //Individual softening using cubic spline kernel

//Phase 3: Springel(2005)/Gadget-2 relative acceleration opening criterion, as a compile-time
//alternative to the improved-BH opening-angle criterion (IMPBH) always used for the group's AABB
//distance test. _MAC_SPRINGEL_ is set via the CMake option MAC_SPRINGEL (CMakeLists.txt), not
//defined here directly, so both variants can be configured into separate build dirs
//(build-release/ vs build-springel/) and compared without editing source between builds. Only
//exercises the *local* tree-walk kernel (dev_approximate_gravity); the LET remote-walk kernel
//(dev_approximate_gravity_let, MPI>1 only) is not yet wired to this MAC -- see AMD_PORT_PLAN.md
//Phase 3 for why that's an explicit, documented scope cut for now.
// 0.005, paired with ErrTolTheta=0.5, is what every example .param file in Gadget-2.0.7/Gadget2/
// parameterfiles/ actually uses -- confirmed by grepping them directly, not assumed. (An earlier
// version of this constant was 0.025, mislabeled as "Gadget-2's own default" without having
// checked; it wasn't grounded in anything in the actual Gadget-2 source. Bonsai's own `theta`
// CLI default of 0.75, main.cpp, is a separate, legitimate Bonsai-native convention, not a
// Gadget-2 one -- left as-is, just don't confuse the two when picking "matching" values.)
#define ERR_TOL_FORCE_ACC 0.005f  //Delta_acc in the Springel(2005) MAC formula

//Tree-walk and stack configuration
//#define LMEM_STACK_SIZE            3072         //Number of storage places PER thread, MUST be power 2 !!!!
// T61: 2048 -> 8192, i.e. the walk's per-warp cell-list ring buffer goes from 65,536 to 262,144
// entries. 2048 was not enough to reach z=0 on a 130M-particle zoom: the ama-gi-1 run overflowed at
// iteration 14288 (z=2.41, snapshot 122 of 682) with a live frontier of 69,307 against the 65,536
// capacity, and the T49-A guard correctly refused to integrate the incomplete forces rather than
// wrapping the ring silently.
//
// Measured at that step, which is what sets the size:
//     frontier 57,233 (87.3% of 65,536) with the pre-T55 spherical Rcut prune
//     frontier 69,307 (105.8%)           with T55's per-axis prune, i.e. the shipping default
// So T55 contributed +21% -- and the run would have failed within a few hundred steps regardless,
// since 87.3% at z=2.41 leaves no room for the remaining 560 snapshots' worth of clustering.
//
// 262,144 is 3.8x the measured peak. No fixed size is provably safe here -- the frontier has a long
// tail (see the audit repo's cell-list-overflow note: a 60x tail against a 20x margin) -- so the
// guard matters more than the number. What this buys is headroom to z=0 at this particle count;
// the guard remains the thing that makes an overflow a clean stop instead of wrong forces.
#define LMEM_STACK_SIZE             8192        //Number of storage places PER thread, MUST be power 2 !!!!
#define LMEM_EXTRA_SIZE             2048
#define CELL_LIST_MEM_PER_WARP     (LMEM_STACK_SIZE*32)
#if ((CELL_LIST_MEM_PER_WARP-1) & CELL_LIST_MEM_PER_WARP) != 0
#error "CELL_LIST_MEM_PER_WARP must be power of 2"
#endif
//#define LMEM_STACK_SIZE            1024         //Number of storage places PER thread, MUST be power 2 !!!!
//#define LMEM_STACK_SIZE            512         //Number of storage places PER thread, MUST be power 2 !!!!
// #define TREE_WALK_BLOCKS_PER_SM    32           //Number of GPU thread-blocks used for tree-walk
                                                //this is per SM, 8 is ok for Fermi architecture, 16 is save side

//Put this in this file since it is a setting
inline int getTreeWalkBlocksPerSM(int devMajor, int devMinor)
{
  switch(devMajor)
  {
    case 1:
      fprintf(stderr, "Sorry devices with compute capability < 2.0 are not supported \n");
      exit(0);
    case 2:     //Fermi
      return 16;     
    case 3:     //Kepler
      return 32;
    default:    //Future proof...
      return 32;
  }  
}

//Factor of extra memory we allocate during multi-GPU runs. By allocating a bit extra
//we reduce the number of memory allocations when particle numbers fluctuate. 1.1 == 10% extra
#define MULTI_GPU_MEM_INCREASE 1.1

//If USE_HASH_TABLE_DOMAIN_DECOMP is set to 1 we build a hash-table, otherwise we use
//sampling particles to get an idea of the domain space used to compute the domain
//decomposition
#define USE_HASH_TABLE_DOMAIN_DECOMP 0

//Number of processors to which we exchange the full domain. Also means if nProcs <= this
//number we will always do a full exchange
#define NUMBER_OF_FULL_EXCHANGE 16


#define TEXTURE_BOUNDARY  512   //Fermi architecture boundary for textures

#define MAXLEVELS 30

//Minimum number of nodes that is required  before we make leafs
#define START_LEVEL_MIN_NODES 16

// T74: BITLEVELS / ILEVELMASK / LEVELMASK are GONE, deliberately.
//
// node_bodies[].x used to pack the tree level into its top 5 bits and the index of the node's
// first body into the bottom 27, which capped the whole code at 2^27 = 134,217,728 particles --
// exactly 512^3, the one production size this port had ever run, so it sat precisely on the limit
// and nobody noticed. Above it the index truncated AND overwrote the level, and the corrupted
// level then indexed level_list[level-1] past its MAXLEVELS entries: measured, 226,492,416
// particles exited 0 with the clock frozen at TimeBegin and 339,738,624 took a GPU page fault.
//
// node_bodies is now a uint4 with the level in its own lane (.z) and the index at full width in
// .x. These three macros are removed rather than left unused so that any site still reaching for
// the old packing FAILS TO COMPILE instead of silently reading the wrong bits.

#ifndef NLEAF
#define NLEAF 16
#endif
// Default flipped 64->32 (AMD_PORT_PLAN.md Phase 3 §13/§14): measured on real gfx1151 hardware via
// rocprofv3 to be ~15% faster (dev_approximate_gravity: 260.8us vs 306.0us mean kernel duration,
// 21-dispatch sample) with no accuracy penalty vs. N^2 direct summation. The speedup has an
// identified cause, not just a correlation: NCRIT==32 exactly matches the confirmed wave32 width
// (§1/Phase 0), so approximate_gravity's NI=(nb_i<=WARP_SIZE)?1:2 dispatch always takes the
// single-particle-per-lane path (NCRIT=64 forced most groups down the dual-particle NI=2 path,
// costing 20% more VGPR/wave: 120 vs 96). Overridable via the CMake option GADGET_HIP_NCRIT
// (CMakeLists.txt -> -DNCRIT=<value>) so 64 (or any other value) stays available as an explicit,
// no-source-edit build option -- see build-ncrit64/ -- same convention as MAC_SPRINGEL.
// T74: CRITBIT / CRITMASK / INVCMASK are GONE, deliberately -- same reason as
// BITLEVELS/ILEVELMASK/LEVELMASK above.
//
// groupSizeInfo[].w used to pack a group's particle count minus one into its top
// 32-log2(NCRIT) bits and the index of its first particle into the rest, which at the shipping
// NCRIT=32 left the index 27 bits and so capped the run at 2^27 = 134,217,728 particles --
// exactly 512^3. Above it the index truncated and the count was corrupted by the spill, and the
// walk integrated the wrong particles: measured, 226,492,416 particles exited 0 with every
// timestep zero and the clock frozen at TimeBegin.
//
// The count is now DERIVED as start[g+1] - start[g] and .w holds the index at full 32-bit width.
// That is sound because groups tile [0,n) contiguously in group-index order, which
// GADGET_HIP_INVARIANTS=1's C-A-19b check verifies on every rebuild -- measured OK on a uniform
// 64^3 IC and on the 130,341,198-particle four-species ama-gi zoom. Both group-construction paths
// guarantee it: store_group_list() emits consecutive compacted boundaries, and the experimental
// GADGET_HIP_COMPACT_GROUPS path chains them explicitly (build.cpp).
//
// groupSizeInfo therefore carries ONE EXTRA entry, [n_groups], holding n as a sentinel so the
// derivation needs no branch for the last group.
//
#ifndef NCRIT
#define NCRIT 32
#endif
#ifndef NTHREAD
#define NTHREAD 128
#endif

#define NLEAFTEST 8

#if NLEAF == 1 || NLEAF == 2
// Both share the NLEAF==2 bit layout (1 bit for count-1 in {0,1}) rather than giving NLEAF==1 its
// own 0-bit / LEAFBIT==32 encoding -- a 32-bit shift by 32 is undefined behavior in C/C++, and
// since a leaf built with NLEAF==1 only ever holds exactly 1 particle, its count-1 field is always
// 0 regardless of how many bits are reserved for it, so reusing the already-safe NLEAF==2 layout
// costs nothing and avoids a genuinely new, untested edge case (2026-08-30, user's de-grouping
// experiment, PLAN.md's Close-encounter root-cause audit / LOG.md Sec 48-51).

#define NLEAF2 1
#define LEAFBIT 31
#define BODYMASK 0x7FFFFFFF
#define INVBMASK 0x80000000

#elif NLEAF == 8

#define NLEAF2 3
#define LEAFBIT 29
#define BODYMASK 0x1FFFFFFF
#define INVBMASK 0xE0000000

#elif NLEAF == 16

#define NLEAF2 4
#define LEAFBIT 28
#define BODYMASK 0x0FFFFFFF
#define INVBMASK 0xF0000000

#elif NLEAF == 32

#define NLEAF2 5
#define LEAFBIT 27
#define BODYMASK 0x07FFFFFF
#define INVBMASK 0xF8000000

#elif NLEAF == 64

#define NLEAF2 6
#define LEAFBIT 26
#define BODYMASK 0x03FFFFFF
#define INVBMASK 0xFC000000

#elif NLEAF == 128

#define NLEAF2 7
#define LEAFBIT 25
#define BODYMASK 0x01FFFFFF
#define INVBMASK 0xFE000000

#else
#error "Please choose correct NLEAF available in node_specs.h"
#endif


#if NCRIT == 1 || NCRIT == 2
// Same rationale as NLEAF==1/2 above: both share the NCRIT==2 bit layout (1 bit for count-1 in
// {0,1}) rather than a separate 0-bit / CRITBIT==32 encoding for NCRIT==1, which would be
// undefined behavior. A group built with NCRIT==1 only ever holds exactly 1 particle, so its
// count-1 field is always 0 regardless of how many bits are reserved for it.

#define NCRIT2 1

#elif NCRIT == 8

#define NCRIT2 3

#elif NCRIT == 16

#define NCRIT2 4

#elif NCRIT == 32

#define NCRIT2 5

#elif NCRIT == 64

#define NCRIT2 6

#elif NCRIT == 128

#define NCRIT2 7

#else
#error "Please choose correct NCRIT available in node_specs.h"
#endif

#if NTHREAD == 8

#define NTHREAD2 3

#elif NTHREAD == 16

#define NTHREAD2 4

#elif NTHREAD == 32

#define NTHREAD2 5

#elif NTHREAD == 64

#define NTHREAD2 6

#elif NTHREAD == 96
// REFUSED, deliberately. NTHREAD2 must be exactly log2(NTHREAD): the tree walk sizes its shared
// memory as (1 << BLOCKDIM2) and derives its warp count as BLOCKDIM2 - WARP_SIZE2
// (dev_approximate_gravity_warp_new.cu). 96 is not a power of two, so no NTHREAD2 is correct for
// it -- the value that used to sit here, 7, is log2(128), which makes the walk reduce over four
// warps when the block has three. That is a wrong answer, not a slow one.
#error "NTHREAD=96 is not supported: NTHREAD must be a power of two (8, 16, 32, 64, 128, 256)."

#elif NTHREAD == 128

#define NTHREAD2 7

#elif NTHREAD == 256

#define NTHREAD2 8

#else
#error "Please choose correct NTHREAD available in node_specs.h"
#endif





#if NCRIT < NLEAF
#error "Fatal, NCRIT < NLEAF. Please check that NCRIT >= NLEAF"
#endif

#endif /* _NODE_SPECS_H_ */

// BONSAI_DENSITY -- the SPH-style density / neighbour-count subsystem inherited from Bonsai.
//
// It has NO effect on gravity in this port. Force softening is read from bodies_forceSoftening
// (dev_approximate_gravity_warp_new.cu:1787), not from bodies_h; bodies_h and bodies_dens feed
// only each other (walk -> bodies_dens -> adjustH -> bodies_h -> walk) plus Bonsai's live-renderer
// quickDump fields p.rho/p.h, which this Gadget-2 port does not use. Verified by full-tree audit
// of every reader of both buffers.
//
// bodies_dens costs a float2 per particle: 1 GB at 512^3, in a 128 GB budget that has to reach
// ~3e8 particles. With this off the buffer is allocated at one element (build.cpp) and the whole
// density chain dead-code-eliminates out of the inner force loop.
//
// To restore: set this to 1. Everything keyed to it is marked `#if BONSAI_DENSITY`.
#define BONSAI_DENSITY 0
