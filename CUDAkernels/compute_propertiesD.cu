#include "hip/hip_runtime.h"
#include "bonsai.h"
#include "support_kernels.cu"
#include <stdio.h>

#include "../profiling/bonsai_timing.h"
PROF_MODULE(compute_propertiesD);

#include "node_specs.h"

// Host-side declarations of the kernels defined below. Included deliberately: it makes the compiler
// compare each declaration against its definition, so a signature that drifts is a build error.
// Without this, the declarations are used only for their address and a wrong one is silent -- which
// is how 25 of them came to diverge (see devFunctionDefinitions.h).
#include "devFunctionDefinitions.h"

static __device__ __forceinline__ void sh_MinMax2(int i, int j, float3 *r_min, float3 *r_max, volatile float3 *sh_rmin, volatile  float3 *sh_rmax)
{
  sh_rmin[i].x  = (*r_min).x = fminf((*r_min).x, sh_rmin[j].x);
  sh_rmin[i].y  = (*r_min).y = fminf((*r_min).y, sh_rmin[j].y);
  sh_rmin[i].z  = (*r_min).z = fminf((*r_min).z, sh_rmin[j].z);
  sh_rmax[i].x  = (*r_max).x = fmaxf((*r_max).x, sh_rmax[j].x);
  sh_rmax[i].y  = (*r_max).y = fmaxf((*r_max).y, sh_rmax[j].y);
  sh_rmax[i].z  = (*r_max).z = fmaxf((*r_max).z, sh_rmax[j].z);
}

//////////////////////////////
//////////////////////////////
//////////////////////////////

//Helper functions for leaf-nodes
static __device__ void compute_monopole(double &mass, double &posx,
                                 double &posy, double &posz,
                                 float4 pos)
{
  mass += pos.w;
  posx += pos.w*pos.x;
  posy += pos.w*pos.y;
  posz += pos.w*pos.z;
}

static __device__ void compute_quadropole(double &oct_q11, double &oct_q22, double &oct_q33,
                                   double &oct_q12, double &oct_q13, double &oct_q23,
                                   float4 pos)
{
  oct_q11 += pos.w * pos.x*pos.x;
  oct_q22 += pos.w * pos.y*pos.y;
  oct_q33 += pos.w * pos.z*pos.z;
  oct_q12 += pos.w * pos.x*pos.y;
  oct_q13 += pos.w * pos.y*pos.z;
  oct_q23 += pos.w * pos.z*pos.x;
}

static __device__ void compute_bounds(float3 &r_min, float3 &r_max,
                               float4 pos)
{
  r_min.x = fminf(r_min.x, pos.x);
  r_min.y = fminf(r_min.y, pos.y);
  r_min.z = fminf(r_min.z, pos.z);

  r_max.x = fmaxf(r_max.x, pos.x);
  r_max.y = fmaxf(r_max.y, pos.y);
  r_max.z = fmaxf(r_max.z, pos.z);
}

//Non-leaf node helper functions
static __device__ void compute_monopole_node(double &mass, double &posx,
                                 double &posy, double &posz,
                                 double4  pos)
{
  mass += pos.w;
  posx += pos.w*pos.x;
  posy += pos.w*pos.y;
  posz += pos.w*pos.z;
}


static __device__ void compute_quadropole_node(double &oct_q11, double &oct_q22, double &oct_q33,
                                        double &oct_q12, double &oct_q13, double &oct_q23,
                                        double4 Q0, double4 Q1)
{
  oct_q11 += Q0.x;
  oct_q22 += Q0.y;
  oct_q33 += Q0.z;
  oct_q12 += Q1.x;
  oct_q13 += Q1.y;
  oct_q23 += Q1.z;
}

static __device__ void compute_bounds_node(float3 &r_min, float3 &r_max,
                                    float4 node_min, float4 node_max)
{
  r_min.x = fminf(r_min.x, node_min.x);
  r_min.y = fminf(r_min.y, node_min.y);
  r_min.z = fminf(r_min.z, node_min.z);

  r_max.x = fmaxf(r_max.x, node_max.x);
  r_max.y = fmaxf(r_max.y, node_max.y);
  r_max.z = fmaxf(r_max.z, node_max.z);
}


KERNEL_DECLARE(compute_leaf)( const int n_leafs,
                              uint *leafsIdxs,
                              uint2 *node_bodies,
                              real4 *body_pos,
                              double4 *multipole,
                              real4 *nodeLowerBounds,
                              real4 *nodeUpperBounds,
                              real4  *body_vel,
                              ulonglong1 *body_id,
			      ::real  *body_h, 
			      const float h_min,
			      const float *body_forceSoftening,  // T28/C-A-04
			      float  *nodeSoftInfo) {            // T28/C-A-04

  CUXTIMER("compute_leaf");
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint id  = bid * blockDim.x + tid;


  volatile __shared__ float3 shmem[256];
  volatile float3 *sh_rmin = (float3*)&shmem [ 0];
  volatile float3 *sh_rmax = (float3*)&shmem[128];

  //Set the shared memory for these threads and exit the thread
  if (id >= n_leafs)
  {
    sh_rmin[tid].x = +1e10f; sh_rmin[tid].y = +1e10f; sh_rmin[tid].z = +1e10f;
    sh_rmax[tid].x = -1e10f; sh_rmax[tid].y = -1e10f; sh_rmax[tid].z = -1e10f;
    return;
  }


  //Since nodes are intermixes with non-leafs in the node_bodies array
  //we get a leaf-id from the leafsIdxs array
  int nodeID = leafsIdxs[id];

  const uint2 bij          =  node_bodies[nodeID];
  const uint firstChild    =  bij.x & ILEVELMASK;
  const uint lastChild     =  bij.y;  //TODO maybe have to increase it by 1

  //Variables holding properties and intermediate answers
  float4 p;

  double mass, posx, posy, posz;
  mass = posx = posy = posz = 0.0;

  double oct_q11, oct_q22, oct_q33;
  double oct_q12, oct_q13, oct_q23;

  oct_q11 = oct_q22 = oct_q33 = 0.0;
  oct_q12 = oct_q13 = oct_q23 = 0.0;
  float3 r_min, r_max;
  r_min = make_float3(+1e10f, +1e10f, +1e10f);
  r_max = make_float3(-1e10f, -1e10f, -1e10f);

  //Loop over the children=>particles=>bodys
  //unroll increases register usage #pragma unroll 16
  float maxEps = -100.0f;
  float minEps = 3.4e38f;   // T28/C-A-04
  int count=0;
  for(int i=firstChild; i < lastChild; i++)
  {
    p      = body_pos[i];
    // INDEX-SPACE MISMATCH, verified and currently inert -- see buffer_registry.h.
    // `i` runs over node_bodies, which is built AFTER sort_bodies, so it is a CURRENT-order
    // particle index. body_vel is tree.bodies_Pvel (compute_properties.cpp:77), which the
    // default per-step sort does NOT permute, so it is ORIGINAL-order: this reads a DIFFERENT
    // particle's eps. Every other array in this kernel (body_pos=bodies_Ppos, body_id, body_h)
    // IS current-order, which is what hides it.
    //
    // Inert on two independent counts, both checked rather than assumed:
    //  1. The value is dead. maxEps -> Q0.w (:177), propagated by compute_non_leaf (:339), and
    //     the ONLY consumer -- the quadrupole path in dev_approximate_gravity_warp_new.cu:819 --
    //     rebuilds jQ0 with .w explicitly set to 0.0f. The monopole path (the default) never
    //     reads Q0 at all. This is contract C-A-04's "computed but never consumed", confirmed.
    //  2. Every IC in this project is single-species, so all eps values are equal anyway.
    //
    // Deliberately NOT changed here: the correct fix depends on reviving per-node max softening
    // for Gadget-2's mixed-softening forced-descent rule (contract C-C-08, currently ABSENT).
    // Whoever implements C-C-08 must fix this first, or it will silently feed that rule the
    // wrong particle's softening. The clean fix is to read tree.bodies_forceSoftening, which is
    // CURRENT-order and is the real Gadget-2 per-particle value, rather than Bonsai's legacy
    // "eps lives in vel.w" convention.
    // T28/C-A-04: read the REAL Gadget-2 per-particle softening, in the same CURRENT (sorted)
    // order this loop indexes body_pos by. The legacy `body_vel[i].w` source is not merely in the
    // wrong index space -- no IC path in this port ever writes a softening there, so it was
    // identically 0.0f for every node in every run. `min` is carried alongside `max` because
    // Gadget's forced-descent test keys on diffsoftflag ("the members differ"), and over any set
    // "some contributor differs from the running max" <=> max != min.
#ifdef UNEQUALSOFTENINGS
    const float epsI = body_forceSoftening ? body_forceSoftening[i] : 0.0f;
    maxEps = fmaxf(epsI, maxEps);
    minEps = fminf(epsI, minEps);
#endif
    count++;
    compute_monopole(mass, posx, posy, posz, p);
    compute_quadropole(oct_q11, oct_q22, oct_q33, oct_q12, oct_q13, oct_q23, p);
    compute_bounds(r_min, r_max, p);
  }

  double4 mon = {posx, posy, posz, mass};

  double im = 1.0/mon.w;
  if(mon.w == 0) im = 0;        //Allow tracer/massless particles
  mon.x *= im;
  mon.y *= im;
  mon.z *= im;

  //jbedorf, store darkMatterMass in Q1.w
  //stellar mass is mon.w-darkMatterMass

  double4 Q0, Q1;
  Q0 = make_double4(oct_q11, oct_q22, oct_q33, maxEps); //Store max softening
  Q1 = make_double4(oct_q12, oct_q13, oct_q23, 0.0f);

  // T28/C-A-04: encode (maxSoft, mixed) into one float. 0.0f = no contributor (Gadget's
  // maxsofttype==7 sentinel -> the walk must force-open). Negative = members differ.
#ifdef UNEQUALSOFTENINGS
  if (nodeSoftInfo)
  {
    const bool  mixed = (count > 0) && (maxEps != minEps);
    const float summary = (count > 0 && maxEps > 0.0f) ? (mixed ? -maxEps : maxEps) : 0.0f;
    nodeSoftInfo[nodeID] = summary;
  }
#endif

  //Store the leaf properties
  multipole[3*nodeID + 0] = mon;       //Monopole
  multipole[3*nodeID + 1] = Q0;        //Quadropole
  multipole[3*nodeID + 2] = Q1;        //Quadropole

  //Store the node boundaries
  nodeLowerBounds[nodeID] = make_float4(r_min.x, r_min.y, r_min.z, 0.0f);
  nodeUpperBounds[nodeID] = make_float4(r_max.x, r_max.y, r_max.z, 1.0f);  //4th parameter is set to 1 to indicate this is a leaf



#if 0
  //Addition for density computation, we require seperate search radii
  //for star particles and dark-matter particles. Note that we could 
  //combine most of this with the loops above. But for clarity
  //I kept them seperate untill it all is tested and functions correctly

  ulonglong1 DARKMATTERID;
  DARKMATTERID.x = 3000000000000000000ULL;

  float3 r_minS, r_maxS, r_minD, r_maxD;
  r_minS = make_float3(+1e10f, +1e10f, +1e10f);
  r_maxS = make_float3(-1e10f, -1e10f, -1e10f);
  r_minD = make_float3(+1e10f, +1e10f, +1e10f);
  r_maxD = make_float3(-1e10f, -1e10f, -1e10f);

  int nStar = 0;
  int nDark = 0;
  for(int i=firstChild; i < lastChild; i++)
  {
    p             = body_pos[i];
    ulonglong1 id = body_id[i];

    if(id.x >= DARKMATTERID.x)
    {
      compute_bounds(r_minD, r_maxD, p);
      nDark++;
    }
    else
    {

	compute_bounds(r_minS, r_maxS, p);
	nStar++;
    }
  }
  
  float fudgeFactor = 1;

  r_maxS.x -= r_minS.x;  r_maxS.y -= r_minS.y;  r_maxS.z -= r_minS.z;
  r_maxD.x -= r_minD.x;  r_maxD.y -= r_minD.y;  r_maxD.z -= r_minD.z;

#if 0
  float volumeS = fudgeFactor*cbrtf(r_maxS.x*r_maxS.y*r_maxS.z); //pow(x,1.0/3);
  float volumeD = fudgeFactor*cbrtf(r_maxD.x*r_maxD.y*r_maxD.z); //, 1.0/3);
#else
  const float maxS = max(max(r_maxS.x, r_maxS.y), r_maxS.z);
  const float maxD = max(max(r_maxD.x, r_maxD.y), r_maxD.z);
  const float volS = maxS*maxS*maxS;
  const float volD = maxD*maxD*maxD;
  const int   npS  = lastChild - firstChild;
  const int   npD  = npS;  /* must have correct count for npS & npD */
  const int   nbS = 32;
  const int   nbD = 64;
  const float roS = float(npS) / volS;
  const float roD = float(npD) / volD;
  const float volumeS = cbrtf(nbS / roS);
  const float volumeD = cbrtf(nbD / roD);
//  assert(volumeS >= 0.0f);
//  assert(volumeD >= 0.0f);
#endif

#if BONSAI_DENSITY
  for(int i=firstChild; i < lastChild; i++)
  {
    ulonglong1 id = body_id[i];
    if(id.x >= DARKMATTERID.x)
    {
//	    assert(0);
	    body_h[i] = volumeD;
    }
    else
    {
//	    if(i == 0) printf("STATD I goes from: %f  to %f gives: %f \n", body_h[i], volumeS, 0.5f*(volumeS + body_h[i]));
	    if(body_h[i] >= 0) 
	      body_h[i] = body_h[i]; // 0.5f*(volumeS + body_h[i]);
	    else
	      body_h[i] = max(h_min,volumeS);
    }
  }
#endif
#else
  {
    const float3 len = make_float3(r_max.x-r_min.x, r_max.y-r_min.y, r_max.z-r_min.z);
    const float  vol = cbrtf(len.x*len.y*len.z);
    float hp  = 0;
    if (vol > 0.0f)
    {
      const float nd  = float(lastChild - firstChild) / vol;
      hp  = cbrtf(42.0f / nd);
    }
    hp = max(hp, h_min);
#if BONSAI_DENSITY
    for(int i=firstChild; i < lastChild; i++)
      if(body_h[i] < 0)
        body_h[i] = hp;
#endif
  }
#endif


  return;
}


//Function goes level by level (starting from deepest) and computes
//the properties of the non-leaf nodes
KERNEL_DECLARE(compute_non_leaf)(const int curLevel,         //Level for which we calc
                                            uint  *leafsIdxs,           //Conversion of ids
                                            uint  *node_level_list,     //Contains the start nodes of each lvl
                                            uint  *n_children,          //Reference from node to first child and number of childs
                                            double4 *multipole,
                                            real4 *nodeLowerBounds,
                                            real4 *nodeUpperBounds,
                                            float *nodeSoftInfo) {   // T28/C-A-04

  CUXTIMER("compute_non_leaf");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  const uint idx = bid * (blockDim.x * blockDim.y) + tid;

  const int endNode   = node_level_list[curLevel];
  const int startNode = node_level_list[curLevel-1];


  if(idx >= (endNode-startNode))     return;

  const int nodeID = leafsIdxs[idx + startNode];

  //Get the children info
  const uint firstChild = n_children[nodeID] & 0x0FFFFFFF;                  //TODO make this name/define?
  const uint nChildren  = ((n_children[nodeID]  & 0xF0000000) >> 28); //TODO make this name/define?

  //Variables
  double mass, posx, posy, posz;
  mass = posx = posy = posz = 0.0;

  double oct_q11, oct_q22, oct_q33;
  double oct_q12, oct_q13, oct_q23;

  oct_q11 = oct_q22 = oct_q33 = 0.0;
  oct_q12 = oct_q13 = oct_q23 = 0.0;

  float3 r_min, r_max;
  r_min = make_float3(+1e10f, +1e10f, +1e10f);
  r_max = make_float3(-1e10f, -1e10f, -1e10f);

  //Process the children (1 to 8)
  float maxEps = -100.0f;
  float childSoftMax = -1.0f, childSoftMin = 3.4e38f;   // T28/C-A-04
  bool  childMixed = false;
  int   nChildSoft = 0;
  for(int i=firstChild; i < firstChild+nChildren; i++)
  {
    //Gogo process this data!
    double4 tmon = multipole[3*i + 0];

    maxEps = max(multipole[3*i + 1].w, maxEps);

    // T28/C-A-04: combine children's (maxSoft, mixed) summaries. Gadget sets diffsoftflag when a
    // contributor differs from the running max AND ORs in each child's inherited bit
    // (forcetree.c:515, :525-536, :566-575). At this level the min over child MAXES is not the min
    // over all members, so `OR(child.mixed)` is required as well as `max != min`: e.g. child A =
    // {1,3} and child B = {3} have equal maxes but A is internally mixed.
#ifdef UNEQUALSOFTENINGS
    if (nodeSoftInfo)
    {
      const float cs = nodeSoftInfo[i];
      if (cs != 0.0f)                       // 0 = no contributor; skip it from the max, per Gadget
      {
        const float cmax = fabsf(cs);
        childSoftMax = fmaxf(childSoftMax, cmax);
        childSoftMin = fminf(childSoftMin, cmax);
        nChildSoft++;
      }
      if (cs < 0.0f) childMixed = true;     // inherited diffsoftflag
    }
#endif

    compute_monopole_node(mass, posx, posy, posz, tmon);
    compute_quadropole_node(oct_q11, oct_q22, oct_q33, oct_q12, oct_q13, oct_q23,
                            multipole[3*i + 1], multipole[3*i + 2]);
    compute_bounds_node(r_min, r_max, nodeLowerBounds[i], nodeUpperBounds[i]);
  }

  //Save the bounds
  nodeLowerBounds[nodeID] = make_float4(r_min.x, r_min.y, r_min.z, 0.0f);
  nodeUpperBounds[nodeID] = make_float4(r_max.x, r_max.y, r_max.z, 0.0f); //4th is set to 0 to indicate a non-leaf

  //Regularize and store the results
  double4 mon = {posx, posy, posz, mass};
  double im = 1.0/mon.w;
  if(mon.w == 0) im = 0; //Allow tracer/massless particles

  mon.x *= im;
  mon.y *= im;
  mon.z *= im;

  double4 Q0, Q1;
  Q0 = make_double4(oct_q11, oct_q22, oct_q33, maxEps); //store max Eps
  Q1 = make_double4(oct_q12, oct_q13, oct_q23, 0.0f);

  // T28/C-A-04: this node is mixed if any child is, or if the children's maxes disagree.
#ifdef UNEQUALSOFTENINGS
  if (nodeSoftInfo)
  {
    const bool  mixed   = childMixed || (nChildSoft > 1 && childSoftMax != childSoftMin);
    const float summary = (nChildSoft > 0) ? (mixed ? -childSoftMax : childSoftMax) : 0.0f;
    nodeSoftInfo[nodeID] = summary;
  }
#endif

  multipole[3*nodeID + 0] = mon;        //Monopole
  multipole[3*nodeID + 1] = Q0;         //Quadropole1
  multipole[3*nodeID + 2] = Q1;         //Quadropole2

  return;
}
// T9 (contract C-B-10b): use the fixed octree cell side instead of the particle-fitted AABB
// extent as the node-size term in the geometric opening criterion. 0 = legacy AABB.
__constant__ int g_t9_cellsize_geo = 0;
__constant__ int g_t9_no_s = 0;

void t9_set_cellsize_geo(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t9_cellsize_geo), &on, sizeof(int));
  fprintf(stderr, "[T9-CELLSIZE-GEO] geometric MAC node size = %s\n",
          on ? "fixed octree cell (Gadget-2 len)" : "AABB extent (legacy)");
}

void t9_set_no_s(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_t9_no_s), &on, sizeof(int));
  fprintf(stderr, "[T9-NO-S] geometric MAC centre-of-mass offset term = %s\n",
          on ? "REMOVED (Gadget-2 exact)" : "present (Barnes IMPBH)");
}

// C-A-01(b) PERF: skip the fixed-cell-centre computation entirely. Not a physics option -- it
// leaves cellCenterInfo unwritten -- but it lets ONE binary measure that block's cost against
// itself, instead of comparing two separately-configured builds (PLAN.md rule 8: diff the build
// configuration before interpreting any difference between two builds). Pair it with
// GADGET_HIP_CA01_CELL_CENTER=0 so the walk does not consume the stale buffer.
__constant__ int g_ca01_cc_skip = 0;

void ca01_set_cc_skip(int on)
{
  hipMemcpyToSymbol(HIP_SYMBOL(g_ca01_cc_skip), &on, sizeof(int));
  if (on) fprintf(stderr, "[CA01-CC-SKIP] fixed-cell-centre computation DISABLED (timing control)\n");
}

KERNEL_DECLARE(compute_scaling)(const int node_count,
                                           double4 *multipole,
                                           real4 *nodeLowerBounds,
                                           real4 *nodeUpperBounds,
                                           uint  *n_children,
                                           real4 *multipoleF,
                                           float theta,
                                           real4 *boxSizeInfo,
                                           real4 *boxCenterInfo,
                                           uint2 *node_bodies,
                                           float  domain_fac,
                                           float  *cellSizeInfo,
                                           real4  corner,          //C-A-01(b): tree.corner
                                           real4  *cellCenterInfo){

  CUXTIMER("compute_scaling");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  const uint idx = bid * (blockDim.x * blockDim.y) + tid;

  if(idx >= node_count)     return;

  // T26: fixed octree-cell side length for this node, matching Gadget-2's own fixed-cube `len`
  // semantics -- independent of how tightly the node's actual particles happen to be clustered
  // (unlike boxSizeInfo below, which is a particle-fitted AABB and is the quantity Ticket 24 found
  // can shrink pathologically for a tightly-clustered node). Each node's own build-time octree
  // depth ("level") is already stored in node_bodies[idx].x's top BITLEVELS..31 bits (written once,
  // unconditionally, by build_tree.cu's cl_build_nodes -- the same field build_tree.cu's own
  // level-decode logic reads via `(bij.x & LEVELMASK) >> BITLEVELS`), so this is a cheap, exact
  // decode, not an approximation: the root domain box (tree.corner/tree.domain_fac, sort_bodies_gpu.cpp)
  // is subdivided by exactly 1 bit/axis (3 bits total) per level down to MAXLEVELS -- see
  // support_kernels.cu's get_key()/get_mask() -- so a node at depth `level` has fixed cube side
  // domain_fac * 2^(MAXLEVELS-level), with no dependency on occupancy. Computed here (not derived
  // later, at kernel-walk time) since domain_fac/node level are both known up front and this is a
  // one-time O(n_nodes) cost identical in kind to boxSizeInfo/boxCenterInfo's own computation below.
  {
    const uint  bijLevel   = (node_bodies[idx].x & LEVELMASK) >> BITLEVELS;
    const uint  shiftLevel = (bijLevel <= (uint)MAXLEVELS) ? (uint)MAXLEVELS - bijLevel : 0u;
    cellSizeInfo[idx] = domain_fac * (float)(1u << shiftLevel);
  }

  double4 monD, Q0, Q1;

  monD = multipole[3*idx + 0];        //Monopole
  Q0   = multipole[3*idx + 1];        //Quadropole1
  Q1   = multipole[3*idx + 2];        //Quadropole2

  //Scale the quadropole
  double im = 1.0 / monD.w;
  if(monD.w == 0) im = 0;               //Allow tracer/massless particles
  Q0.x = Q0.x*im - monD.x*monD.x;
  Q0.y = Q0.y*im - monD.y*monD.y;
  Q0.z = Q0.z*im - monD.z*monD.z;
  Q1.x = Q1.x*im - monD.x*monD.y;
  Q1.y = Q1.y*im - monD.y*monD.z;
  Q1.z = Q1.z*im - monD.x*monD.z;

  //Switch the y and z parameter
  double temp = Q1.y;
  Q1.y = Q1.z; Q1.z = temp;

  //Convert the doubles to floats
  float4 mon            = make_float4(monD.x, monD.y, monD.z, monD.w);
  multipoleF[3*idx + 0] = mon;
  multipoleF[3*idx + 1] = make_float4(Q0.x, Q0.y, Q0.z, Q0.w);        //Quadropole1
  multipoleF[3*idx + 2] = make_float4(Q1.x, Q1.y, Q1.z, Q1.w);        //Quadropole2

  float4 r_min, r_max;
  r_min = nodeLowerBounds[idx];
  r_max = nodeUpperBounds[idx];

  //Compute center and size of the box

  float3 boxCenter;
  boxCenter.x = 0.5*(r_min.x + r_max.x);
  boxCenter.y = 0.5*(r_min.y + r_max.y);
  boxCenter.z = 0.5*(r_min.z + r_max.z);

  float3 boxSize = make_float3(fmaxf(fabs(boxCenter.x-r_min.x), fabs(boxCenter.x-r_max.x)),
                               fmaxf(fabs(boxCenter.y-r_min.y), fabs(boxCenter.y-r_max.y)),
                               fmaxf(fabs(boxCenter.z-r_min.z), fabs(boxCenter.z-r_max.z)));

  // C-A-01 part (b): the node's FIXED octree-cell centre -- Gadget-2's `nop->center`.
  //
  // Not the AABB midpoint (`boxCenter` above). Gadget-2 sets `center` once, geometrically, when the
  // node is created (forcetree.c:190-206) and never refits it; the AABB midpoint moves inside the
  // cell as the node's particles cluster, the same way boxSizeInfo's extent shrinks. Both errors
  // are of the same origin and the proximity/safety box (`fabs(center[a]-pos) < 0.6*len`) reads
  // both, so fixing only the size left the box centred on the wrong point.
  //
  // Computed from the node's own GRID INDEX, not by snapping a particle-derived point onto a grid.
  // The index is what the octree is built on, so this returns the cell the tree actually assigned
  // rather than a reconstruction that has to be argued to agree with it. Two consequences:
  //   - no double precision anywhere. An earlier version divided a position by the cell size, a
  //     ratio reaching ~1e9 at 512^3, which needed double to land on the right cell. Shifting the
  //     integer coordinate instead keeps every quantity small: the index is < 2^level, exact in
  //     float for every depth this tree reaches.
  //   - it is correct for a node whose AABB is degenerate or whose members are all coincident,
  //     where a midpoint-based derivation is only as good as the AABB.
  //
  // The half-cell offset is not incidental. The key quantisation is `roundf`, not truncation
  // (build_tree.cu:253-256, EXACT_KEY is not defined -- contract C-A-16), so the fine cell with
  // index k covers [corner + (k-0.5)*domain_fac, corner + (k+0.5)*domain_fac): the whole grid sits
  // half a FINE cell below a floor-based one. A node at shift s therefore centres on
  // corner + (C + 0.5)*cellSize - 0.5*domain_fac. Dropping that last term picks the neighbouring
  // cell for any node whose contents lie within half a fine cell of a boundary.
#ifdef _MAC_SPRINGEL_
  if (!g_ca01_cc_skip)
  {
    const float cell = cellSizeInfo[idx];

    // The index, then the centre -- the cell's own arithmetic, with no reference to how tightly
    // its particles happen to sit inside it.
    //
    // Divide by the CELL, not by domain_fac. (mid - corner)/cell is at most 2^level, exact in
    // float at every depth this tree reaches. The first version of this code divided by
    // domain_fac, a ratio reaching ~1e9 at 512^3, and then reached for double to repair a
    // precision problem it had created itself; a later version shifted the fine-grid integer
    // instead, which forms that same ~1e9 ratio in float and is off by one cell for nodes whose
    // seed coordinate lands near a boundary. Both are gone.
    //
    // The half-fine-cell term: key quantisation is `roundf`, not truncation (build_tree.cu:253-256,
    // EXACT_KEY undefined -- contract C-A-16), so fine cell k covers
    // [corner + (k-0.5)*domain_fac, corner + (k+0.5)*domain_fac) and the whole grid sits half a
    // FINE cell below a floor-based one. Included for consistency with the grid the tree is keyed
    // on, and applied to BOTH the index and the centre so the two agree. It is worth being precise
    // about its size: 0.5*domain_fac is 4.7e-08 at 512^3, about 200x BELOW one float32 ULP of a
    // coordinate near the box scale, so it changes no stored value there -- measured, three ways,
    // identical results with and without it. It matters only if this tree is ever built on a box
    // where domain_fac is not negligible against coordinate precision.
  #ifndef EXACT_KEY
    const float halfFine = 0.5f * domain_fac;
  #else
    const float halfFine = 0.0f;   // truncation: cells start on the grid, no offset
  #endif
    // EXACT_KEY is a build_tree.cu-local define. If it is ever enabled it must be enabled for both
    // translation units; the CA01 audit's cell-centre containment test is what would catch a drift.
    const float ix = floorf((boxCenter.x - corner.x + halfFine) / cell);
    const float iy = floorf((boxCenter.y - corner.y + halfFine) / cell);
    const float iz = floorf((boxCenter.z - corner.z + halfFine) / cell);
    const float ccx = corner.x + (ix + 0.5f) * cell - halfFine;
    const float ccy = corner.y + (iy + 0.5f) * cell - halfFine;
    const float ccz = corner.z + (iz + 0.5f) * cell - halfFine;
    cellCenterInfo[idx] = make_float4(ccx, ccy, ccz, 0.0f);
  }
#else
  (void) corner; (void) cellCenterInfo;   // buffer is size 1 in a geometric-MAC build (build.cpp)
#endif

  //Calculate distance between center of the box and the center of mass
  float3 s3     = make_float3((boxCenter.x - mon.x), (boxCenter.y - mon.y), (boxCenter.z - mon.z));
  double s      = sqrt((s3.x*s3.x) + (s3.y*s3.y) + (s3.z*s3.z));

  //If mass-less particles form a node, the s would be huge in opening angle, make it 0
  if(fabs(mon.w) < 1e-10) s = 0;

  //Length of the box, note times 2 since we only computed half the distance before
  //
  // T9 (contract C-B-10b): Gadget-2's geometric criterion uses `nop->len`, the FIXED octree cell
  // side (forcetree.c:1270), not a particle-fitted extent. The AABB value below is always <= len,
  // so the walk opens FEWER nodes than the requested theta implies -- the port's effective theta is
  // coarser than the parameter file states. cellSizeInfo (computed above from the node's own
  // build-time level) is exactly Gadget-2's `len`. Env-gated via GADGET_HIP_T9_CELLSIZE_GEO so the
  // two can be A/B'd from one binary. The `+s` IMPBH term below is deliberately left alone so this
  // isolates the node-size term only.
  float l = g_t9_cellsize_geo ? cellSizeInfo[idx]
                              : 2*fmaxf(boxSize.x, fmaxf(boxSize.y, boxSize.z));

  //Store the box size and opening criteria
  boxSizeInfo[idx].x = boxSize.x;
  boxSizeInfo[idx].y = boxSize.y;
  boxSizeInfo[idx].z = boxSize.z;
  boxSizeInfo[idx].w = __int_as_float(n_children[idx]);

#if 1
  boxCenterInfo[idx].x = boxCenter.x;
  boxCenterInfo[idx].y = boxCenter.y;
  boxCenterInfo[idx].z = boxCenter.z;
#else /* added by egaburov, see dev_approximate_gravity_warp.cu for matching code*/
  boxCenterInfo[idx].x = mon.x;
  boxCenterInfo[idx].y = mon.y;
  boxCenterInfo[idx].z = mon.z;
#endif

  //Extra check, shouldnt be necessary, probably it is otherwise the test for leaf can fail
  //So it IS important Otherwise 0.0 < 0 can fail, now it will be: -1e-12 < 0 
  if(l < 0.000001)
    l = 0.000001;

  #ifdef IMPBH
    // T9b: Gadget-2's geometric criterion is `len*len > r2*theta*theta` (forcetree.c:1270) with NO
    // centre-of-mass-offset term. The `+s` here is Barnes (1994) "improved BH", an addition this
    // port inherited from Bonsai, not from the reference. It enlarges the opening radius (safe
    // direction) but it is an extra term, and combined with a corrected (larger) node size it
    // over-opens. GADGET_HIP_T9_NO_S=1 drops it, giving Gadget-2's criterion exactly.
    float cellOp = g_t9_no_s ? (l/theta) : ((l/theta) + s);
  #else
    //Minimum distance method
    float cellOp = (l/theta); 
  #endif
    
  cellOp = cellOp*cellOp;

  uint2 bij     = node_bodies[idx];
  uint pfirst   = bij.x & ILEVELMASK;
  uint nchild   = bij.y - pfirst;
  

  //If this is (leaf)node with only 1 particle then we change 
  //the opening criteria to a large number to force that the 
  //leaf will be opened and the particle data is used 
  //instead of an approximation.
  //This because sometimes (mass*pos)*(1.0/mass) != pos
  //even in full double precision
  if(nchild == 1)
  {
    cellOp = 10e10; //Force this node to be opened
  }

  if(r_max.w > 0)
  {
    cellOp = -cellOp;       //This is a leaf node
  }

  boxCenterInfo[idx].w = cellOp;


  //Change the indirections of the leaf nodes so they point to
  //the particle data
  bool leaf = (r_max.w > 0);
  if(leaf)
  {
    pfirst = pfirst | ((nchild-1) << LEAFBIT);
    boxSizeInfo[idx].w = __int_as_float(pfirst);
  }


  return;
}

//Compute the properties for the groups
KERNEL_DECLARE(gpu_setPHGroupData)(const int n_groups,
                                          const int n_particles,   
                                          real4 *bodies_pos,
                                          int2  *group_list,                                                
                                          real4 *groupCenterInfo,
                                          real4 *groupSizeInfo){
  CUXTIMER("setPHGroupData");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  if(bid >= n_groups)     return;

  //Do a reduction on the particles assigned to this group

  volatile __shared__ float3 shmem[2*NCRIT];
  volatile float3 *sh_rmin = (float3*)&shmem [ 0];
  volatile float3 *sh_rmax = (float3*)&shmem[NCRIT];

  float3 r_min = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_max = make_float3(-1e10f, -1e10f, -1e10f);

  int start = group_list[bid].x;
  int end   = group_list[bid].y;
  
  int partIdx = start + threadIdx.x;

  //Set the shared memory with the data
  if (partIdx >= end)
  {
    sh_rmin[tid].x = r_min.x; sh_rmin[tid].y = r_min.y; sh_rmin[tid].z = r_min.z;
    sh_rmax[tid].x = r_max.x; sh_rmax[tid].y = r_max.y; sh_rmax[tid].z = r_max.z;
  }
  else
  {
    sh_rmin[tid].x = r_min.x = bodies_pos[partIdx].x; sh_rmin[tid].y = r_min.y = bodies_pos[partIdx].y; sh_rmin[tid].z = r_min.z = bodies_pos[partIdx].z;
    sh_rmax[tid].x = r_max.x = bodies_pos[partIdx].x; sh_rmax[tid].y = r_max.y = bodies_pos[partIdx].y; sh_rmax[tid].z = r_max.z = bodies_pos[partIdx].z;
  }


  __syncthreads();
  // do reduction in shared mem  
  if(blockDim.x >= 512) if (tid < 256) {sh_MinMax2(tid, tid + 256, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 256) if (tid < 128) {sh_MinMax2(tid, tid + 128, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 128) if (tid < 64)  {sh_MinMax2(tid, tid + 64,  &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();

  // Unsynchronised warp-reduction tail replaced. Same construct as gpu_boundaryReduction in
  // build_tree.cu, where it was caught returning a partial, run-dependent min/max (tickets/T21).
  // These two reductions build the GROUP AABBs (groupSizeInfo/groupCenterInfo) that the opening
  // criterion reads, so a wrong result here perturbs every force, not just a diagnostic.
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s) sh_MinMax2(tid, tid + s, &r_min, &r_max, sh_rmin, sh_rmax);
    __syncthreads();
  }
  
  // write result for this block to global mem
  if (tid == 0)
  {

    //Compute the group center and size
    float3 grpCenter;
    grpCenter.x = 0.5*(r_min.x + r_max.x);
    grpCenter.y = 0.5*(r_min.y + r_max.y);
    grpCenter.z = 0.5*(r_min.z + r_max.z);

    float3 grpSize = make_float3(fmaxf(fabs(grpCenter.x-r_min.x), fabs(grpCenter.x-r_max.x)),
                                 fmaxf(fabs(grpCenter.y-r_min.y), fabs(grpCenter.y-r_max.y)),
                                 fmaxf(fabs(grpCenter.z-r_min.z), fabs(grpCenter.z-r_max.z)));

    //Store the box size and opening criteria
    groupSizeInfo[bid].x = grpSize.x;
    groupSizeInfo[bid].y = grpSize.y;
    groupSizeInfo[bid].z = grpSize.z;

    int nchild             = end-start;
    start                  = start | (nchild-1) << CRITBIT;
    groupSizeInfo[bid].w   = __int_as_float(start);  

    float l = max(grpSize.x, max(grpSize.y, grpSize.z));

    groupCenterInfo[bid].x = grpCenter.x;
    groupCenterInfo[bid].y = grpCenter.y;
    groupCenterInfo[bid].z = grpCenter.z;

    //Test stats for physical group size
    groupCenterInfo[bid].w = l;

  } //end tid == 0
}//end copyNode2grp



#if 0
//Compute the properties for the groups
KERNEL_DECLARE(gpu_setPHGroupDataGetKey)(const int n_groups,
                                          const int n_particles,
                                          real4 *bodies_pos,
                                          int2  *group_list,
                                          real4 *groupCenterInfo,
                                          real4 *groupSizeInfo,
                                          uint4  *body_key,
                                          float4 corner){
  CUXTIMER("setPHGroupDataGetKey");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  if(bid >= n_groups)     return;

  //Do a reduction on the particles assigned to this group

  volatile __shared__ float3 shmem[2*NCRIT];
  volatile float3 *sh_rmin = (float3*)&shmem [ 0];
  volatile float3 *sh_rmax = (float3*)&shmem[NCRIT];

  float3 r_min = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_max = make_float3(-1e10f, -1e10f, -1e10f);

  int start = group_list[bid].x;
  int end   = group_list[bid].y;

  int partIdx = start + threadIdx.x;

  //Set the shared memory with the data
  if (partIdx >= end)
  {
    sh_rmin[tid].x = r_min.x; sh_rmin[tid].y = r_min.y; sh_rmin[tid].z = r_min.z;
    sh_rmax[tid].x = r_max.x; sh_rmax[tid].y = r_max.y; sh_rmax[tid].z = r_max.z;
  }
  else
  {
    sh_rmin[tid].x = r_min.x = bodies_pos[partIdx].x; sh_rmin[tid].y = r_min.y = bodies_pos[partIdx].y; sh_rmin[tid].z = r_min.z = bodies_pos[partIdx].z;
    sh_rmax[tid].x = r_max.x = bodies_pos[partIdx].x; sh_rmax[tid].y = r_max.y = bodies_pos[partIdx].y; sh_rmax[tid].z = r_max.z = bodies_pos[partIdx].z;
  }


  __syncthreads();
  // do reduction in shared mem
  if(blockDim.x >= 512) if (tid < 256) {sh_MinMax2(tid, tid + 256, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 256) if (tid < 128) {sh_MinMax2(tid, tid + 128, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 128) if (tid < 64)  {sh_MinMax2(tid, tid + 64,  &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();

  // Unsynchronised warp-reduction tail replaced. Same construct as gpu_boundaryReduction in
  // build_tree.cu, where it was caught returning a partial, run-dependent min/max (tickets/T21).
  // These two reductions build the GROUP AABBs (groupSizeInfo/groupCenterInfo) that the opening
  // criterion reads, so a wrong result here perturbs every force, not just a diagnostic.
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s) sh_MinMax2(tid, tid + s, &r_min, &r_max, sh_rmin, sh_rmax);
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)
  {

    //Compute the group center and size
    float3 grpCenter;
    grpCenter.x = 0.5*(r_min.x + r_max.x);
    grpCenter.y = 0.5*(r_min.y + r_max.y);
    grpCenter.z = 0.5*(r_min.z + r_max.z);

    float3 grpSize = make_float3(fmaxf(fabs(grpCenter.x-r_min.x), fabs(grpCenter.x-r_max.x)),
                                 fmaxf(fabs(grpCenter.y-r_min.y), fabs(grpCenter.y-r_max.y)),
                                 fmaxf(fabs(grpCenter.z-r_min.z), fabs(grpCenter.z-r_max.z)));

    //Store the box size and opening criteria
    groupSizeInfo[bid].x = grpSize.x;
    groupSizeInfo[bid].y = grpSize.y;
    groupSizeInfo[bid].z = grpSize.z;

    real4 pos = bodies_pos[start];

    int nchild             = end-start;
    start                  = start | (nchild-1) << CRITBIT;
    groupSizeInfo[bid].w   = __int_as_float(start);

    float l = max(grpSize.x, max(grpSize.y, grpSize.z));

    groupCenterInfo[bid].x = grpCenter.x;
    groupCenterInfo[bid].y = grpCenter.y;
    groupCenterInfo[bid].z = grpCenter.z;

    //Test stats for physical group size
    groupCenterInfo[bid].w = l;

    int4 crd;

    real domain_fac = corner.w;

    #ifndef EXACT_KEY
       crd.x = (int)roundf(__fdividef((pos.x - corner.x), domain_fac));
       crd.y = (int)roundf(__fdividef((pos.y - corner.y) , domain_fac));
       crd.z = (int)roundf(__fdividef((pos.z - corner.z) , domain_fac));
    #else
       crd.x = (int)((pos.x - corner.x) / domain_fac);
       crd.y = (int)((pos.y - corner.y) / domain_fac);
       crd.z = (int)((pos.z - corner.z) / domain_fac);
    #endif

    body_key[bid] = get_key(crd);

  } //end tid == 0
}//end copyNode2grp

//Compute the key for the groups
KERNEL_DECLARE(gpu_setPHGroupDataGetKey2)(const int n_groups,
                                      real4 *bodies_pos,
                                      int2  *group_list,
                                      uint4  *body_key,
                                      float4 corner){
  CUXTIMER("setPHGroupDataGetKey2");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const uint idx = bid * (blockDim.x * blockDim.y) + tid;

  if(idx >= n_groups)     return;


  int start = group_list[idx].x;
  real4 pos = bodies_pos[start];

//  int end   = group_list[idx].y-1;
//  real4 pos = bodies_pos[end];

//  int end   = group_list[idx].y-1;
//  int start = group_list[idx].x;
//  start     = (end+start) / 2;
//  real4 pos = bodies_pos[start];


  int4 crd;

  real domain_fac = corner.w;

  #ifndef EXACT_KEY
     crd.x = (int)roundf(__fdividef((pos.x - corner.x), domain_fac));
     crd.y = (int)roundf(__fdividef((pos.y - corner.y) , domain_fac));
     crd.z = (int)roundf(__fdividef((pos.z - corner.z) , domain_fac));
  #else
     crd.x = (int)((pos.x - corner.x) / domain_fac);
     crd.y = (int)((pos.y - corner.y) / domain_fac);
     crd.z = (int)((pos.z - corner.z) / domain_fac);
  #endif
    // uint2 key =  get_key_morton(crd);
    // body_key[idx] = make_uint4(key.x, key.y, 0,0);
    body_key[idx] = get_key(crd); //has to be PH key in order to prevent the need for sorting

}//end copyNode2grp

#endif

