#include "hip/hip_runtime.h"
#ifndef _OCTREE_H_
#define _OCTREE_H_

#include "gadget_timeline.h"

#include "gadget_params.h"
#include "gadget_softening.h"
#include "gadget_cosmology.h"
#include "gadget_driftfac.h"
#include <unordered_map>
#ifdef GADGET_HIP_HIGHRES
#include "pm_zoom.h"
#endif

#ifdef USE_MPI
  #include "mpi.h"
#endif



#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <windows.h>
#endif

#define USE_CUDA

#ifdef USE_CUDA
  #include "my_cuda_rt.h"
#else
  #include "my_ocl.h"
#endif

#include "tipsydefs.h"

#include "node_specs.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include "logFileWriter.h"
#include "SharedMemory.h"
#include "tipsyIO.h"
#include "log.h"
#include "FileIO.h"



#ifdef USE_MPI
  #include "MPIComm.h"
  extern MPIComm *myComm;
#endif

#ifndef WIN32
  #include <unistd.h>
#endif




typedef float              real;
typedef float2             real2;
typedef unsigned int       uint;
typedef unsigned long long ullong; //ulonglong1

#define NBLOCK_REDUCE     256
#define NBLOCK_BOUNDARY   120
#define NTHREAD_BOUNDARY  256
#define NBLOCK_PREFIX     512           //At the moment only used during memory alloc

#define NMAXSAMPLE 20000                //Used by first on host domain division



/*
 * V1 IDs, 32 bit integers
 * >  200000000               => Dark-matter
 * >= 100000000 <  200000000  => Bulge
 * >= 0         <  100000000  => Disk
 *    Possible:
 *      >= 0         <  40000000 => Disk
 *      >= 40000000  <  50000000 => Glowing stars in spiral arms
 *      >= 50000000  <  70000000 => Dust
 *      >= 70000000  < 100000000 => Glow mass less dust particles
 *
 * V2 IDs, 64 bit integers => 9.223.372.036.854.775.807
 *
 * >  3.000.000.000.000.000.000                             => Dark-matter
 * >= 2.000.000.000.000.000.000 < 3.000.000.000.000.000.000 => Bulge
 * >= 0                         < 2.000.000.000.000.000.000 => Disk
 *
 *
 */

#define DARKMATTERID  3000000000000000000
#define DISKID        0
#define BULGEID       2000000000000000000




typedef struct setupParams {
  int jobs;                     //Minimal number of jobs for each 'processor'
  int blocksWithExtraJobs;      //Some ' processors'  do one extra job all with bid < bWEJ
  int extraElements;            //The elements that didn't fit completely
  int extraOffset;              //Start of the extra elements

} setupParams;


typedef struct sampleRadInfo
{
  int     nsample;
  double4 rmin;
  double4 rmax;
}sampleRadInfo;


inline int cmp_uint4(uint4 a, uint4 b) {
  if      (a.x < b.x) return -1;
  else if (a.x > b.x) return +1;
  else {
    if       (a.y < b.y) return -1;
    else  if (a.y > b.y) return +1;
    else {
      if       (a.z < b.z) return -1;
      else  if (a.z > b.z) return +1;
      return 0;
    } //end z
  }  //end y
} //end x, function


struct cmp_ph_key{
  bool operator () (const uint4 &a, const uint4 &b){
    return ( cmp_uint4( a, b) < 1);
  }
};


class particleSet
{
public:
	int n;							 //Number of bodies
    my_dev::dev_mem<real4>  pos;     //The particles positions
    my_dev::dev_mem<uint4>  key;     //The particles keys
    my_dev::dev_mem<real4>  vel;     //Velocities
    my_dev::dev_mem<real4>  acc0;    //Acceleration
    my_dev::dev_mem<real4>  acc1;    //Acceleration
    my_dev::dev_mem<float2> time;    //The timestep details (.x=tb, .y=te
    my_dev::dev_mem<ullong> ids;
    my_dev::dev_mem<real4>  Ppos;    //Predicted position
    my_dev::dev_mem<real4>  Pvel;    //Predicted velocity

    //Density related buffers
    my_dev::dev_mem<real>  h;       //The particles search radius
    my_dev::dev_mem<real2> dens;    //The particles density (x) and number of neighbors (y)

    particleSet(){ n = 0;}

    void setN(int particles) { n = particles; }

    void allocate(int n_bodies = -1)
    {
    	if(n_bodies <= 0) n_bodies = n;
		//Particle properties
		pos.cmalloc(n_bodies+1, true);   //+1 to set end pos, host mapped? TODO mapped not needed right since we use Ppos?
		vel.cmalloc(n_bodies,   false);
		key.cmalloc(n_bodies+1, false);  //+1 to set end key
		ids.cmalloc(n_bodies+1, false);  //+1 to set end key

		Ppos.cmalloc(n_bodies+1, true);  //Memory to store predicted positions, host mapped
		Pvel.cmalloc(n_bodies+1, true);  //Memory to store predicted velocities, host mapped


		acc0.ccalloc(n_bodies, false);   //ccalloc -> init to 0
		acc1.ccalloc(n_bodies, false);   //ccalloc -> init to 0
		time.ccalloc(n_bodies, false);   //ccalloc -> init to 0

		h.cmalloc   (n_bodies, true);
		dens.cmalloc(n_bodies, true);

		//Initialize to -1
		for(int i=0; i < n_bodies; i++) h[i] = -1;
		h.h2d();
    }

    void reallocate(int n_bodies = -1)
    {
    	if(n_bodies <= 0) n_bodies = n;
		//Particle properties
		pos.cresize(n_bodies+1, true);   //+1 to set end pos, host mapped? TODO not needed right since we use Ppos
		vel.cresize(n_bodies, false);
		key.cresize(n_bodies+1, false);  //+1 to set end key
		ids.cresize(n_bodies+1, false);  //+1 to set end key

		Ppos.cresize(n_bodies+1, true);  //Memory to store predicted positions, host mapped
		Pvel.cresize(n_bodies+1, true);  //Memory to store predicted velocities, host mapped


		acc0.cresize(n_bodies, false);   //ccalloc -> init to 0
		acc1.cresize(n_bodies, false);   //ccalloc -> init to 0
		time.cresize(n_bodies, false);   //ccalloc -> init to 0

		h.cresize   (n_bodies, true);
		dens.cresize(n_bodies, true);
    }

};

//Structure and properties of a tree
class tree_structure
{
  public:
    int n;                                //Number of particles in the tree
    int n_leafs;                          //Number of leafs in the tree
    int n_nodes;                          //Total number of nodes in the tree (including leafs)
    int n_groups;                         //Number of groups
    int n_levels;                         //Depth of the tree
    
    uint startLevelMin;                   //The level from which we start the tree-walk
                                          //this is decided by the tree-structure creation

    //Variables used for iteration
    int n_active_groups;
    int n_active_particles;

    real4 corner;                         //Corner of tree-structure
    real  domain_fac;                     //Domain_fac of tree-structure

    particleSet	bodies;

    my_dev::dev_mem<real4>  bodies_pos;     //The particles positions
    my_dev::dev_mem<uint4>  bodies_key;     //The particles keys
    my_dev::dev_mem<real4>  bodies_vel;     //Velocities
    my_dev::dev_mem<real4>  bodies_acc0;    //Acceleration
    my_dev::dev_mem<real4>  bodies_acc1;    //Acceleration
    // Phase 2: INTEGER timeline. .x = Ti_begstep, .y = Ti_endstep, in ticks (GADGET-2
    // allvars.h uses int Ti_begstep). Two particles that should re-synchronise now hold
    // bit-identical values by construction; as float32 scale factors they differed by 1 ULP
    // and the rendezvous was missed, costing a whole system step (T24/T38).
    my_dev::dev_mem<int2>   bodies_time;    //The timestep details (.x=Ti_beg, .y=Ti_end
    my_dev::dev_mem<ullong> bodies_ids;
    my_dev::dev_mem<real4>  bodies_Ppos;    //Predicted position
    my_dev::dev_mem<real4>  bodies_Pvel;    //Predicted velocity
    
    //Density related buffers
    my_dev::dev_mem<real>  bodies_h;       //The particles search radius
    my_dev::dev_mem<real2> bodies_dens;    //The particles density (x) and number of neighbours (y)

    // Phase 5 ticket 03 (PLAN.md): per-particle Gadget-2 type (0-5) and its resolved
    // ForceSoftening value. `bodies_type` is host-only (never uploaded to the device -- the
    // gravity kernels only ever need the already-resolved length, not the raw type index).
    // IMPORTANT (bug found during ticket 06's own end-to-end zoom test, LOG.md): sort_bodies()'s
    // SFC sort NEVER reorders this array -- only tree.bodies_ids (among host-observable arrays) is
    // unconditionally reordered by every sort_bodies() call. `bodies_type`'s own array index is
    // therefore NOT trustworthy as index-aligned with bodies_pos/bodies_acc1 across rebuilds; it
    // must instead be rebuilt every rebuild from the current (always-correct) bodies_ids through
    // octree::gadgetIdToType's stable (particle ID -> type) map -- which is exactly what
    // octree::recomputeSoftening() now does, first thing, before using bodies_type for anything.
    // A no-op (and therefore harmless to skip) whenever gadgetIdToType is empty (uniform-default-
    // type runs, e.g. --plummer with no --param IC -- permuting an array of identical values was
    // never observably wrong, which is exactly why tickets 03-05 never caught this).
    // `bodies_forceSoftening` is a real device array, recomputed by octree::recomputeSoftening() at
    // tree-rebuild granularity (matching Gadget-2's own set_softenings() call sites,
    // GADGET2_NOTES.md-style citation: PHASE5_ZOOM_COMOVING_SPEC.md Sec 4.2) and consumed exactly
    // like bodies_h/bodies_dens above.
    std::vector<int>       bodies_type;
    my_dev::dev_mem<real>  bodies_forceSoftening;
    // Phase 5 ticket 06 (PLAN.md): device-resident copy of bodies_type -- unlike the softening
    // case above, the zoom force-accumulation kernels (pm_add_masked_force_to_acc()) need the
    // type on the DEVICE to decide per-particle whether the fine grid's force applies
    // (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.1's type-filtered readback). Refreshed every rebuild
    // inside recomputeSoftening(), right after that function rebuilds the (equally rebuild-fragile,
    // see bodies_type's own comment above) host-side bodies_type -- NOT a one-time upload.
    // UNCONDITIONAL as of Ticket 07 (PLAN.md): the tree-walk gravity kernel's per-target Rcut/Asmth
    // selection (dev_approximate_gravity_warp_new.cu) reads this in every build now, not just
    // GADGET_HIP_HIGHRES ones -- harmless in a non-HIGHRES build (the zoom-mask constant it's
    // tested against, g_pm_zoom_mask, only exists/matters there) or a non-zoom run (mask stays 0),
    // but the array itself must always exist so the kernel signature doesn't need two variants.
    my_dev::dev_mem<int>   bodies_typeDevice;

    my_dev::dev_mem<uint>   oriParticleOrder;  //Used in the correct function to speedup reorder

    
    my_dev::dev_mem<uint2> level_list;    //List containing the start and end positions of each level
    my_dev::dev_mem<uint>  n_children;
    my_dev::dev_mem<uint2> node_bodies;
    my_dev::dev_mem<uint>  leafNodeIdx;    //First n_leaf items represent indices of leafs
                                           //remaining (n_nodes-n_leafs) are indices of non-leafs

    my_dev::dev_mem<uint>  node_level_list; //List containing start and end idxs in (leafNode idx) for each level
    my_dev::dev_mem<uint>  body2group_list; //Contains per particle to which group it belongs
    my_dev::dev_mem<uint2> group_list;      //The group to particle relation


    //Variables used for properties
    my_dev::dev_mem<real4>  multipole;      	//Array storing the properties for each node (mass, mono, quad pole)

    my_dev::dev_mem<uint>  activeGrpList;       //Non-compacted list of active groups
    my_dev::dev_mem<uint>  active_group_list;   //Compacted list of active groups
    my_dev::dev_mem<uint>  activePartlist;      //List of active particles
    my_dev::dev_mem<uint>  ngb;                 //List of nearest neighbors

    my_dev::dev_mem<int2>  interactions;        //Counts the number of interactions, mainly for debugging and performance

    //Properties of the tree-node boxes
    my_dev::dev_mem<float4> boxSizeInfo;
    my_dev::dev_mem<float4> groupSizeInfo;
    my_dev::dev_mem<float4> boxCenterInfo;
    my_dev::dev_mem<float4> groupCenterInfo;
    // T28/C-A-04: per-node softening summary for Gadget-2's mixed-softening rules.
    // Encoding: |value| = max ForceSoftening over the node's members; sign < 0 means the members
    // do NOT all share one softening (Gadget's `diffsoftflag`, bitflags bit 5); 0.0f is the
    // "undefined / no contributor" sentinel (Gadget's maxsofttype == 7), which the walk must treat
    // as force-open. A bare maximum cannot express "do the members differ", which is what the
    // forced-descent rule actually tests -- hence the sign channel rather than a second buffer.
    // Sized and indexed exactly like cellSizeInfo, which the walk already fetches per cell.
    my_dev::dev_mem<float>  nodeSoftInfo;
    my_dev::dev_mem<float>  groupMaxAccInfo;  //Per-group max |a_old|, Springel(2005) MAC input (Phase 3)
    my_dev::dev_mem<float>  groupMaxSofteningInfo;  //T27: per-group max Gadget-2 ForceSoftening,
                                                     //softening-based bJ floor input (mirrors groupMaxAccInfo)
    my_dev::dev_mem<float>  cellSizeInfo;     //T26: per-node FIXED octree-cell side length (domain_fac
                                               //* 2^(MAXLEVELS-level)), matching Gadget-2's own `len` --
                                               //independent of the node's particle-fitted AABB
                                               //(boxSizeInfo), which is what split_node_grav_springel's
                                               //bJ used exclusively before this ticket.

    // C-A-01 part (b): per-node FIXED octree-cell CENTRE, the companion to cellSizeInfo's side.
    // Gadget-2's proximity/safety box is `fabs(nop->center[a] - pos) < 0.6*nop->len`
    // (forcetree.c:1288-1292) and `nop->center` is the octree cube's geometric centre, set once at
    // insertion (forcetree.c:190-206) and never fitted to particles. boxCenterInfo is the AABB
    // MIDPOINT, which is a different point -- it drifts inside the cell as the node's particles
    // cluster, exactly the way boxSizeInfo's extent shrinks. Computed at tree-build time rather
    // than derived in the walk because the derivation needs (pos - corner)/cell, a ratio reaching
    // ~1e9 at 512^3 and therefore double precision -- affordable once per node per rebuild,
    // not per node visit in the hot walk.
    my_dev::dev_mem<real4> cellCenterInfo;

    my_dev::dev_mem<uint4> parallelBoundaries;

    //Combined buffers:
    /*
      Buffer1 used during: Sorting, Tree-construction, and Tree-traverse:
      Sorting:
        - SrcValues, Output, simpleKeys, permutation, output32b, valuesOutput
      Tree-construction:
        - ValidList, compactList
      Tree-traverse:
        - Interactions, NGB, Active_partlist
    */

    my_dev::dev_mem<uint> generalBuffer1;


    my_dev::dev_mem<float4> fullRemoteTree;
    uint4 remoteTreeStruct;				  //Properties of the remote tree-structure (particles, nodes, offsets)

    

  tree_structure(){ n = 0;}


  void setN(int particles) { n = particles; }

};



// --debug (or GADGET_HIP_DEBUG=1): enable the per-step developer probes. These were left
// unconditional after the investigations that added them and dominated a production log; they
// print device pointers, buffer sizes and group-max traces that do not change between iterations.
extern bool gadget_hip_debug_log;

class octree {
protected:
  const MPI_Comm &mpiCommWorld;
  int devID;
  
  char *execPath;
  char *src_directory;
  
  //Device configuration
  int nMultiProcessors;
  int nBlocksForTreeWalk;

   //Simulation properties
  int           iter;
  // Phase 2: ticks are the state of record; t_current/t_previous are DERIVED from them for the
  // many call sites that legitimately want a scale factor (output epochs, snapshot headers,
  // drift/kick table inputs). Nothing stores `a` as state any more.
  gadget_tick_t Ti_current, Ti_previous;
  GadgetTimeline gadgetTimeline;
  double        gadgetDPerTick;   // == gadgetTimeline.dPerTick; set_args needs an addressable copy
  float         t_current, t_previous;
  float         snapshotIter;   
  float         quickDump, quickRatio;
  bool          quickSync, useMPIIO, mpiRenderMode;
  string        snapshotFile;
  float         nextSnapTime;
  float         nextQuickDump;

  float         statisticsIter;
  float         nextStatsTime;
  int 			rebuild_tree_rate;


  float eps2;

  // Phase 5 tickets 03/04 (PLAN.md): the full parsed --param config, set once via
  // setGadgetParams() when a --param run supplies one (ticket 01's GadgetParams already carries
  // every field both the softening (03) and timestep (04) tickets need -- one shared copy, not a
  // per-ticket subset). haveGadgetParams==false (the default, e.g. every dev/diagnostic run not
  // using --param) makes recomputeSoftening() fall back to a flat length for every type driven by
  // the existing eps/eps2 CLI value (h=2.8*eps, a deliberate, documented behavior change from the
  // old literal-Plummer-eps2 default, PLAN.md Phase 5.5) and recomputeTimestepGlobals() fall back
  // to a non-comoving-equivalent (no displacement constraint, atime=fac1=hubble_a=1).
  GadgetParams gadgetParams;
  bool         haveGadgetParams = false;

  // Phase 5 ticket 05 (PLAN.md): driftfac.c drift/grav-kick tables, built once (initComovingTables(),
  // called from main.cpp right after setGadgetParams() when ComovingIntegrationOn) and uploaded to
  // these persistent device buffers -- consumed every step by predict()/correct() via
  // predictParticles/correctParticles's set_args() calls. haveGadgetComovingTables==false makes
  // both kernels fall back to their pre-ticket flat-dt_cb behavior exactly (see timestep.cu).
  my_dev::dev_mem<float> gadgetDriftTable;
  my_dev::dev_mem<float> gadgetGravKickTable;
  // T29/C-D-06: DOUBLE. These are the origin and span of the drift/kick table coordinate; holding
  // them in float quantises the table lookup's input, and no amount of double arithmetic downstream
  // recovers that. One float32 ULP of the table coordinate `u` is ~0.98 of one 2^24 timeline tick.
  double gadgetLogTimeBegin = 0.0;
  double gadgetLogTimeMax   = 0.0;
  bool  haveGadgetComovingTables = false;

  // Phase 5 ticket 08 (PLAN.md): periodic Gadget-2-format snapshot output for a --param run.
  // Tickets 01/02 built the .param parser and the gadget_snapshot_write() function itself
  // (validated via its own write+read round-trip self-test), but never wired that function into
  // the actual per-step simulation loop -- a real gap found while building ticket 08's own
  // end-to-end test, since without it a --param run has no way to produce the Gadget-2-format
  // snapshots ticket 08's exit criteria need to exist at all. nextGadgetSnapTime mirrors Gadget-2's
  // own output-time bookkeeping (run.c ~279-330: multiplicative in `a` under comoving integration,
  // additive otherwise) -- initialized from gadgetParams.TimeOfFirstSnapshot in setGadgetParams().
  double nextGadgetSnapTime  = -1.0;

  // --------------------------------------------------------------------------------------------
  // Restart (stop/resume). Deliberately NOT Gadget-2's restart-file format -- the user confirmed
  // compatibility is not required, and Gadget's format is a raw dump of its own structs.
  //
  // A snapshot is NOT sufficient to resume from: it carries positions, velocities, IDs and masses,
  // but none of the integrator state. Resuming from one silently restarts every particle
  // synchronised (losing the individual-timestep bin structure) and with no previous force. What
  // must additionally be preserved:
  //     bodies_time  -- each particle's (Ti_begstep, Ti_endstep)
  //     bodies_acc0  -- the previous step's force, read by correct_particles' kick
  //     t_current / t_previous / iter / the snapshot counter / the snapshot and statistics
  //     cadence cursors / the displacement ceiling
  //
  // INDEX SPACE: the file is written entirely in ORIGINAL (load) order, so reading it back is
  // indistinguishable from a fresh IC load and sort_bodies rebuilds the permutation from scratch.
  // The default per-iteration reorder leaves bodies_vel/Pvel/time/acc0 in original order but
  // permutes bodies_Ppos/bodies_ids, so those two are un-permuted through oriParticleOrder on the
  // way out. Getting this wrong would pair each particle's position with another's velocity -- the
  // same defect class as C-B-13/C-C-18/C-D-08, and invisible until the run diverged.
public:  // main() configures the restart from the CLI/parameterfile, so these are not protected.
  bool   writeRestartFile(const char *path);
  bool   readRestartFile(const char *path);
  double restartCpuInterval = 0.0;   // CpuTimeBetRestartFile, seconds; 0 disables periodic dumps
  // Promoted from a function-static in iterate_once so a resume can restore it: the statistics
  // cadence cursor is integrator state like any other, and a resume that reset it would silently
  // shift every subsequent [ENERGY-EXACT] sample.
  double restartNextEnergyStat = -1.0;
  bool   restartResumed        = false;
  double lastRestartWrite   = 0.0;
  std::string restartPath;           // where to write/read; empty disables
protected:
  int    gadgetSnapshotCount = 0;

  float inv_theta;
  int   dt_limit;
  float eta;
  float timeStep;
  float tEnd;
  int   iterEnd;
  float theta;

  bool  useDirectGravity;

  // Phase 3: Springel(2005)/Gadget-2 relative acceleration MAC tolerance -- only consumed by the
  // tree-walk kernel when built with _MAC_SPRINGEL_ (node_specs.h); harmless to always carry the
  // member/CLI flag either way. See AMD_PORT_PLAN.md Phase 3.
  float errTolForceAcc;

  //Simulation statistics
  double Ekin, Ekin0, Ekin1;
  double Epot, Epot0, Epot1;
  double Etot, Etot0, Etot1;

  bool   store_energy_flag;
  double tinit;

  LOGFILEWRITER *logFileWriter;



  // Device context
  my_dev::context *devContext;

  
  //Streams
  my_dev::dev_stream *gravStream;
  my_dev::dev_stream *execStream;
  my_dev::dev_stream *copyStream;
  my_dev::dev_stream *LETDataToHostStream;
  

  // scan & split kernels
  my_dev::kernel  compactCount, exScanBlock, compactMove, splitMove;

  // tree construction kernels
  my_dev::kernel  build_key_list;
  my_dev::kernel  build_valid_list;
  my_dev::kernel  build_nodes;
  my_dev::kernel  link_tree;
  my_dev::kernel  define_groups;
  my_dev::kernel  build_level_list;
  my_dev::kernel  store_groups;

  my_dev::kernel  boundaryReduction;
  my_dev::kernel  boundaryReductionGroups;
  my_dev::kernel  build_body2group_list;


  // tree properties kernels
  my_dev::kernel  propsNonLeafD, propsLeafD, propsScalingD;
  my_dev::kernel  setPHGroupData;
  my_dev::kernel  setActiveGrps;

  //Time integration kernels
  my_dev::kernel getTNext;
  my_dev::kernel predictParticles;
  my_dev::kernel getNActive;
  my_dev::kernel approxGrav;
  my_dev::kernel approxGravLET;
  my_dev::kernel correctParticles;
  my_dev::kernel computeDt;
  my_dev::kernel gadgetTimestepGlobals;   // T35: device-side per-type (count, v2sum, min mass)
  my_dev::kernel gadgetRefreshSoftening;   // T35: device-side per-particle softening/type refresh
  my_dev::kernel gadgetBoxWrap;            // T37: Gadget's do_box_wrapping, applied before the tree build
  my_dev::kernel computeEnergy;
  my_dev::kernel computeEnergyPreKick;  // T19: pre-kick energy sample, matches Gadget-2's timing

  //Other
  my_dev::kernel directGrav;

  //Parallel kernels
  my_dev::kernel internalMoveSFC2;
  my_dev::kernel extractOutOfDomainParticlesAdvancedSFC2;
  my_dev::kernel insertNewParticlesSFC;
  my_dev::kernel domainCheckSFCAndAssign;

  ///////////////////////

  // accurate Win32 timing
#ifdef WIN32
  LARGE_INTEGER sysTimerFreq;
  LARGE_INTEGER sysTimerAtStart;
#endif

  ///////////

public:
   my_dev::context * getDevContext() { return devContext; };


   //Memory used in the whole system, not depending on a certain number of particles
   my_dev::dev_mem<int>   	tnext;   // Phase 2: sync point in ticks
   my_dev::dev_mem<uint>  	nactive;
   //General memory buffers
   my_dev::dev_mem<float3>  devMemRMIN;
   my_dev::dev_mem<float3>  devMemRMAX;

   my_dev::dev_mem<uint> devMemCounts;
   my_dev::dev_mem<uint> devMemCountsx;

   tree_structure localTree;
   tree_structure remoteTree;

   tipsyIO *fileIO;

   double get_time();
   void resetEnergy() {store_energy_flag = true;}
   void set_src_directory(string src_dir);

   void writeLogData(std::string &str){ devContext->writeLogEvent(str.c_str());}
   void writeLogToFile(){ this->logFileWriter->updateLogData(devContext->getLogData());}

   int getAllignmentOffset(int n);
   int getTextureAllignmentOffset(int n, int size);

   //GPU kernels and functions
   void load_kernels();
   void resetCompact();

   void gpuCompact(my_dev::dev_mem<uint> &srcValues,
                   my_dev::dev_mem<uint> &output, int N, int *validCount);
   void gpuSplit(my_dev::dev_mem<uint> &srcValues,
                 my_dev::dev_mem<uint> &output, int N, int *validCount);
   void gpuSort(my_dev::dev_mem<uint4> &srcKeys,
                my_dev::dev_mem<uint>   &permutation, //For 32bit values
                my_dev::dev_mem<uint>   &tempB,       //For 32bit values
                my_dev::dev_mem<uint>   &tempC,       //For 32bit keys
                my_dev::dev_mem<uint>   &tempD,       //For 32bit keys
                my_dev::dev_mem<char>   &tempE,       //For sorting space
                int N);

   template <typename T>
   void dataReorder(const int N, my_dev::dev_mem<uint> &permutation,
                    my_dev::dev_mem<T>  &dIn, my_dev::dev_mem<T>  &scratch,
                    bool overwrite = true,
                    bool devOnly   = false);

   template <typename T>
   void dataReorder2(const int N, my_dev::dev_mem<uint> &permutation,
                     my_dev::dev_mem<T>  &dIn, my_dev::dev_mem<T>  &dOut);

    void sort_bodies(tree_structure &tree, bool doDomainUpdate, bool doFullShuffle = false);
    void getBoundaries(tree_structure &tree, real4 &r_min, real4 &r_max);
    void getBoundariesGroups(tree_structure &tree, real4 &r_min, real4 &r_max);  

    void allocateParticleMemory(tree_structure &tree);
    void allocateTreePropMemory(tree_structure &tree);
    void reallocateParticleMemory(tree_structure &tree);

    void build(tree_structure &tree);
    void compute_properties (tree_structure &tree);
    void compute_properties_double(tree_structure &tree);
    void setActiveGrpsFunc(tree_structure &tree);


  struct IterationData {
      IterationData() : Nact_since_last_tree_rebuild(0),
          totalGravTime(0), lastGravTime(0), totalBuildTime(0),
          lastBuildTime(0), totalDomTime(0), lastDomTime(0),
          totalWaitTime(0), lastWaitTime(0), startTime(0),
          totalGPUGravTimeLocal(0), totalGPUGravTimeLET(0),
          lastGPUGravTimeLocal(0), lastGPUGravTimeLET(0),
          lastLETCommTime(0), totalLETCommTime(0),
          totalDomUp(0), totalDomEx(0), totalDomWait(0),
          totalPredCor(0){}

      int    Nact_since_last_tree_rebuild;
      double totalGravTime; //CPU timers, includes any non-hidden communication cost
      double lastGravTime;
      double totalBuildTime;
      double lastBuildTime;
      double totalDomTime;
      double lastDomTime;
      double totalWaitTime;
      double lastWaitTime;
      double startTime;
      double totalGPUGravTimeLocal; //GPU timers, gravity only
      double totalGPUGravTimeLET;
      double lastGPUGravTimeLocal;
      double lastGPUGravTimeLET;
      double lastLETCommTime; //Time it takes to communicate/build LET structures
      double totalLETCommTime;
      double totalDomUp;
      double totalDomEx;
      double totalDomWait;
      double totalPredCor;
  };

  void iterate(bool amuse = false);
  void iterate_setup(); 
  void iterate_teardown(IterationData &idata); 
  bool iterate_once(IterationData &idata); 

  //Bonsai IO related
  void terminateIO() const;
  template<typename THeader, typename TData>
    void dumpDataCommon(
        SharedMemoryBase<THeader> &header, SharedMemoryBase<TData> &data,
        const std::string &fileNameBase, const float ratio, const bool sync);
  void dumpData();
  void dumpDataMPI();
  void lReadBonsaiFile(std::vector<real4 > &,std::vector<real4 > &, std::vector<ullong> &,
                      float &tCurrent, const std::string &fileName, const int rank, const int nrank,
                      const MPI_Comm &comm, const bool restart = true, const int reduceFactor = 1);

  //Sub functions of iterate, should probably be private
  void   predict(tree_structure &tree);
  void   approximate_gravity(tree_structure &tree);
  // Phase 5 tickets 03/04 (PLAN.md): stores the whole parsed --param config for subsequent
  // recomputeSoftening()/recomputeTimestepGlobals() calls -- called once from main.cpp when
  // --param is given.
  void   setGadgetParams(const GadgetParams &params);
  // Bug fix (found during Phase 5 ticket 06's own end-to-end zoom test, LOG.md): tree.bodies_type
  // is a plain host std::vector<int>, but sort_bodies()'s SFC sort (sort_bodies_gpu.cpp) never
  // reorders it -- only tree.bodies_ids (among host-observable arrays) is unconditionally
  // reordered by every sort_bodies() call, in BOTH its doFullShuffle branches. Tickets 03-05 never
  // caught this because every prior test IC used a single uniform particle type (permuting an
  // array of identical values is a no-op); a real multi-type IC silently misaligns bodies_type
  // (and therefore bodies_forceSoftening, the timestep type-dependent softening lookup, and the
  // zoom particle-type mask) with the actual post-sort particle order. Fixed by never trusting
  // bodies_type's own array index: `setGadgetTypeMap()` records the STABLE (particle ID -> type)
  // association once, from the original IC load order (main.cpp, before any sort ever runs);
  // `recomputeSoftening()` then rebuilds tree.bodies_type from the CURRENT (always correctly
  // reordered) tree.bodies_ids through this map every tree rebuild, before using it for anything.
  void   setGadgetTypeMap(const std::vector<ullong> &ids, const std::vector<int> &types);
  std::unordered_map<ullong, int> gadgetIdToType;

  // T35 performance: the (id -> type) association as a FLAT table indexed by particle id, plus its
  // device copy. Semantically identical to gadgetIdToType above -- same rule, same source -- but it
  // lets recomputeSoftening() run entirely on the device instead of moving 1.07 GB of ids to the
  // host, doing 134M hash lookups, and pushing 1.07 GB back every rebuild (measured: 1.78 s of a
  // 5.4 s sparse step at 512^3). Built once, at IC load, by setGadgetTypeMap().
  //
  // Only usable when the ids are dense and small enough to index directly; Gadget ICs number
  // particles 1..N, but nothing guarantees it, so setGadgetTypeMap() checks and leaves this empty
  // if they are not. recomputeSoftening() then falls back to the original host+map path, which is
  // kept precisely so an unusual IC stays correct rather than fast.
  std::vector<unsigned char>       gadgetTypeById;     // index: particle id; value: type
  my_dev::dev_mem<unsigned char>   gadgetTypeByIdDev;
  bool                             gadgetTypeByIdOk = false;
  ullong                           gadgetTypeByIdMax = 0;
  // Which of the six Gadget types this IC actually contains. The C-C-21 softening check runs on the
  // 6-entry table rather than on the expanded per-particle array, and an unpopulated type routinely
  // has softening 0 in a legitimate parameter file (SofteningDisk/Bulge/Stars/Bndry are all 0 in
  // the 512^3 reference), so checking every table slot unconditionally would warn on every run.
  int                              gadgetTypeUsed[6] = {0,0,0,0,0,0};
  // T35: 6-bin device scratch for the per-type timestep reduction (count, sum |v|^2, min mass).
  my_dev::dev_mem<double>             dtAggVSum;
  my_dev::dev_mem<double>             dtAggMinMass;
  my_dev::dev_mem<unsigned long long> dtAggCount;
  // Recomputes tree.bodies_forceSoftening from tree.bodies_type + the current gadgetParams (or
  // the flat eps-based fallback) and the current sim time, then uploads it -- called once per
  // tree rebuild (gpu_iterate.cpp), matching Gadget-2's own set_softenings() call granularity
  // (gravtree.c:51: recomputed before each new force-tree build, not every timestep).
  void   recomputeSoftening(tree_structure &tree);

  // Node storage is sized as TreeAllocFactor * n_bodies, read from the parameter file exactly as
  // Gadget-2 does (read_ic.c:344 / init.c:131), because the node count is a property of the
  // particle DISTRIBUTION, not of N -- it grows as structure forms, and grows much harder in a
  // zoom-in run's refined region. Measured on this project's own 512^3 reference run, nodes/N is
  // 0.157 at z=99, 0.172 at z=19, 0.277 at z=3.1 and 0.276 at z=0: a 1.8x rise over the run, so a
  // value picked from the IC alone would be an overrun waiting to happen. No single compile-time
  // constant can be both safe for a zoom run and economical for a uniform box -- which is exactly
  // why Gadget exposes it.
  //
  // Default matches Gadget-2's own TreeAllocFactor default, giving ~2.9x headroom over the
  // measured 512^3 peak (0.8 / 0.276). That figure is only correct with C-A-10's fix in place:
  // the per-level bound check used to add gpuCompact's MARKER count rather than the node count
  // (two markers per node, build_tree.cu:331-332 vs :358), so the effective headroom was half
  // this -- 1.45x -- and the abort fired at 0.5 * TreeAllocFactor * N. Overrunning it is a hard error naming the parameter, never a silent
  // walk past the buffer (which is what the previous hard-coded bound was defending against).
  double treeAllocFactor() const;

  // C-A-07 probe: does the between-rebuild AABB refit shrink nodes? See its definition in
  // gpu_iterate.cpp. No-op unless GADGET_HIP_CA07 is set.
  void   ca07_probe(tree_structure &tree, bool wasRebuild, int iter);

  // C-A-19 / C-A-21 invariant probe; also the direct test of scanKernels.cu's scans, since they
  // produce the leaf partition. No-op unless GADGET_HIP_INVARIANTS is set.
  void   invariant_probe(tree_structure &tree, int iter);
  void   spike_probe(tree_structure &tree, int iter, const char *where);

// C-A-11: the [BUG4-STAGE] / [BUG5-STAGE] / [BUG7-DIAG4] markers are residue from the prior audit's
// T4/T5/T7 investigations, all of which are closed. Each is a hipDeviceSynchronize() plus a stderr
// write, and they sit in the PRODUCTION tree-build path: ~90 forced device syncs and ~90 stderr
// writes per rebuild, at rebuild_tree_rate=1, none of it gated. Gadget-2's force_treebuild_single
// has no I/O in its loops. They are kept rather than deleted -- they earned their place -- but are
// now off unless GADGET_HIP_STAGE_TRACE is set, which is also what makes any timing measurement in
// this area meaningful (a per-level sync penalises deep trees, i.e. exactly the NLEAF comparison).
inline bool gadget_hip_stage_trace()
{
  static const bool on = (getenv("GADGET_HIP_STAGE_TRACE") != NULL);
  return on;
}
#define STAGE_TRACE(...)                                        \
  do {                                                          \
    if (gadget_hip_stage_trace()) {                             \
      hipDeviceSynchronize();                                   \
      fprintf(stderr, __VA_ARGS__);                             \
    }                                                           \
  } while (0)
  // Phase 5 ticket 04 (PLAN.md): recomputes the global RMS-displacement timestep ceiling
  // (find_dt_displacement_constraint(), timestep.c:566-644) from tree.bodies_type + a d2h copy of
  // masses/velocities -- same tree-rebuild call granularity as recomputeSoftening() (this is a
  // real device-to-host sync, not something to do every timestep). Result cached in
  // gadgetDtDisplacement, consumed every step by compute_dt() via correct()'s set_args() call.
  void   recomputeTimestepGlobals(tree_structure &tree);
  double gadgetDtDisplacement = 1.0e30;
  // Phase 5 ticket 05 (PLAN.md): builds the drift/grav-kick tables (gadget_driftfac.cpp) from
  // gadgetParams' TimeBegin/TimeMax/Omega0/OmegaLambda/Hubble and uploads them to
  // gadgetDriftTable/gadgetGravKickTable -- call once, after setGadgetParams(), only when
  // ComovingIntegrationOn (a no-op otherwise, leaving haveGadgetComovingTables false).
  void   initComovingTables();
  // Phase 5 ticket 08 (PLAN.md): writes one Gadget-2-format snapshot (gadget_snapshot_write(),
  // ticket 02) of the current particle state to gadgetParams.OutputDir, named
  // "<SnapshotFileBase>_NNN" per Gadget-2's own convention -- a no-op when !haveGadgetParams.
  // Called from gpu_iterate.cpp's per-step loop when t_current crosses nextGadgetSnapTime (and
  // once, unconditionally, at the final step) -- see nextGadgetSnapTime's own declaration comment.
  void   writeGadgetSnapshot(bool preKick = false);

  // PLAN 4.1 deliverable 1: per-buffer device-memory inventory (GADGET_HIP_MEMREPORT).
  void   reportDeviceMemory(const char *stage);

#ifdef GADGET_HIP_HIGHRES
  // Phase 5 ticket 06 (PLAN.md): PLACEHIGHRESREGION zoom config -- runtime, not compile-time
  // (pm_zoom.h's own documented divergence from upstream). Set once from main.cpp's
  // --zoom-mask/--zoom-enlarge CLI flags.
  unsigned int gadgetZoomMask    = 0;
  float        gadgetZoomEnlarge = 1.0f; // Gadget-2's real default: ENLARGEREGION is normally undefined (Makefile:24), so no enlargement
  bool         haveGadgetZoom    = false;
  ZoomRegion       gadgetZoomRegion;
  PMIsolatedSolver gadgetZoomSolver;
  bool             gadgetZoomSolverReady = false;
  void setZoomConfig(unsigned int mask, float enlargeRegion);
  // (Re)computes gadgetZoomRegion from the tree's current particle positions/types and
  // (re)builds gadgetZoomSolver's cached kernel for the new region -- called once at startup and
  // reactively whenever a high-res-typed particle is found outside the current region
  // (pm_zoom_region_out_of_range(), matching Gadget-2's own unconditional recompute-and-retry,
  // PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.3). `asmth0`/`rcut0`/`gridSize` describe the COARSE grid
  // (periodic BoxSize-based or isolated meshSize-based, whichever this build uses) -- the caller
  // (gpu_iterate.cpp) already computes these for the coarse grid's own tree-kernel setup.
  void recomputeZoomRegion(tree_structure &tree, double asmth0, double rcut0, int gridSize,
                            float periodicBoxSize);
#endif

  void   direct_gravity(tree_structure &tree);
  void   computeGroupMaxAccel(tree_structure &tree);  //Phase 3: Springel MAC per-group |a_old|
  void   correct(tree_structure &tree);
  double compute_energies(tree_structure &tree, bool usePreKickState = false);

  //Parallel version functions

  int procId, nProcs;                   //Process ID in the mpi stack, number of processors in the commm world
  int sharedPID;                        //Shared process ID to be used with the shared memory buffers
  unsigned long long  nTotalFreq_ull;   //Total Number of particles over all processes


  double prevDurStep;   //Duration of gravity time in previous step
  double thisPartLETExTime;     //The time it took to communicate with the neighbours during the last step

  double4 *currentRLow, *currentRHigh;  //Contains the actual domain distribution, to be used
                                        //during the LET-tree generatino

  real4 *globalGrpTreeCntSize;

  uint *globalGrpTreeCount;
  uint *globalGrpTreeOffsets;

  int  *fullGrpAndLETRequest;
  int2 *fullGrpAndLETRequestStatistics;

  std::vector<int> infoGrpTreeBuffer;
  std::vector<int> exchangePartBuffer;


  float maxExecTimePrevStep;      //Maximum duration of gravity computation over all processes
  float avgExecTimePrevStep;      //Average duration of gravity computation over all processes


  int grpTree_n_nodes;
  int grpTree_n_topNodes;

  real4 rMinLocalTree;            //for particles
  real4 rMaxLocalTree;            //for particles

  real4 rMinGlobal;
  real4 rMaxGlobal;
  
  bool letRunning;
  
  sampleRadInfo *curSysState;

  //Functions
  void mpiSetup();


  //Utility
  void      mpiSync();
  int       mpiGetRank();
  int       mpiGetNProcs();
  void      AllSum(double &value);
  int       SumOnRootRank(int value);
  double    SumOnRootRank(double value);

  //Main MPI functions

  //Functions for domain division
  void mpiSumParticleCount(int numberOfParticles);

  void ICRecv(int procId, vector<real4> &bodyPositions, vector<real4> &bodyVelocities,  vector<ullong> &bodiesIDs);
  void ICSend(int destination, real4 *bodyPositions, real4 *bodyVelocities,  ullong *bodiesIDs, int size);


  void sendCurrentRadiusInfo(real4 &rmin, real4 &rmax);
  
  //Function for Device assisted domain division
  int gpu_exchange_particles_with_overflow_check_SFC2(tree_structure &tree,
                                                    bodyStruct *particlesToSend,
                                                    int *nparticles, int *nsendDispls, int *nreceive,
                                                    int nToSend);
  void approximate_gravity_let(tree_structure &tree, tree_structure &remoteTree,
                                 int bufferSize, bool doActivePart);



   //Local Essential Tree related functions
  void build_NewTopLevels(int n_bodies,
                       uint4 *keys,
                       uint2 *nodes,
                       uint4 *node_keys,
                       uint  *node_levels,
                       int &n_levels,
                       int &n_nodes,
                       int &startGrp,
                       int &endGrp);

  void computeProps_TopLevelTree(
      int topTree_n_nodes,
      int topTree_n_levels,
      uint* node_levels,
      uint2 *nodes,
      real4* topTreeCenters,
      real4* topTreeSizes,
      real4* topTreeMultipole,
      real4* nodeCenters,
      real4* nodeSizes,
      real4* multiPoles,
      double4* tempMultipoleRes);

  void makeLET();

  void parallelDataSummary(tree_structure &tree, float lastExecTime, float lastExecTime2, double &domUpdate, double &domExch, bool initalSetup);


  void gpuRedistributeParticles_SFC(uint4 *boundaries);

  void build_GroupTree(int n_bodies, uint4 *keys, uint2 *nodes, uint4 *node_keys, uint  *node_levels,
                       int &n_levels, int &n_nodes, int &startGrp, int &endGrp);

  void computeProps_GroupTree(real4 *grpCenter, real4 *grpSize, real4 *treeCnt,
                              real4 *treeSize,  uint2 *nodes,   uint  *node_levels, int    n_levels);

  void sendCurrentInfoGrpTree();

  void exchangeSamplesAndUpdateBoundarySFC(uint4 *sampleKeys,    int  nSamples,
                                           uint4 *globalSamples, int  *nReceiveCnts, int *nReceiveDpls,
                                           int    totalCount,   uint4 *parallelBoundaries, float lastExectime,
                                           bool initialSetup);

  void essential_tree_exchangeV2(tree_structure &tree,
                                 tree_structure &remote,
                                 vector<real4> &topLevelTrees,
                                 vector<uint2> &topLevelTreesSizeOffset,
                                 int     nTopLevelTrees);

  void mergeAndLaunchLETStructures(tree_structure &tree, tree_structure &remote,
                                   real4 **treeBuffers,  int* treeBuffersSource, int &topNodeOnTheFlyCount,
                                   int &recvTree, bool &mergeOwntree, int &procTrees, double &tStart);

  void checkGPUAndStartLETComputation(tree_structure &tree,
                                      tree_structure &remote,
                                      int            &topNodeOnTheFlyCount,
                                      int            &nReceived,
                                      int            &procTrees,
                                      double         &tStart,
                                      double         &totalLETExTime,
                                      bool            mergeOwntree,
                                      int            *treeBuffersSource,
                                      real4         **treeBuffers);


  int recursiveTopLevelCheck(uint4 checkNode, real4* treeBoxSizes, real4* treeBoxCenters, real4* treeBoxMoments,
                          real4* grpCenter, real4* grpSize, int &DistanceCheck, int &DistanceCheckPP, int maxLevel);

  //End functions for parallel code


  //Library interface functions  
  void  setEps(float eps);
  float getEps();
  void  setDt(float dt);
  float getDt();
  void  setTheta(float theta);
  float getTheta();
  void  setTEnd(float tEnd);
  float getTEnd();
  void  setTime(float);
  float getTime();
  float getPot();
  float getKin();

  //End library functions

  // Phase 2: ticks are the state of record. Setting a scale factor (IC epoch, restart, snapshot
  // resume) re-derives them; if the timeline is not up yet, initTimeline() does it afterwards.
  void set_t_current(const float t)
  {
    t_current = t_previous = t;
    if (gadgetDPerTick > 0.0) Ti_current = Ti_previous = gadgetTimeline.toTick((double) t);
  }
  // Called once before the step loop. With --param the span comes from TimeBegin/TimeMax; without
  // it (the --plummer/--cube development paths) the timeline is linear over [t_current, tEnd].
  void initTimeline(double tBegin, double tMax, int comoving)
  {
    gadgetTimeline.init(tBegin, tMax, comoving);
    gadgetDPerTick = gadgetTimeline.dPerTick;
    Ti_current = Ti_previous = gadgetTimeline.toTick((double) t_current);
  }
  void set_nextSnapTime(const float t) { nextSnapTime = t; }
  float get_t_current() const       { return t_current; }
  gadget_tick_t get_Ti_current() const { return Ti_current; }
  void setUseDirectGravity(bool s)  { useDirectGravity = s;    }
  bool getUseDirectGravity() const  { return useDirectGravity; }
  void setErrTolForceAcc(float e)   { errTolForceAcc = e;      }
  float getErrTolForceAcc() const   { return errTolForceAcc;   }

  octree(const MPI_Comm &comm,
         my_dev::context *devContext_,
         char **argv, const int device = 0, const float _theta = 0.75, const float eps = 0.05,
         string snapF = "", float snapI = -1,  
         const float _quickDump       = 0.0,
         const float _quickRatio      = 0.1,
         const bool  _quickSync       = true,
         const bool  _useMPIIO        = false,
         const bool  _mpiRenderMode   = false,
         float tempTimeStep           = 1.0 / 16.0,
         float tempTend               = 1000,
         int _iterEnd                 = (1<<30),
         const int _rebuild           = 2,
         bool direct                  = false,
         const int shrdpid            = 0)
  : devContext(devContext_), mpiCommWorld(comm), rebuild_tree_rate(_rebuild), procId(0), nProcs(1),
    thisPartLETExTime(0), useDirectGravity(direct), errTolForceAcc(ERR_TOL_FORCE_ACC), quickDump(_quickDump), quickRatio(_quickRatio),
    quickSync(_quickSync), useMPIIO(_useMPIIO), mpiRenderMode(_mpiRenderMode), nextQuickDump(0.0), sharedPID(shrdpid)
  {
    iter            = 0;
    t_current       = t_previous = 0;
    Ti_current      = Ti_previous = 0;
    gadgetDPerTick  = 0.0;
    src_directory   = NULL;

    if(argv != NULL)  execPath = argv[0];

    localTree.n = 0;

    devContext = devContext_;

    //Setup the MPI processName and some buffers
    mpiSetup();


    statisticsIter = 0; //0=disabled, 1 = Every N-body unit, 2= every 2nd n-body unit, etc..
    nextStatsTime  = 0;
    nextSnapTime   = 0;

    snapshotIter      = snapI;
    snapshotFile      = snapF;
    store_energy_flag = true;

    timeStep = tempTimeStep;
    tEnd     = tempTend;
    iterEnd  = _iterEnd;

    //Theta, time-stepping
    inv_theta   = 1.0f/_theta;
    eps2        = eps*eps;
    eta         = 0.02f;
    theta       = _theta;

    const float dt_max = 1.0f / (1 << 4); //Calc dt_limit
    dt_limit = int(-log(dt_max)/log(2.0f));

    execStream          = NULL;
    gravStream          = NULL;
    copyStream          = NULL;
    LETDataToHostStream = NULL;
    
    infoGrpTreeBuffer. resize(7*nProcs);
    exchangePartBuffer.resize(8*nProcs);

    globalGrpTreeCntSize = NULL;


    maxExecTimePrevStep = 100;    //Some large values to force updates
    avgExecTimePrevStep = 1;      //Some large values to force updates


    //An initial guess for group broadcasted information
    //We set the statistics for our neighboring processes
    //to 1 and all remote ones to 0 for starters
    fullGrpAndLETRequest           = new int[nProcs];
    fullGrpAndLETRequestStatistics = new int2[nProcs];

    for(int i=0; i < nProcs; i++)
    {
      fullGrpAndLETRequestStatistics[i] = make_int2(0,0);
    }

    prevDurStep = -1;   //Set it to negative so we know its the first step

    
#ifdef USE_MPI
    logFileWriter = new LOGFILEWRITER(nProcs, myComm->MPI_COMM_I, myComm->MPI_COMM_J);
#else
    logFileWriter = new LOGFILEWRITER(nProcs, 0, 0);
#endif


    fileIO = new tipsyIO();


#ifdef WIN32
    // initialize windows timer
    QueryPerformanceFrequency(&sysTimerFreq);
    QueryPerformanceCounter(&sysTimerAtStart);
#endif
  }
  ~octree() {
    delete[] currentRLow;
    delete[] currentRHigh;
    delete[] curSysState;

    delete logFileWriter;
    delete fileIO;

    if(globalGrpTreeCntSize) delete[] globalGrpTreeCntSize;
    if(globalGrpTreeCount)   delete[] globalGrpTreeCount;
    if(globalGrpTreeOffsets) delete[] globalGrpTreeOffsets;

    if(fullGrpAndLETRequest)           delete[] fullGrpAndLETRequest;
    if(fullGrpAndLETRequestStatistics) delete[] fullGrpAndLETRequestStatistics;
  };
};



#endif // _OCTREE_H_
