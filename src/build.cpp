#include "hip/hip_runtime.h"
#include "octree.h"
#include "pm.h"
#include "build.h"

// See octree.h's treeAllocFactor() for why this is a parameter and not a constant.
// Gadget-2's own default (Gadget-2.0.7/Gadget2 parameterfiles) is 0.8.
static const double DEFAULT_TREE_ALLOC_FACTOR = 0.8;

double octree::treeAllocFactor() const
{
  // GADGET_HIP_TREE_ALLOC_FACTOR overrides both, for the runs that have no parameter file at all
  // (--plummer, --direct and the other built-in generators). Without it those paths are stuck at
  // the default, which is fine for NLEAF=16 but not for the NLEAF sweep that C-A-02 needs.
  static const double envFactor = []() {
    const char *e = getenv("GADGET_HIP_TREE_ALLOC_FACTOR");
    return e ? atof(e) : 0.0;
  }();
  if (envFactor > 0.0) return envFactor;

  double f = (haveGadgetParams && gadgetParams.TreeAllocFactor > 0.0)
             ? gadgetParams.TreeAllocFactor
             : DEFAULT_TREE_ALLOC_FACTOR;
  // A typo'd tiny value would otherwise fault before the diagnostic below could ever print.
  if (f < 0.05) f = 0.05;
  return f;
}

// BUG7 investigation (T4, gadget-audit) -- defined in build_tree.cu, see comment there.
extern "C" void bug7_reset_diag(int validListLen);
extern "C" void bug7_read_diag(int *maxOobWriteIdx, int *maxOobLastChild, int *oobCount);

void octree::allocateParticleMemory(tree_structure &tree)
{
  //Allocates the memory to hold the particles data
  //and the arrays that have the same size as there are
  //particles. Eg valid arrays used in tree construction
  int n_bodies = tree.n;


  //MULTI_GPU_MEM_INCREASE% extra space, only in parallel when
  if(nProcs > 1) n_bodies = (int)(n_bodies*MULTI_GPU_MEM_INCREASE);    //number of particles can fluctuate

  //Particle properties
  tree.bodies_pos.cmalloc(1, true);              // bound to bodies_Ppos, see gpu_iterate.cpp   //+1 to set end pos, host mapped? TODO not needed right since we use Ppos
  tree.bodies_key.cmalloc(n_bodies+1, false);   //+1 to set end key
  tree.bodies_ids.cmalloc(n_bodies+1, false);   //+1 to set end key

  tree.bodies_Ppos.cmalloc(n_bodies+1, true);   //Memory to store predicted positions, host mapped
  tree.bodies_Pvel.cmalloc(n_bodies+1, true);   //Memory to store predicted velocities, host mapped

  tree.bodies_vel.cmalloc(n_bodies, false);
  tree.bodies_acc0.ccalloc(n_bodies, false);    //ccalloc -> init to 0
  tree.bodies_acc1.ccalloc(n_bodies, false);    //ccalloc -> init to 0
  tree.bodies_time.ccalloc(n_bodies, false);    //ccalloc -> init to 0

  //density
#if BONSAI_DENSITY
  tree.bodies_h.cmalloc(n_bodies, true);
#else
  tree.bodies_h.cmalloc(1, true);             // density subsystem off, see node_specs.h
#endif
#if BONSAI_DENSITY
  tree.bodies_dens.cmalloc(n_bodies, true);
#else
  tree.bodies_dens.cmalloc(1, true);          // density subsystem off, see node_specs.h
#endif
  //Init to -1
#if BONSAI_DENSITY
  for(int i=0; i < n_bodies; i++) tree.bodies_h[i] = -1;
  tree.bodies_h.h2d();
#endif

  // Phase 5 ticket 03 (PLAN.md): per-particle resolved Gadget-2 ForceSoftening, refreshed by
  // octree::recomputeSoftening() before it's ever read by the gravity kernel -- initial content
  // here is never actually consumed, but zero-init (via ccalloc, not cmalloc) is still safer than
  // leaving it as garbage in case some future caller reads it before the first tree rebuild.
  tree.bodies_forceSoftening.ccalloc(n_bodies, true);
  tree.bodies_type.resize(n_bodies, 1); // default Gadget-2 type 1 (Halo/DM); main.cpp overwrites

  // Phase 5 ticket 06 (PLAN.md): on-device mirror of bodies_type, refreshed alongside
  // bodies_forceSoftening in octree::recomputeSoftening() so it stays correctly reordered after
  // every tree rebuild's SFC sort. Originally only needed by the GADGET_HIP_HIGHRES-only
  // pm_add_masked_force_to_acc/pm_add_masked_potential_to_acc; made UNCONDITIONAL in Ticket 07
  // (PLAN.md) since the tree-walk gravity kernel now also reads it (for the per-target Rcut/Asmth
  // selection) in every build, not just HIGHRES ones.
  tree.bodies_typeDevice.ccalloc(n_bodies, false);


  tree.oriParticleOrder.cmalloc(n_bodies, false);      //To desort the bodies tree later on
  //iteration properties / information
  tree.activePartlist.ccalloc(n_bodies+2, false);   //+2 since we use the last two values as a atomicCounter (for grp count and semaphore access)
  tree.ngb.ccalloc(1, false);           // write-only, see dev_approximate_gravity_warp_new.cu
  tree.interactions.cmalloc(1, false);  // INTCOUNT off, see dev_approximate_gravity_warp_new.cu

  tree.body2group_list.cmalloc(n_bodies, false);

  tree.level_list.cmalloc(MAXLEVELS);
  tree.node_level_list.cmalloc(MAXLEVELS*2 , false);


  //The generalBuffer is also used during the tree-walk, so the size has to be at least
  //large enough to store the tree-walk stack. Add 4096 for extra memory alignment space
  //Times 2 since first half is used for regular walks, 2nd half for walks that go outside
  //the memory stack and require another walk with 'unlimited' stack size
#if 0
  size_t treeWalkStackSize = (2ULL*LMEM_STACK_SIZE*NTHREAD*(size_t)nBlocksForTreeWalk) + 4096;
#else
  size_t treeWalkStackSize = (2ULL*(LMEM_STACK_SIZE*NTHREAD + LMEM_EXTRA_SIZE)*(size_t)nBlocksForTreeWalk) + 4096;
#endif


  // Tree properties, tree size is not known at fore hand so allocate worst possible outcome.
  // 2026-08-30 (PLAN.md's NCRIT=1 de-grouping investigation, LOG.md): the old formula here
  // (tempmem = n_bodies) assumed leaf count stays comfortably below n_bodies -- true only for
  // NLEAF>1, where each leaf absorbs multiple particles. With NLEAF=1 (a leaf per particle),
  // leaf count approaches n_bodies itself, and ANY tree where every internal node has >=2
  // children needs internal_nodes <= leaves-1 more nodes on top of that (standard tree-counting
  // property) -- so total node count can approach 2*n_bodies, not n_bodies. Confirmed as the real
  // cause of a hard GPU memory-access fault at N=134,217,728 / NCRIT=1 / NLEAF=1 (crashed exactly
  // at build()'s per-level loop, level=9 -- matching log8(134M)~9, the depth needed to resolve
  // individual particles at that N): node_key/n_children/node_bodies, all sized off this same
  // tempmem, silently walked past their buffer with no bounds check anywhere in the loop. Doubled
  // to a derivation-backed safe worst case rather than a guessed larger constant.
  // Was a hard-coded 2.0 * n_bodies -- the NLEAF=1 worst case derived above. That bound is
  // correct but enormously pessimistic for the NLEAF=16 the code actually builds with: measured
  // at 512^3 the real node count is 0.16-0.28 * n_bodies, so 2.0 over-allocated node_bodies,
  // n_children and node_key by ~7x (6.6 GB at 512^3). Now TreeAllocFactor, Gadget-style, with an
  // explicit per-level bound check in build() below instead of a worst-case constant.
  size_t tempmem = (size_t)(treeAllocFactor() * (double)n_bodies);
  if (tempmem < 2048) tempmem = 2048;

  tree.n_children.cmalloc (tempmem, false);
  tree.node_bodies.cmalloc(tempmem, false);

  // generalBuffer1 must simultaneously hold, during build()'s per-level loop: validList
  // (2*n_bodies uints) + compactList (2*n_bodies uints) + node_key (4*tempmem uints, a uint4
  // buffer) + levelOffset (256) + maxLevel (256). Made explicit here (rather than the old
  // "3*n*4"-style formula, whose margin over the OLD, smaller node_key silently evaporated once
  // node_key grew with the tempmem fix above) with a 10% safety margin, matching this project's
  // own MULTI_GPU_MEM_INCREASE convention used elsewhere for the same kind of "actual count not
  // known ahead of time" sizing. long arithmetic to stay clear of int32 overflow at this scale.
  // generalBuffer1 is a SHARED arena, and its size must be the max over every consumer -- not
  // just build()'s. The other big one is the per-iteration particle reorder
  // (sort_bodies_gpu.cpp): real4Buffer1(4N) + float2Buffer(2N) + ullBuffer(2N) + realBuffer(1N)
  // = 9N uints, plus the radix-sort scratch (srcValues 4N + tempB..tempE 4N = 8N), both entirely
  // independent of the node count. That requirement used to be met only by accident, because the
  // node scratch below was sized at a pessimistic 2*N; with TreeAllocFactor it no longer is, so
  // it is stated explicitly here. cmalloc_copy now also bounds-checks, so getting this wrong is
  // a named error rather than a GPU page fault.
  // Every consumer, stated explicitly and in uints. Anything not listed here is smaller than the
  // max of these, and cmalloc_copy bounds-checks besides, so an omission is a named error rather
  // than a page fault.
  //   gpuSort   srcValues(uint4 = 4N) + tempB/C/D(uint = 3N) + tempE(char = 0.25N). These are all
  //             passed to ONE gpuSort call and are live simultaneously -- they cannot share, and
  //             this is the real floor on the arena (C-A-22).
  //   reorder   one shared region, C-A-22: the largest single dataReorder view is real4 = 4N.
  //   timestep  compute_dt + correct_particles, gpu_iterate.cpp: float2(2N)+real4(4N)+float(1N).
  //   build     this function's own per-level layout, below.
  const size_t sortScratchNeed     =  8ULL*(size_t)n_bodies + 4096ULL;
  const size_t reorderScratchNeed  =  4ULL*(size_t)n_bodies + 4096ULL;
  const size_t timestepScratchNeed =  7ULL*(size_t)n_bodies + 4096ULL;
  const size_t buildScratchNeed = 2ULL*n_bodies + 2ULL*n_bodies + 4ULL*tempmem + 512ULL;
  size_t generalBuffer1Need = std::max(std::max(buildScratchNeed, sortScratchNeed),
                                       std::max(reorderScratchNeed, timestepScratchNeed));
  // Bug 9 / PLAN 4.1 deliverable 5: this was `int tempSize = (int)(... * 11 / 10)`. The `long`
  // arithmetic above was deliberate and correct, and then truncated to int on assignment.
  // tempSize is ~13.2*n elements, so it overflowed int32 at n ~ 1.63e8 -- above 512^3 (1.34e8)
  // but below the 3e8 target. Kept in size_t end to end.
  size_t tempSize = (generalBuffer1Need * 11 / 10) + 4096;
  tempSize        = std::max(tempSize, treeWalkStackSize);

  //General buffer is used at multiple locations and reused in different functions
  tree.generalBuffer1.cmalloc(tempSize, true);

  //General memory buffers

  //Allocate shared buffers
  this->tnext.		  ccalloc(NBLOCK_REDUCE,false);
  this->nactive.	  ccalloc(NBLOCK_REDUCE,false);
  this->devMemRMIN.   cmalloc(NBLOCK_BOUNDARY, false);
  this->devMemRMAX.	  cmalloc(NBLOCK_BOUNDARY, false);
  this->devMemCounts. cmalloc(NBLOCK_PREFIX, false);
  this->devMemCountsx.cmalloc(NBLOCK_PREFIX, true);

  if(mpiGetNProcs() > 1)
  {
    int remoteSize = (int)(n_bodies*0.5); //TODO some more realistic number
    if(remoteSize < 1024 ) remoteSize = 2048;

    this->remoteTree.fullRemoteTree.cmalloc(remoteSize, true);
    tree.parallelBoundaries.cmalloc(mpiGetNProcs()+1, true);
  }

}


void octree::reallocateParticleMemory(tree_structure &tree)
{
  //Reallocate the memory to hold the particles data
  //and the arrays that have the same size as there are
  //particles. Eg valid arrays used in tree construction
  int n_bodies = tree.n;


  if(tree.activePartlist.get_size() < tree.n)
    n_bodies *= MULTI_GPU_MEM_INCREASE;


  bool reduce = false;  //Set this to true to limit memory usage by only allocating what
                        //is required. If its false, then memory is not reduced and a larger
                        //buffer is kept

  //Particle properties
  tree.bodies_pos.cresize(1, reduce);            // bound to bodies_Ppos, see gpu_iterate.cpp   //+1 to set boundary condition
  tree.bodies_key.cresize(n_bodies+1, reduce);   //+1 to set boundary condition
  tree.bodies_ids.cresize(n_bodies+1, reduce);   //

  tree.bodies_Ppos.cresize(n_bodies+1, reduce);   //Memory to store predicted positions
  tree.bodies_Pvel.cresize(n_bodies+1, reduce);   //Memory to store predicted velocities

  tree.bodies_vel.cresize (n_bodies, reduce);
  tree.bodies_acc0.cresize(n_bodies, reduce);    //ccalloc -> init to 0
  tree.bodies_acc1.cresize(n_bodies, reduce);    //ccalloc -> init to 0
  tree.bodies_time.cresize(n_bodies, reduce);    //ccalloc -> init to 0
  
  //Density
#if BONSAI_DENSITY
  tree.bodies_h.cresize(n_bodies, reduce);
#else
  tree.bodies_h.cresize(1, reduce);           // density subsystem off, see node_specs.h
#endif
#if BONSAI_DENSITY
  tree.bodies_dens.cresize(n_bodies, reduce);
#else
  tree.bodies_dens.cresize(1, reduce);        // density subsystem off, see node_specs.h
#endif
  
  tree.oriParticleOrder.cresize(n_bodies,   reduce);     //To desort the bodies tree later on
  //iteration properties / information
  tree.activePartlist.cresize(  n_bodies+2, reduce);      //+1 since we use the last value as a atomicCounter
  tree.ngb.cresize(            1,   reduce);
  tree.interactions.cresize(   1,   reduce);

  tree.body2group_list.cresize(n_bodies, reduce);

  //Tree properties, tree size is not known at forehand so
  //allocate worst possible outcome
  // 2026-08-30: same fix and rationale as allocateParticleMemory's own tempmem/tempSize
  // computation above (see its comment) -- NLEAF=1 can drive total node count toward 2*n_bodies,
  // not n_bodies. Kept in sync with that function deliberately; if you change one, change both.
  size_t tempmem = (size_t)(treeAllocFactor() * (double)n_bodies);
  if (tempmem < 2048) tempmem = 2048;
  tree.n_children.cresize(tempmem, reduce);
  tree.node_bodies.cresize(tempmem, reduce);


  //Don't forget to resize the generalBuffer....
  size_t treeWalkStackSize = (2ULL*(LMEM_STACK_SIZE*NTHREAD + LMEM_EXTRA_SIZE)*(size_t)nBlocksForTreeWalk) + 4096;

  // generalBuffer1 is a SHARED arena, and its size must be the max over every consumer -- not
  // just build()'s. The other big one is the per-iteration particle reorder
  // (sort_bodies_gpu.cpp): real4Buffer1(4N) + float2Buffer(2N) + ullBuffer(2N) + realBuffer(1N)
  // = 9N uints, plus the radix-sort scratch (srcValues 4N + tempB..tempE 4N = 8N), both entirely
  // independent of the node count. That requirement used to be met only by accident, because the
  // node scratch below was sized at a pessimistic 2*N; with TreeAllocFactor it no longer is, so
  // it is stated explicitly here. cmalloc_copy now also bounds-checks, so getting this wrong is
  // a named error rather than a GPU page fault.
  // Every consumer, stated explicitly and in uints. Anything not listed here is smaller than the
  // max of these, and cmalloc_copy bounds-checks besides, so an omission is a named error rather
  // than a page fault.
  //   gpuSort   srcValues(uint4 = 4N) + tempB/C/D(uint = 3N) + tempE(char = 0.25N). These are all
  //             passed to ONE gpuSort call and are live simultaneously -- they cannot share, and
  //             this is the real floor on the arena (C-A-22).
  //   reorder   one shared region, C-A-22: the largest single dataReorder view is real4 = 4N.
  //   timestep  compute_dt + correct_particles, gpu_iterate.cpp: float2(2N)+real4(4N)+float(1N).
  //   build     this function's own per-level layout, below.
  const size_t sortScratchNeed     =  8ULL*(size_t)n_bodies + 4096ULL;
  const size_t reorderScratchNeed  =  4ULL*(size_t)n_bodies + 4096ULL;
  const size_t timestepScratchNeed =  7ULL*(size_t)n_bodies + 4096ULL;
  const size_t buildScratchNeed = 2ULL*n_bodies + 2ULL*n_bodies + 4ULL*tempmem + 512ULL;
  size_t generalBuffer1Need = std::max(std::max(buildScratchNeed, sortScratchNeed),
                                       std::max(reorderScratchNeed, timestepScratchNeed));
  // Bug 9 / PLAN 4.1 deliverable 5: this was `int tempSize = (int)(... * 11 / 10)`. The `long`
  // arithmetic above was deliberate and correct, and then truncated to int on assignment.
  // tempSize is ~13.2*n elements, so it overflowed int32 at n ~ 1.63e8 -- above 512^3 (1.34e8)
  // but below the 3e8 target. Kept in size_t end to end.
  size_t tempSize = (generalBuffer1Need * 11 / 10) + 4096;
  tempSize        = std::max(tempSize, treeWalkStackSize);

  //General buffer is used at multiple locations and reused in different functions
  tree.generalBuffer1.cresize(tempSize, reduce);

  my_dev::base_mem::printMemUsage();
}

void octree::allocateTreePropMemory(tree_structure &tree)
{
  devContext->startTiming(execStream->s());
  int n_nodes = tree.n_nodes;

  //Allocate memory
  if(tree.groupCenterInfo.get_size() > 0)
  {
    if(tree.boxSizeInfo.get_size() <= n_nodes)
      n_nodes *= MULTI_GPU_MEM_INCREASE;

    //Resize, so we don't allocate if we already have mem allocated
    tree.multipole.cresize_nocpy(3*n_nodes,     false);
    tree.boxSizeInfo.cresize_nocpy(n_nodes,     false); //host allocated
    tree.boxCenterInfo.cresize_nocpy(n_nodes,   false); //host allocated
    tree.cellSizeInfo.cresize_nocpy(n_nodes,    false); //T26: fixed octree-cell size per node
    tree.cellCenterInfo.cresize_nocpy(n_nodes,  false); //C-A-01(b): fixed octree-cell centre per node
#ifdef UNEQUALSOFTENINGS
    tree.nodeSoftInfo.cresize_nocpy(n_nodes,    false); //T28/C-A-04: per-node softening summary
#endif

    int n_groups = tree.n_groups;
    if(tree.groupSizeInfo.get_size() <= n_groups)
      n_groups *= MULTI_GPU_MEM_INCREASE;

    tree.groupSizeInfo.cresize_nocpy(n_groups,   false);
    tree.groupCenterInfo.cresize_nocpy(n_groups, false);
    tree.groupMaxAccInfo.cresize_nocpy(n_groups, false);  //Phase 3: Springel MAC input
    tree.groupMaxSofteningInfo.cresize_nocpy(n_groups, false);  //T27: softening-based bJ floor input
  }
  else
  {
    //First call to this function
    n_nodes = (int)(n_nodes * 1.1f);
    tree.multipole.cmalloc(3*n_nodes, true); //host allocated

    tree.boxSizeInfo.cmalloc(n_nodes, true);     //host allocated
    tree.groupSizeInfo.cmalloc(tree.n_groups, true);
    tree.cellSizeInfo.cmalloc(n_nodes, true);    //T26: fixed octree-cell size per node
    tree.cellCenterInfo.cmalloc(n_nodes, true);  //C-A-01(b): fixed octree-cell centre per node
#ifdef UNEQUALSOFTENINGS
    tree.nodeSoftInfo.cmalloc(n_nodes, true);    //T28/C-A-04: per-node softening summary
#else
    // Not compiled in: one element, never read. Keeps .p() valid without paying n_nodes floats.
    tree.nodeSoftInfo.cmalloc(1, true);
#endif

    tree.boxCenterInfo.cmalloc(n_nodes, true); //host allocated
    tree.groupCenterInfo.cmalloc(tree.n_groups,true);
    tree.groupMaxAccInfo.cmalloc(tree.n_groups, true);    //Phase 3: Springel MAC input
    tree.groupMaxSofteningInfo.cmalloc(tree.n_groups, true);    //T27: softening-based bJ floor input
  }
  devContext->stopTiming("Memory", 11, execStream->s());
}

void octree::build (tree_structure &tree) {

  devContext->startTiming(execStream->s());
  int level      = 0;
  int validCount = 0;
  int offset     = 0;

  this->resetCompact();

  /******** create memory buffers **********/

  my_dev::dev_mem<uint>   validList;
  my_dev::dev_mem<uint>   compactList;
  my_dev::dev_mem<uint>   levelOffset;
  my_dev::dev_mem<uint>   maxLevel;
  my_dev::dev_mem<uint4>  node_key;



  int memBufOffset = validList.cmalloc_copy  (tree.generalBuffer1, tree.n*2, 0);
      memBufOffset = compactList.cmalloc_copy(tree.generalBuffer1, tree.n*2, memBufOffset);
  int memBufOffsetValidList = memBufOffset;

  // 2026-08-30: node_key must have the same worst-case node capacity as n_children/node_bodies
  // (allocateParticleMemory/reallocateParticleMemory's own tempmem, see their comments) -- all
  // three are indexed by the same running node offset during this function's per-level loop
  // below, so undersizing any one of them independently reintroduces the same overflow.
  // That coupling is why this MUST track treeAllocFactor() too: leaving it at the old hard-coded
  // 2*tree.n while the other two shrank did not merely waste memory, it pushed this function's
  // generalBuffer1 layout (4*n + 4*tempmem uints) past the end of the arena and wrote out of
  // bounds -- silently, with a plausible-looking result, until cmalloc_copy's bounds check
  // caught it.
  size_t tempmem = (size_t)(treeAllocFactor() * (double)tree.n);
  if (tempmem < 2048) tempmem = 2048;

  memBufOffset = node_key.cmalloc_copy   (tree.generalBuffer1, tempmem,  memBufOffset);
  memBufOffset = levelOffset.cmalloc_copy(tree.generalBuffer1, 256,      memBufOffset);
  memBufOffset = maxLevel.cmalloc_copy   (tree.generalBuffer1, 256,      memBufOffset);

  //Memory layout of the above (in uint):
  //[[validList--2*tree.n],[compactList--2*tree.n],[node_key--4*tempmem, tempmem~=2*tree.n],
  //[levelOffset--256], [maxLevel--256]] -- generalBuffer1's own size (allocateParticleMemory/
  //reallocateParticleMemory) is derived to match this layout with a 10% margin, not guessed.

  //Set the default values to zero
  validList.  zeroMemGPUAsync(execStream->s());
  levelOffset.zeroMemGPUAsync(execStream->s());
  maxLevel.   zeroMemGPUAsync(execStream->s());
  //maxLevel.zeroMem is required to let the tree-construction work properly.
  //It assumes maxLevel is zero for determining the start-level / min-level value.


  /******** set kernels parameters **********/



  build_valid_list.set_args(0, &tree.n, &level, tree.bodies_key.p(),  validList.p(), this->devMemCountsx.p());
  build_valid_list.setWork(tree.n, 128);

  build_nodes.set_args(0, &level, devMemCountsx.p(),  levelOffset.p(), maxLevel.p(),
                       tree.level_list.p(), compactList.p(), tree.bodies_key.p(), node_key.p(),
                       tree.n_children.p(), tree.node_bodies.p());
  build_nodes.setWork(vector<size_t>{120*32,4}, vector<size_t>{128,1});


  /******  build the levels *********/
  // make sure previous resetCompact() has finished.
  this->devMemCountsx.waitForCopyEvent();
//  devContext.startTiming(execStream->s());

  if(nProcs > 1)
  {
      LOGF(stderr,"Before copy ppos valid\n");
    //Start copying the particle positions to the host, will overlap with tree-construction
    localTree.bodies_Ppos.d2h(tree.n, false, LETDataToHostStream->s());
  }

//  double tBuild0 = get_time();

#if 0
  build_tree_node_levels(*this, validList, compactList, levelOffset, maxLevel, execStream->s());
#else
  size_t nodeSlotCapacity = tree.n_children.get_size();
  size_t nodeSlotsUsed    = 0;
  for (level = 0; level < MAXLEVELS; level++) {
    build_valid_list.execute2(execStream->s());         //Mark bodies to be combined into nodes
    STAGE_TRACE("[BUG5-STAGE] after build_valid_list level=%d\n", level);
    int newNodesThisLevel = 0;
    gpuCompact(validList, compactList, tree.n*2, &newNodesThisLevel);  //number of created nodes
    STAGE_TRACE("[BUG5-STAGE] after gpuCompact level=%d\n", level);
    // build_nodes is about to write node_key / n_children / node_bodies at [offset, offset+n).
    // All three are sized off tempmem = TreeAllocFactor * n_bodies, and none of the device kernels
    // bounds-check, so an under-sized factor used to be a silent out-of-bounds write that showed
    // up only as a GPU memory fault. Stop here instead, naming the parameter -- this is Gadget's
    // own force_treeallocate contract (forcetree.c: "failed to allocate memory for %d tree-nodes").
    // C-A-10: gpuCompact counts MARKERS, not nodes. cl_build_valid_list emits two per entry --
    // a start marker at bi and an end marker at bj-1 (build_tree.cu:331-332) -- which is exactly
    // why the consumer kernel halves it: `uint n = (*compact_list_len)/2;` (build_tree.cu:358).
    // This accumulator did not, so it counted every node twice: the abort fired at
    // 0.5 * TreeAllocFactor * N instead of TreeAllocFactor * N, silently halving a user-facing
    // parameter, and the headroom quoted in octree.h was wrong by the same factor of 2.
    nodeSlotsUsed += (size_t)newNodesThisLevel / 2;
    if (nodeSlotsUsed > nodeSlotCapacity)
    {
      fprintf(stderr,
        "\nFATAL: tree node storage exhausted at level %d.\n"
        "  nodes needed so far : %zu\n"
        "  nodes allocated     : %zu  (TreeAllocFactor %.3f x %d particles)\n"
        "Increase TreeAllocFactor in the parameter file and re-run. The node count grows as\n"
        "structure forms and is much higher for zoom-in runs and for small NLEAF (this build\n"
        "uses NLEAF=%d); for a uniform cosmological box 0.8 is ample, a highly refined zoom\n"
        "region can need several times that.\n\n",
        level, nodeSlotsUsed, nodeSlotCapacity, treeAllocFactor(), tree.n, NLEAF);
      exit(1);
    }
    build_nodes.execute2(execStream->s());              //Assemble the nodes
    STAGE_TRACE("[BUG5-STAGE] after build_nodes level=%d\n", level);
  } //end for level

  // reset counts to 1 so next compact proceeds...
  this->resetCompact();
#endif

//  execStream->sync();
//  const double dt = get_time() - tBuild0;
//  fprintf(stderr, " done in %g sec : %g Mptcl/sec\n", dt, tree.n/1e6/dt);
//  devContext.stopTiming("Create-nodes", 10, execStream->s());

  maxLevel.d2h(1);      level  = maxLevel[0];
  levelOffset.d2h(1);   offset = levelOffset[0];

  // BUG7 FIX (T7, gadget-audit): cl_build_nodes's *last_level device output (just read back above as
  // `level`) is a dual-purpose sentinel (build_tree.cu ~387-419): it's set to the literal integer 1
  // as a "minimum level reached" boolean the first time a level's node count exceeds
  // START_LEVEL_MIN_NODES, and gets RE-armed to that same literal 1 on every SUBSEQUENT level that
  // also exceeds START_LEVEL_MIN_NODES (`if (n > START_LEVEL_MIN_NODES) *last_level = 1;`, an
  // unconditional overwrite, not a one-time latch) -- it is only ever overwritten with the real
  // final level value when node creation naturally drops to n<=0 within this iteration's MAXLEVELS
  // budget. For an unusually dense/wide tree (a close encounter) that keeps producing new,
  // above-threshold nodes at every level all the way to MAXLEVELS, that n<=0 termination never
  // fires, so `level` is left stuck at the literal sentinel 1 even though the tree genuinely has many
  // real levels. Confirmed live via BUG7-DIAG4 (T4's resolution): `level=15` for many healthy
  // iterations, then a sudden discontinuous drop to `level=1` in the exact iteration that crashes.
  //
  // Fix: the per-level build loop above (`for (level = 0; level < MAXLEVELS; level++)`)
  // unconditionally launches cl_build_nodes for every one of the MAXLEVELS levels on every call to
  // build() (no early break), and cl_build_nodes's "last block" epilogue unconditionally writes
  // `level_list[level] = (n>0) ? (offset,offset+n) : (0,0)` on every single launch (build_nodes's
  // launch grid is a fixed 120*32*4 = 15360 blocks, so the `numBlocks > 1` last-block path always
  // executes, regardless of tree size) -- tree.level_list is therefore ALWAYS fully and freshly
  // populated for every index in [0, MAXLEVELS) after the loop, for THIS iteration, completely
  // independent of whether the buggy device-side sentinel got stuck. Reconstruct the true final
  // level directly from level_list's own (offset, offset+n) contents on the host, applying the same
  // termination rule the kernel intended (empty at i, previous level had real nodes) but computed
  // reliably here instead of racing a device sentinel that a later, still-dense level can silently
  // re-arm. Note this also correctly subsumes the fully-degenerate level<=1 case (level_list[0].x is
  // always 0 by construction -- the root's cumulative node offset -- so testing "previous level had
  // nodes" via (y>x) rather than the kernel's own (x>0) check is required to detect termination right
  // after level 0; using (x>0) here would reproduce the same blind spot on the device side).
  tree.level_list.d2h();
  {
    int reconstructedLevel = level;
    for (int i = 1; i < MAXLEVELS; i++)
    {
      bool emptyHere    = (tree.level_list[i].y <= tree.level_list[i].x);
      bool prevHadNodes = (tree.level_list[i-1].y > tree.level_list[i-1].x) || (tree.level_list[i-1].x > 0);
      if (emptyHere && prevHadNodes)
      {
        reconstructedLevel = i;
        break;
      }
      if (i == MAXLEVELS - 1)
      {
        // Node creation never naturally stopped within the whole per-iteration budget -- treat as
        // genuinely too deep so the existing bailout below fires, rather than silently accepting a
        // wrong shallow value.
        reconstructedLevel = MAXLEVELS;
      }
    }
    if (reconstructedLevel != level)
    {
      fprintf(stderr, "[BUG7-FIX-T7] raw device sentinel level=%d disagreed with level_list "
              "reconstruction reconstructedLevel=%d -- using reconstructed value\n",
              level, reconstructedLevel);
    }
    level = reconstructedLevel;
  }

  /***** Link the tree ******/



  //The maximum number of levels that can be used is MAXLEVEl
  LOG("Tree built with %d levels\n", level);
  if(level >= MAXLEVELS)
  {
    // C-A-08: this used to `exit(0)` -- a SUCCESS status on a fatal error, so any harness that
    // checks return codes records the abort as a completed run. And the advice named the wrong
    // cause: "far away particles / too large box" is the OPPOSITE of the condition that actually
    // reaches here. The tree deepens without bound when particles are too CLOSE -- >NLEAF bodies
    // sharing one SFC key produce a node at every level, which is Gadget-2's coincident-particle
    // case (it escapes via the subnode randomiser at forcetree.c:225-238; this port has no such
    // escape, contract C-A-03).
    fprintf(stderr,
      "\nFATAL: tree depth reached MAXLEVELS (%d) -- the build cannot terminate.\n"
      "This means more than NLEAF (%d) particles share one Morton key, i.e. they are at\n"
      "identical or near-identical positions: the key stops discriminating and a node is created\n"
      "at every level. Gadget-2 escapes this by randomising the subnode of coincident particles\n"
      "(forcetree.c:225-238); this port has no such escape (contract C-A-03).\n"
      "  particles : %d\n"
      "  box size  : %f  (key resolution %g, so positions closer than that are indistinguishable)\n"
      "Check the IC for duplicated particles rather than for distant ones.\n\n",
      MAXLEVELS, NLEAF, tree.n, tree.domain_fac * (1 << MAXLEVELS), tree.domain_fac);
    exit(1);
  }

  tree.n_nodes       = offset;
  tree.n_levels      = level-1;
  tree.startLevelMin = 0;
  for(int i=0; i < level; i++)
  {
    LOG("%d\t%d\t%d\n", i, tree.level_list[i].x, tree.level_list[i].y);
    //Determine which level is to used as min_level
    if(((tree.level_list[i].y - tree.level_list[i].x) > START_LEVEL_MIN_NODES) && (tree.startLevelMin == 0))
    {
      tree.startLevelMin = i;
    }
  }
  LOG("Start at: Level: %d  begin-end: %d %d \n", tree.startLevelMin,
      tree.level_list[tree.startLevelMin].x, tree.level_list[tree.startLevelMin].y);

  // BUG7 FIX (T4, gadget-audit): define_groups (build_group_list2) reads
  // tree.level_list[tree.startLevelMin+1] as its coarse-group boundary source. tree.level_list is
  // allocated ONCE (MAXLEVELS, near build()'s start) and only indices [0, level-1] are freshly
  // written by THIS iteration's per-level tree-build loop -- entries [level, MAXLEVELS-1] retain
  // STALE data from whichever earlier iteration last built a deeper tree (or uninitialized garbage
  // on the very first build ever). The loop above can legitimately set startLevelMin = level-1 (the
  // deepest level built this iteration, if it's the first level whose node-count exceeds
  // START_LEVEL_MIN_NODES) -- when that happens, startLevelMin+1 == level, an index NOT written
  // this iteration, and the kernel reads corrupted cross-iteration data as its group-boundary
  // range. Root-caused via a real crashing repro (phase0-ours.param close encounter, previously
  // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION inside build_group_list2): confirmed directly, the
  // diagnostic print immediately preceding the fault showed exactly this condition
  // (level=1, startLevelMin=0, startLevelMin+1==level, reading a stale level_list[1] left over from
  // an earlier, deeper tree) -- the close encounter collapses many particles to
  // Morton-key-indistinguishable positions (see T6/FINDINGS.md's duplicate-position finding),
  // driving the tree pathologically shallow (level=1 observed) right at the crash epoch.
  // Fix: clamp startLevelMin so startLevelMin+1 always references a level actually built THIS
  // iteration (startLevelMin+1 <= level-1, i.e. startLevelMin <= level-2). When level==1 (no level
  // beyond the root was built at all -- the fully-degenerate case), there is no valid "next level"
  // to reference at all; disable the coarse-group mechanism cleanly for this iteration by passing a
  // synthetic {0,1} range downstream (define_groups's own `idx < startLevelBeginEnd.y-1` guard
  // becomes `idx < 0`, always false for the kernel's unsigned idx -- a real, intentional no-op, not
  // a sentinel that happens to be safe by luck) rather than reading unwritten memory. The per-particle
  // idx%NCRIT group-boundary logic in the same kernel is untouched and still runs normally in this
  // case -- only the node_bodies-based coarse refinement is skipped for this one iteration.
  if (level <= 1) {
    tree.startLevelMin = 0;
    // level_list[0] IS valid (always written), but level_list[0+1] would not be -- overwrite the
    // host copy of level_list[1] with the disabling sentinel so the kernel launch below (which
    // passes &tree.level_list[tree.startLevelMin+1] by reference) sees {0,1}, not stale memory.
    if (tree.level_list.get_size() > 1) {
      tree.level_list[1] = make_uint2(0, 1);
      tree.level_list.h2d();
    }
  } else if (tree.startLevelMin + 1 >= level) {
    tree.startLevelMin = level - 2;
  }


  //Link the tree

  link_tree.set_args(0, &offset,  tree.n_children.p(), tree.node_bodies.p(), tree.bodies_Ppos.p(), &tree.corner,
                        tree.level_list.p(), validList.p(), node_key.p(), tree.bodies_key.p(), &tree.startLevelMin);
  link_tree.setWork(tree.n_nodes , 128);
  link_tree.execute2(execStream->s());
  STAGE_TRACE("[BUG5-STAGE] after link_tree n_nodes=%d\n", tree.n_nodes);

  //After executing link_tree, the id_list contains for each node the ID of its parent.
  //Valid_list contains for each node if its a leaf (valid) or a normal node -> non_valid
  //Execute a split on the validList to get separate id lists
  //for the leafs and nodes. Used when computing multipole expansions

  if(tree.leafNodeIdx.get_size() > 0) tree.leafNodeIdx.cresize_nocpy(tree.n_nodes, false);
  else                                tree.leafNodeIdx.cmalloc      (tree.n_nodes , false);

  //Split the leaf nodes and non-leaf nodes
  gpuSplit(validList, tree.leafNodeIdx, tree.n_nodes, &tree.n_leafs);
  STAGE_TRACE("[BUG5-STAGE] after gpuSplit n_leafs=%d\n", tree.n_leafs);

  LOG("Total nodes: %d N_leafs: %d  non-leafs: %d \n", tree.n_nodes, tree.n_leafs, tree.n_nodes - tree.n_leafs);

  //Build the level list based on the leafIdx list, required for easy
  //access during the compute node properties / multipole computation
  build_level_list.set_args(0, &tree.n_nodes,  &tree.n_leafs, tree.leafNodeIdx.p(), tree.node_bodies.p(), validList.p());
  build_level_list.setWork(tree.n_nodes-tree.n_leafs, 128);
  validList.zeroMemGPUAsync(execStream->s());
  build_level_list.execute2(execStream->s());
  STAGE_TRACE("[BUG5-STAGE] after build_level_list\n");

  // BUG5 FIX: tree.node_level_list is allocated ONCE, early, at a fixed size (MAXLEVELS*2,
  // allocateParticleMemory()/build.cpp) under the assumption that non-leaf nodes are stored
  // level-major, so the number of level-transition marks gpu_build_level_list() produces is
  // roughly 2*(number of distinct levels actually used) <= 2*MAXLEVELS. That assumption does not
  // hold in general: gpu_build_level_list() marks a transition every time two ADJACENT entries in
  // leafNodeIdx (id vs id+1/id-1) differ in level, and nothing guarantees same-level non-leaf
  // nodes are stored contiguously in that array -- under a pathological, very unbalanced tree
  // (e.g. one branch driven to near-MAXLEVELS depth by two nearly-coincident particles in a real
  // close encounter, while the rest of the tree stays shallow), the transition count can be much
  // larger than 2*MAXLEVELS. gpuCompact()'s own compactMove kernel has no bounds check against its
  // output buffer's actual capacity (it only knows the SOURCE length N) -- confirmed as the exact
  // cause of a real, hardware-caught HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (a genuine
  // out-of-bounds device write, not adjacent-buffer corruption) via synchronized per-kernel stage
  // markers bracketing the crash to precisely this call. Fix: resize node_level_list to the true
  // worst case before this call -- gpuCompact can never write more valid entries than its source
  // length N, so N itself (2*(tree.n_nodes-tree.n_leafs)) is a safe, tight upper bound. Only grows
  // (reduce=false), matching the existing tree.leafNodeIdx/tree.group_list resize pattern just
  // above/below in this same function.
  {
    const int neededNodeLevelListSize = std::max(MAXLEVELS*2, 2*(tree.n_nodes - tree.n_leafs));
    if (tree.node_level_list.get_size() < neededNodeLevelListSize)
      tree.node_level_list.cresize_nocpy(neededNodeLevelListSize, false);
  }

  //Compact the node-level boundaries into the node_level_list
  gpuCompact(validList, tree.node_level_list, 2*(tree.n_nodes-tree.n_leafs), 0);
  STAGE_TRACE("[BUG5-STAGE] after node_level_list gpuCompact size=%d needed=%d\n",
          tree.node_level_list.get_size(), 2*(tree.n_nodes-tree.n_leafs));

  /************   Start building the particle groups   *************

      We use the minimum tree-level to set extra boundaries, which ensures
      that groups are based on the tree-structure and will not be very big

      The previous computed offsets are used to build all boundaries.
      The ones based on top level boundaries and the group ones (every NCRIT particles)
  */

  validList.zeroMemGPUAsync(execStream->s());

  // BUG7 investigation (T4, gadget-audit): validList is allocated tree.n*2 (see
  // validList.cmalloc_copy(..., tree.n*2, 0) above) -- that IS the true bound build_group_list2's
  // unguarded write should be checked against.
  bug7_reset_diag(tree.n * 2);

  // BUG7-DIAG4: the write-guard above never fired on a real repro crash -- the fault happens
  // synchronously in the hipDeviceSynchronize() right after this kernel launches (before the
  // host-side diagnostic read below can even run), so it's a HARDWARE fault DURING the kernel, not
  // something my guarded write path was catching. New hypothesis: tree.level_list is cmalloc'd ONCE
  // (build.cpp early) and only entries [0, level-1] are freshly written by THIS iteration's
  // per-level tree-build loop (via build_valid_list, see the line ~292 kernel call) -- entries
  // [level, MAXLEVELS-1] retain STALE data from whatever iteration last built a deeper tree (or
  // uninitialized garbage on the very first build). tree.startLevelMin's own loop (just above) can
  // legitimately set startLevelMin = level-1 (the deepest level exceeding START_LEVEL_MIN_NODES
  // happens to be the last level built this iteration) -- when that happens,
  // tree.level_list[tree.startLevelMin+1] = tree.level_list[level], an entry NOT written this
  // iteration. That stale/garbage uint2 becomes `startLevelBeginEnd` inside build_group_list2,
  // and its .y field bounds the loop `if (idx < startLevelBeginEnd.y-1) { node_bodies[idx]... }` --
  // if .y is stale-huge, this reads node_bodies far out of its real allocated range, a genuine OOB
  // READ (not the write this ticket's earlier guard checked) that would explain a hardware fault
  // the write-guard can't catch. Test directly: this print IS host-side fprintf, which (unlike
  // device printf) is confirmed to flush even when a crash follows shortly after.
  if (gadget_hip_stage_trace())
  fprintf(stderr, "[BUG7-DIAG4] level=%d startLevelMin=%d startLevelMin+1==level? %s "
          "level_list[startLevelMin+1]=(%u,%u) MAXLEVELS_check\n",
          level, tree.startLevelMin, (tree.startLevelMin+1 == level) ? "YES-SUSPECT" : "no",
          tree.level_list[tree.startLevelMin+1].x, tree.level_list[tree.startLevelMin+1].y);

  // T10: rotate group boundaries for this rebuild (no-op unless GADGET_HIP_T10_GROUP_SHIFT=1).
  {
    static int t10_on = -1;
    if (t10_on < 0) { const char *e = getenv("GADGET_HIP_T10_GROUP_SHIFT"); t10_on = e ? atoi(e) : 0; }
    t10_set_group_shift(t10_on);
  }

  define_groups.set_args(0, &tree.n, validList.p(), &tree.level_list[tree.startLevelMin+1], tree.node_bodies.p(),
                            tree.node_level_list.p(), &level);
  define_groups.setWork(tree.n, 128);
  define_groups.execute2(execStream->s());
  STAGE_TRACE("[BUG5-STAGE] after define_groups\n");
  {
    int maxOobWriteIdx = -1, maxOobLastChild = -1, oobCount = 0;
    bug7_read_diag(&maxOobWriteIdx, &maxOobLastChild, &oobCount);
    if (oobCount > 0) {
      fprintf(stderr, "[BUG7-FOUND] define_groups would have written OOB: oobCount=%d "
              "maxOobWriteIdx=%d maxOobLastChild=%d validListLen(bound)=%d n_particles=%d\n",
              oobCount, maxOobWriteIdx, maxOobLastChild, tree.n * 2, tree.n);
    }
  }

  //Copy the node_level_list back to host since we need it to compute the tree properties
  LOG("Finished level list \n");
  tree.node_level_list.d2h();
  for(int i=0; i < level; i++)
  {
    LOG("node_level_list: %d \t%d\n", i, tree.node_level_list[i]);
  }

  //Compact the validList to get the list of group IDs
  gpuCompact(validList, compactList, tree.n*2, &validCount);
  this->resetCompact();
  tree.n_groups = validCount/2;
  LOG("Found number of groups: %d \n", tree.n_groups);

  if(tree.group_list.get_size() > 0) tree.group_list.cresize_nocpy(tree.n_groups, false);
  else                               tree.group_list.cmalloc      (tree.n_groups, false);


  store_groups.set_args(0, &tree.n, &tree.n_groups, compactList.p(), tree.body2group_list.p(), tree.group_list.p());
  store_groups.setWork(-1, NCRIT, tree.n_groups);
  store_groups.execute2(execStream->s());
  STAGE_TRACE("[BUG5-STAGE] after store_groups n_groups=%d\n", tree.n_groups);

  // 2026-08-30: optional compact-group override -- root-cause fix for the group-corner MAC
  // transverse-force bug (PLAN.md Close-encounter root-cause audit / LOG.md Sec 48-50, re-confirmed
  // same day via real Gadget-2-vs-port Zel'dovich y/z-velocity comparison, ~87x excess, and a
  // population-wide corner-score/force-leakage correlation of 0.40 measured directly on this
  // kernel's own real data). The blind idx%NCRIT chunking above has no compactness/centering
  // control and can produce elongated groups (aspect ratio down to ~0.09-0.4, measured). This
  // instead snaps group boundaries onto the ALREADY-BUILT tree's own LEAVES -- merging adjacent
  // small leaves up to ~NCRIT and only falling back to blind sub-chunking for the rare leaf still
  // oversized after that (shouldn't happen in practice, since NLEAF<=NCRIT is already enforced
  // elsewhere, but kept as a defensive fallback).
  // Deliberately does NOT reorder any particle data -- physically permuting bodies_pos/bodies_key
  // would break cl_build_valid_list's Morton-ADJACENT-key assumption (build_tree.cu), silently
  // producing a wrong-topology tree; this only chooses different, already-valid cut points into
  // the array the tree build already produced, so the tree itself is untouched. Env-var gated
  // (single-GPU only, matching this port's existing PM/no-MPI-slab-decomposition simplification)
  // for a safe, easily-reverted A/B against the original blind chunking above.
  //
  // BUG (found live, 2026-08-30): an earlier version of this block used a single tree LEVEL's
  // node_bodies listing as the atom set, assuming it fully partitions [0,n). It doesn't: once a
  // node becomes a leaf (a sparse region), its subtree stops descending and it never appears again
  // at deeper levels -- leaving real gaps in that level's own coverage. The old code's "chain
  // boundaries contiguously" safety net silently papered over those gaps by force-extending an
  // adjacent group across them, lumping spatially-unrelated particles into the wrong group --
  // confirmed by a live run whose energy blew up (~105% relative error, growing every iteration)
  // instead of merely under-improving. Fixed by using the tree's actual LEAVES (tree.leafNodeIdx)
  // as the atom set instead of one arbitrary level -- leaves are guaranteed by the tree build's own
  // correctness to exactly and completely tile [0,n) with no gaps or overlaps, whatever level each
  // one happens to terminate at.
  if (getenv("GADGET_HIP_COMPACT_GROUPS") && this->mpiGetNProcs() == 1 && tree.n > 0)
  {
    tree.node_bodies.d2h();
    tree.leafNodeIdx.d2h();

    std::vector<std::pair<uint,uint>> atoms;
    atoms.reserve(tree.n_leafs);
    for (int i = 0; i < tree.n_leafs; i++)
    {
      uint2 nb = tree.node_bodies[tree.leafNodeIdx[i]];
      uint start = nb.x & ILEVELMASK;
      uint end   = nb.y;
      if (end > start) atoms.push_back(std::make_pair(start, end));
    }
    std::sort(atoms.begin(), atoms.end());

    std::vector<std::pair<uint,uint>> newGroups;
    uint curStart = 0, curCount = 0; bool haveCur = false;
    for (size_t ai = 0; ai < atoms.size(); ai++)
    {
      uint aStart = atoms[ai].first, aEnd = atoms[ai].second;
      uint aCount = aEnd - aStart;
      if (aCount > (uint)(2 * NCRIT))
      {
        if (haveCur) { newGroups.push_back(std::make_pair(curStart, aStart)); haveCur = false; curCount = 0; }
        for (uint s = aStart; s < aEnd; s += (uint)NCRIT)
          newGroups.push_back(std::make_pair(s, std::min(s + (uint)NCRIT, aEnd)));
        continue;
      }
      // BUG (found while measuring C-A-13): this used to accumulate first and close the group
      // only AFTER curCount reached NCRIT, so a group could end up with up to NCRIT + NLEAF - 1
      // particles. The walk processes at most NCRIT particles per group, so every particle past
      // that was silently never given a force. Measured at 500k: average group size 35.1 > NCRIT
      // (14244 groups for 500000 particles), and 360832 of 500000 particles came out with
      // EXACTLY zero acceleration, Etot positive (+0.112 against the correct -0.25).
      // Close the current group BEFORE adding an atom that would overflow it.
      if (haveCur && curCount + aCount > (uint)NCRIT)
      {
        newGroups.push_back(std::make_pair(curStart, aStart));
        haveCur = false; curCount = 0;
      }
      if (!haveCur) { curStart = aStart; curCount = 0; haveCur = true; }
      curCount += aCount;
      if (curCount >= (uint)NCRIT)
      {
        newGroups.push_back(std::make_pair(curStart, aEnd));
        haveCur = false; curCount = 0;
      }
    }
    if (haveCur) newGroups.push_back(std::make_pair(curStart, (uint)tree.n));

    if (!newGroups.empty())
    {
      newGroups.front().first = 0;
      newGroups.back().second = (uint)tree.n;
      for (size_t i = 1; i < newGroups.size(); i++)
        newGroups[i].first = newGroups[i-1].second;
    }

    // The chaining above closes any gap by extending a group's start backwards, which can push it
    // over NCRIT again. Check rather than assume -- an oversized group is silently unforced.
    {
      uint worst = 0;
      for (size_t i = 0; i < newGroups.size(); i++)
        worst = std::max(worst, newGroups[i].second - newGroups[i].first);
      if (worst > (uint)NCRIT)
        fprintf(stderr, "[COMPACT-GROUPS] FATAL: largest group is %u particles > NCRIT=%d; the "
                        "walk would leave the excess unforced.\n", worst, NCRIT), exit(1);
    }

    int newNGroups = (int)newGroups.size();
    if (tree.group_list.get_size() > 0) tree.group_list.cresize_nocpy(newNGroups, false);
    else                                tree.group_list.cmalloc      (newNGroups, false);

    for (int g = 0; g < newNGroups; g++)
      tree.group_list[g] = make_uint2(newGroups[g].first, newGroups[g].second);
    tree.group_list.h2d();

    for (int g = 0; g < newNGroups; g++)
      for (uint i = newGroups[g].first; i < newGroups[g].second; i++)
        tree.body2group_list[i] = g;
    tree.body2group_list.h2d();

    tree.n_groups = newNGroups;
    fprintf(stderr, "[COMPACT-GROUPS] n_leafs=%d n_groups=%d (blind chunking gave %d)\n",
            tree.n_leafs, newNGroups, validCount/2);
  }

  //Memory allocation for the valid group lists
  //TODO get rid of this if by calling cresize when cmalloc is already called from inside the cmalloc call
  if(tree.active_group_list.get_size() > 0)
  {
    tree.active_group_list.cresize_nocpy(tree.n_groups, false);
    tree.activeGrpList.cresize_nocpy(tree.n_groups, false);
  }
  else
  {
    tree.active_group_list.cmalloc(tree.n_groups, false);
    tree.activeGrpList.cmalloc(tree.n_groups, false);
  }

  LOG("Tree built complete!\n");
  devContext->stopTiming("Tree-construction", 2, execStream->s());

  /*************************/
}


//This function builds a hash-table for the particle-keys which is required for the
//domain distribution based on the SFC
void octree::parallelDataSummary(tree_structure &tree,
                                 float lastExecTime, float lastExecTime2,
                                 double &domComp, double &domExch,
                                 bool initialSetup) {
  double t0 = get_time();

  bool updateBoundaries = false;

  //Update if the maximum duration is 10% larger than average duration
  //and always update the first couple of iterations to create load-balance
  if(iter < 32 || (  100*((maxExecTimePrevStep-avgExecTimePrevStep) / avgExecTimePrevStep) > 10 ))
  {
    updateBoundaries = true;
  }

  //updateBoundaries = true; //TEST, keep always update for now


  real4 r_min = {+1e10, +1e10, +1e10, +1e10};
  real4 r_max = {-1e10, -1e10, -1e10, -1e10};
  this->getBoundaries(tree, r_min, r_max); //Used for predicted position keys further down

  build_key_list.set_args(0, tree.bodies_key.p(), tree.bodies_pos.p(), &tree.n, &tree.corner);
  if(updateBoundaries)
  {
    //Build keys on current positions, since those are already sorted, while predicted are not
    build_key_list.set_args(0, tree.bodies_key.p(), tree.bodies_pos.p(), &tree.n, &tree.corner);
    build_key_list.setWork(tree.n, 128);
    build_key_list.execute2(execStream->s());

    /* added by evghenii, needed for 2D domain decomposition in parallel.cpp */
    tree.bodies_key.d2h(true,execStream->s());
  }

   //Get the global boundaries and compute the corner / size of tree
   this->sendCurrentRadiusInfo(r_min, r_max);
   real size     = 1.001f*std::max(r_max.z - r_min.z,
                          std::max(r_max.y - r_min.y, r_max.x - r_min.x));

   tree.corner   = make_real4(0.5f*(r_min.x + r_max.x) - 0.5f*size,
                              0.5f*(r_min.y + r_max.y) - 0.5f*size,
                              0.5f*(r_min.z + r_max.z) - 0.5f*size,
                              size/(1 << MAXLEVELS));

   if(updateBoundaries)
     execStream->sync(); //This one has to be finished when we start updating the domain
                         //as it contains the keys on which we sample to update boundaries

   //Compute keys again, needed for the redistribution
   //Note we can call this in parallel with the computation of the domain.
   //This is done on predicted positions, to make sure that particles AFTER
   //prediction are separated by boundaries
   build_key_list.reset_arg(1,   tree.bodies_Ppos.p());
   build_key_list.execute2(execStream->s());

   if(updateBoundaries)
   {
     exchangeSamplesAndUpdateBoundarySFC(NULL, 0, NULL,
                                         NULL,  NULL, 0,
                                         &tree.parallelBoundaries[0], lastExecTime,
                                         initialSetup);
   }


    domComp = get_time()-t0;
    char buff5[1024];
    sprintf(buff5,"EXCHANGEA-%d: tUpdateBoundaries: %lg\n", procId,  domComp);
    devContext->writeLogEvent(buff5);

    //Boundaries computed, now exchange the particles
    LOGF(stderr, "Computing, exchanging and recompute of domain boundaries took: %f \n",domComp);
    t0 = get_time();
    gpuRedistributeParticles_SFC(&tree.parallelBoundaries[0]); //Redistribute the particles
    domExch = get_time()-t0;

    LOGF(stderr, "Redistribute domain took: %f\n", get_time()-t0);

  /*************************/

}


