#pragma once
//===========================================================================================
// Device-buffer index-space registry.   PLAN section 4.1, deliverable 2.
//
// Bonsai keeps particle buffers in TWO index spaces at once, and which space a buffer is in
// depends on where you are in the iteration. The rule has never been written down as data --
// only as prose in scattered comments, some of which were wrong -- and getting it wrong has
// produced four confirmed bugs:
//
//   T12       correct_particles left pos/vel untouched for inactive particles, so a raw slot's
//             owning particle churned out from under its own stale float data.
//   T23 Bug 2 computeGroupMaxAccel read bodies_acc0 with the current sorted index; every
//             group's MAC input silently used a different set of particles' accelerations.
//   T27       bodies_forceSoftening -- this one was a MISDIAGNOSIS. T27 added an
//             oriParticleOrder translation for it by analogy with bodies_acc0; the buffer is
//             actually written after the sort and is CURRENT-order, so the translation reads
//             the wrong particle. Inert only because every IC here is single-species.
//   C-D-03b   setActiveGroups paired body2grouplist[idx] (rebuilt AFTER the sort) with
//             bodies_time[idx] (never resorted). Due particles were never activated, t_current
//             was pinned, and 99.6% of all iterations did nothing. Not in the 90-contract sweep.
//
// THE INVARIANT, verified against runtime/src/sort_bodies_gpu.cpp (not from comments):
//
//   Phase 1  predict_particles, before sort_bodies. oriParticleOrder is identity (correct_particles
//            reset it at the end of the previous step), so every buffer may be indexed directly.
//
//   Phase 2  after sort_bodies, through approximate_gravity, up to correct_particles. This is
//            where the two spaces diverge and where every one of the four bugs above lived.
//              SPACE_CURRENT  -- permuted by sort_bodies (or produced after it): index directly.
//              SPACE_ORIGINAL -- NOT permuted: index as buf[oriParticleOrder[idx]].
//
//   Phase 3  correct_particles rewrites pos/vel/acc0/time at the NEW index for every particle and
//            sets oriParticleOrder[idx] = idx, which re-unifies the spaces for the next Phase 1.
//
// sort_bodies' default per-step path (!doFullShuffle) permutes exactly four buffers:
// bodies_key (:184), bodies_Ppos, bodies_ids, bodies_h (:221-223). The doFullShuffle path
// permutes nine, but is not taken by the normal per-iteration call.
//===========================================================================================

// CLASSIFICATION RULE -- corrected 2026-09-06, after it produced two wrong entries.
//
// WRONG rule (what the first version of this registry used):
//     "not permuted by sort_bodies  =>  SPACE_ORIGINAL"
// That is a non-sequitur. A buffer can also be WRITTEN after the sort, in the current order,
// without ever being permuted -- bodies_acc1, bodies_dens, bodies_typeDevice and
// bodies_forceSoftening are all like this. The naive rule classified every one of them as
// ORIGINAL; three were wrong, and one of those (bodies_forceSoftening) had already caused a real
// latent bug in T27's fix.
//
// CORRECT rule: find the LAST WRITE before the phase-2 read, and ask what index it used.
//     written before sort_bodies, not permuted            -> SPACE_ORIGINAL
//     permuted by sort_bodies                             -> SPACE_CURRENT
//     written after sort_bodies at the current index      -> SPACE_CURRENT
//
// Every entry below now cites its writer, not merely its absence from the reorder list.

enum BufferSpace
{
  SPACE_CURRENT,    // Phase 2: index directly with the current sorted index.
  SPACE_ORIGINAL,   // Phase 2: MUST go through oriParticleOrder[idx].
  SPACE_NODE,       // Indexed by tree node id; rebuilt after the sort, so no translation.
  SPACE_GROUP,      // Indexed by group id; likewise rebuilt after the sort.
  SPACE_GLOBAL      // Not particle-indexed at all (tables, scratch, reductions).
};

enum BufferEvidence
{
  EV_VERIFIED,      // Read off sort_bodies_gpu.cpp / the kernel that writes it.
  EV_INFERRED       // Follows from the same rule but no consumer was audited. Treat with care.
};

struct BufferSpec
{
  const char   *name;
  BufferSpace   space;
  BufferEvidence evidence;
  const char   *note;
  // Fraction of this buffer's bytes that exist ONLY to carry quadrupole data and are dead in a
  // monopole-only build (which is the default, and is also Gadget-2's own behaviour -- it is
  // monopole-only by design). Reported by reportDeviceMemory() as recoverable capacity, so the
  // saving from removing quadrupole support is a measured number rather than a guess.
  float         quadrupoleDeadFraction;
};

// Order follows octree.h's declaration order.
static const BufferSpec kBufferRegistry[] =
{
  // --- particle buffers permuted by sort_bodies' default path -------------------------------
  {"bodies_key",            SPACE_CURRENT,  EV_VERIFIED, "reordered sort_bodies_gpu.cpp:184"},
  {"bodies_Ppos",           SPACE_CURRENT,  EV_VERIFIED, "reordered :221; drifted every step for ALL particles"},
  {"bodies_ids",            SPACE_CURRENT,  EV_VERIFIED, "reordered :222 -- this is why raw slots churn"},
  {"bodies_h",              SPACE_CURRENT,  EV_VERIFIED, "reordered :223. NOW WRITE-ONLY for physics: with BONSAI_DENSITY=0 its only readers are gone (adjustH and the walk density). Still written by compute_leaf and reordered by dataReorder, and still exchanged by the nProcs>1 paths in parallel.cpp -- which is why it is still full size. 512 MB at 512^3, the next cut"},

  // --- particle buffers NOT permuted: translate in Phase 2 ----------------------------------
  {"bodies_pos",            SPACE_ORIGINAL, EV_VERIFIED, "T12; rewritten at NEW idx by correct_particles"},
  {"bodies_vel",            SPACE_ORIGINAL, EV_VERIFIED, "T12; rewritten at NEW idx by correct_particles"},
  {"bodies_acc0",           SPACE_ORIGINAL, EV_VERIFIED, "T23 Bug 2 -- computeGroupMaxAccel must translate"},
  {"bodies_Pvel",           SPACE_ORIGINAL, EV_VERIFIED, "correct_particles reads pVel[unsortedIdx]"},
  {"bodies_time",           SPACE_ORIGINAL, EV_VERIFIED, "C-D-03b -- setActiveGroups must translate"},
  {"bodies_forceSoftening", SPACE_CURRENT,  EV_VERIFIED, "written AFTER the sort by recomputeSoftening (gpu_iterate.cpp:592); bodies_type is rebuilt from the resorted bodies_ids at :2120, so CURRENT order. T27 said ORIGINAL and was wrong; I copied that claim into this registry without checking it."},
  {"bodies_dens",           SPACE_CURRENT,  EV_VERIFIED, "DISABLED (BONSAI_DENSITY=0 in node_specs.h): allocated at 1 element. When on, written by the walk at `addr`, the same index it writes acc_out/active_inout with. Carries no gravity: softening comes from bodies_forceSoftening, not bodies_h"},
  {"bodies_typeDevice",     SPACE_CURRENT,  EV_VERIFIED, "written by recomputeSoftening AFTER the sort, in lockstep with bodies_forceSoftening (gpu_iterate.cpp:2161)"},

  // --- produced after the sort, so already in the current space -----------------------------
  {"bodies_acc1",           SPACE_CURRENT,  EV_VERIFIED, "written by approximate_gravity at the new idx"},
  {"oriParticleOrder",      SPACE_CURRENT,  EV_VERIFIED, "IS the map: current idx -> original idx"},

  // --- node / group indexed ------------------------------------------------------------------
  {"level_list",            SPACE_NODE,   EV_INFERRED, ""},
  {"n_children",            SPACE_NODE,   EV_INFERRED, ""},
  {"node_bodies",           SPACE_NODE,   EV_VERIFIED, "carries the build-time level bits (T26)"},
  {"leafNodeIdx",           SPACE_NODE,   EV_INFERRED, ""},
  {"node_level_list",       SPACE_NODE,   EV_VERIFIED, "Bug 5 overflowed this under unbalanced trees"},
  {"multipole",             SPACE_NODE,   EV_VERIFIED, "3 real4 per node: M0 (mass+COM), Q0, Q1. The monopole path reads only M0 (dev_approximate_gravity_warp_new.cu:1030,:864); the quadrupole path reads Q0/Q1 at :809-810 but zeroes Q0.w. So 2 of 3 real4 per node are dead in the default build", 2.0f/3.0f},
  {"boxSizeInfo",           SPACE_NODE,   EV_VERIFIED, "AABB half-extents (C-B-06/C-B-10b)"},
  {"boxCenterInfo",         SPACE_NODE,   EV_VERIFIED, "geometric centre + cellOp (C-B-05)"},
  {"cellSizeInfo",          SPACE_NODE,   EV_VERIFIED, "fixed octree cell side (T26/T9)"},
  {"body2group_list",       SPACE_CURRENT, EV_VERIFIED, "particle -> group, rebuilt AFTER the sort: C-D-03b"},
  {"group_list",            SPACE_GROUP,  EV_VERIFIED, "start/end particle per group"},
  {"groupSizeInfo",         SPACE_GROUP,  EV_VERIFIED, ""},
  {"groupCenterInfo",       SPACE_GROUP,  EV_VERIFIED, ""},
  {"groupMaxAccInfo",       SPACE_GROUP,  EV_VERIFIED, "C-B-01/C-B-02 input"},
  {"groupMaxSofteningInfo", SPACE_GROUP,  EV_VERIFIED, "T27"},
  {"activeGrpList",         SPACE_GROUP,  EV_VERIFIED, "written by setActiveGroups"},
  {"active_group_list",     SPACE_GROUP,  EV_INFERRED, ""},
  {"activePartlist",        SPACE_CURRENT, EV_VERIFIED, "set by the walk at the current idx"},

  // --- not particle-indexed -------------------------------------------------------------------
  {"ngb",                   SPACE_GLOBAL, EV_VERIFIED, "DEAD: allocated at 1 element. The walk's only store wrote each particle's own index back (`ngb_out[addr] = addr`, tagged \"JB Fixed this for demo\"); no reader exists. Store removed"},
  {"interactions",          SPACE_GLOBAL, EV_VERIFIED, "DISABLED (INTCOUNT=false in dev_approximate_gravity_warp_new.cu): allocated at 1 element. Diagnostic interaction counters feeding one log line. NOTE the stores used to sit OUTSIDE the INTCOUNT guard, so turning counting off still wrote zeros across the whole array"},
  {"parallelBoundaries",    SPACE_GLOBAL, EV_INFERRED, "MPI domain split; dead at nProcs==1"},
  {"generalBuffer1",        SPACE_GLOBAL, EV_VERIFIED, "shared scratch; pinned to treeWalkStackSize floor"},
  {"fullRemoteTree",        SPACE_GLOBAL, EV_INFERRED, "LET; dead at nProcs==1"},
  {"gadgetDriftTable",      SPACE_GLOBAL, EV_VERIFIED, "drift-factor table"},
  {"gadgetGravKickTable",   SPACE_GLOBAL, EV_VERIFIED, "grav-kick-factor table"},
  {"devMemRMIN",            SPACE_GLOBAL, EV_VERIFIED, "bounding-box reduction"},
  {"devMemRMAX",            SPACE_GLOBAL, EV_VERIFIED, "bounding-box reduction"},
  {"devMemCounts",          SPACE_GLOBAL, EV_INFERRED, ""},
  {"devMemCountsx",         SPACE_GLOBAL, EV_INFERRED, ""},
};

static const int kBufferRegistryCount =
  (int)(sizeof(kBufferRegistry) / sizeof(kBufferRegistry[0]));

inline const char *bufferSpaceName(BufferSpace s)
{
  switch (s)
  {
    case SPACE_CURRENT:  return "CURRENT";
    case SPACE_ORIGINAL: return "ORIGINAL";
    case SPACE_NODE:     return "NODE";
    case SPACE_GROUP:    return "GROUP";
    default:             return "GLOBAL";
  }
}

inline const BufferSpec *bufferSpecFor(const char *name)
{
  for (int i = 0; i < kBufferRegistryCount; i++)
    if (__builtin_strcmp(kBufferRegistry[i].name, name) == 0) return &kBufferRegistry[i];
  return 0;
}
