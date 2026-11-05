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
// Since T81 item B, sort_bodies permutes EVERY per-particle buffer every step (bodies_key,
// bodies_Ppos, bodies_vel, bodies_ids, bodies_acc0, bodies_time, and bodies_h under
// BONSAI_DENSITY), so there is exactly ONE particle index space in every phase: the current
// sorted order. There is no "original order" and no permutation map -- the four bugs above were
// all a consequence of a buffer surviving the sort unpermuted, and that class is now unwriteable.
//
// Buffers produced AFTER the sort (bodies_acc1, bodies_forceSoftening, bodies_typeDevice,
// body2group_list, ...) are also in the current order by construction. A particle buffer is
// therefore always indexable with CurrentIdx; IdSpace remains only for GravPM, which is indexed by
// particle id so it survives the re-sort between PM steps.
//===========================================================================================

// CLASSIFICATION RULE -- corrected 2026-09-06, after it produced two wrong entries, and superseded
// by T81 item B, which permutes every particle buffer every step and thereby removed the ORIGINAL
// space entirely. A buffer can still be WRITTEN after the sort in the current order (bodies_acc1,
// bodies_forceSoftening, bodies_typeDevice) without ever being permuted; those are SPACE_CURRENT
// because they are produced at the current index, not because they were permuted.
//
// Every entry below cites its writer, not merely its presence/absence in the reorder list.

enum BufferSpace
{
  SPACE_CURRENT,    // Particle-indexed: sort permutes it, or it is written after the sort.
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
  // --- particle buffers permuted by sort_bodies (T81 B: ALL of them) --------------------------
  {"bodies_key",            SPACE_CURRENT,  EV_VERIFIED, "reordered sort_bodies_gpu.cpp"},
  {"bodies_Ppos",           SPACE_CURRENT,  EV_VERIFIED, "reordered; drifted every step for ALL particles"},
  {"bodies_ids",            SPACE_CURRENT,  EV_VERIFIED, "reordered -- this is why raw slots churn"},
  {"bodies_vel",            SPACE_CURRENT,  EV_VERIFIED, "reordered (T81 A); correct_particles updates it in place"},
  {"bodies_acc0",           SPACE_CURRENT,  EV_VERIFIED, "reordered (T81 B); the T23 Bug 2 class is now unwriteable"},
  {"bodies_time",           SPACE_CURRENT,  EV_VERIFIED, "reordered (T81 B); the C-D-03b class is now unwriteable"},
  {"bodies_h",              SPACE_CURRENT,  EV_VERIFIED, "reordered. NOW WRITE-ONLY for physics: with BONSAI_DENSITY=0 its only readers are gone (adjustH and the walk density). Still written by compute_leaf and reordered by dataReorder, and still exchanged by the nProcs>1 paths in parallel.cpp -- which is why it is still full size. 512 MB at 512^3, the next cut"},
  {"bodies_pos",            SPACE_CURRENT,  EV_VERIFIED, "T81 B; bound to bodies_Ppos (C-A-22 fold), so already permuted"},

  // --- produced after the sort, so already in the current space -----------------------------
  {"bodies_forceSoftening", SPACE_CURRENT,  EV_VERIFIED, "written AFTER the sort by recomputeSoftening (gpu_iterate.cpp); bodies_type is rebuilt from the resorted bodies_ids, so CURRENT order. T27 said ORIGINAL and was wrong; I copied that claim into this registry without checking it."},
  {"bodies_dens",           SPACE_CURRENT,  EV_VERIFIED, "DISABLED (BONSAI_DENSITY=0 in node_specs.h): allocated at 1 element. When on, written by the walk at `addr`, the same index it writes acc_out/active_inout with. Carries no gravity: softening comes from bodies_forceSoftening, not bodies_h"},
  {"bodies_typeDevice",     SPACE_CURRENT,  EV_VERIFIED, "written by recomputeSoftening AFTER the sort, in lockstep with bodies_forceSoftening (gpu_iterate.cpp)"},

  // --- produced after the sort, so already in the current space -----------------------------
  {"bodies_acc1",           SPACE_CURRENT,  EV_VERIFIED, "written by approximate_gravity at the new idx"},

  // --- node / group indexed ------------------------------------------------------------------
  {"level_list",            SPACE_NODE,   EV_INFERRED, ""},
  {"n_children",            SPACE_NODE,   EV_INFERRED, ""},
  {"node_bodies",           SPACE_NODE,   EV_VERIFIED, "uint4: .x first body, .y one past last, .z level (T74)"},
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
