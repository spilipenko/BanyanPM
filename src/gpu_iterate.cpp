#include "hip/hip_runtime.h"
#undef NDEBUG
#include "octree.h"
#include "gadget_units.h"          // GFreeAcc / GCarrier (T44 item 1)
#include "gadget_index_spaces.h"   // CurrentOrder / OriginalOrder / IdSpace (T44 item 1)
#include "buffer_registry.h"
#include  "postProcessModules.h"
#include "pm.h"
// Phase 5 ticket 08 (PLAN.md): unconditional, unlike pm_zoom.h below -- writeGadgetSnapshot()
// needs GadgetSnapshotHeader/GadgetParticleData in every --param-capable build, not just
// GADGET_HIP_HIGHRES ones.
#include "gadget_snapshot.h"
#ifdef GADGET_HIP_HIGHRES
#include "pm_zoom.h"
#endif

#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>
#include <tuple>
#include <cstring>

bool gadget_hip_debug_log = false;
// T40: GPU time of the previous step's tree walk, in ms (see hipEventElapsedTime below).
static float g_t40LastWalkMs = 0.0f;   // set from main.cpp (--debug / GADGET_HIP_DEBUG)
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unistd.h>   // access(F_OK), for the OutputDir/stop file check

using namespace std;

// Transverse-force bug trace (user request 2026-08-30): defined in
// dev_approximate_gravity_warp_new.cu, alongside the g_trace* __device__ globals they wrap.
void traceY_reset();
void traceY_read(double &approxY, double &directY, int &approxN, int &directN);
void traceY_dump(const char *path);

static double de_max  = 0;
static double dde_max = 0;

// T16 (2026-09-02, ticket T16): persisted initial/previous-iteration TOTAL energy for the
// PRODUCTION de/d(de) computation, tracked separately from the existing raw Etot0/Etot1 class
// members (octree.h) because those are read directly, still raw, by GADGET_HIP_COMOVING_DE_CHECK's
// own "de_raw" diagnostic field below -- this promotion must not disturb that. See the comment
// at the production de/d(de) computation (compute_energies()) for the full rationale.
static double Etot0_prod = 0.0;
static double Etot1_prod = 0.0;

// Duplicate-position collision tracer (user request 2026-08-30): scans a position buffer for
// bit-exact duplicate (x,y,z) values among distinct particles and logs the underlying, stable
// particle IDs (tree.bodies_ids) plus the pipeline stage where the collision was observed --
// localizes exactly which kernel first produces a collision, rather than inferring it after the
// fact from a crash-adjacent full-state dump (which only shows the end result, not when/where it
// first appeared). Called after predict() (checking bodies_Ppos, this step's predicted positions,
// computed for every particle whether active or not) and after correct() (checking bodies_pos,
// the real committed positions) -- comparing the two pinpoints whether predict_particles() itself
// produces the collision from still-distinct inputs, or whether it was already present in the
// velocity/acceleration state predict_particles merely propagates forward.
static void traceDuplicatePositions(tree_structure &tree, my_dev::dev_mem<real4> &posBuf,
                                     const char *stage, int iter)
{
  posBuf.d2h();
  tree.bodies_ids.d2h();

  struct Key { uint32_t x, y, z; bool operator==(const Key &o) const { return x == o.x && y == o.y && z == o.z; } };
  struct KeyHash { size_t operator()(const Key &k) const {
      return (size_t)k.x ^ ((size_t)k.y << 21) ^ ((size_t)k.z << 42); } };

  std::unordered_map<Key, std::vector<int>, KeyHash> groups;
  groups.reserve(tree.n);
  for (int i = 0; i < tree.n; i++)
  {
    real4 p = posBuf[i];
    uint32_t xb, yb, zb;
    memcpy(&xb, &p.x, sizeof(xb));
    memcpy(&yb, &p.y, sizeof(yb));
    memcpy(&zb, &p.z, sizeof(zb));
    groups[Key{xb, yb, zb}].push_back(i);
  }

  int nDupGroups = 0, nDupParticles = 0;
  for (auto &kv : groups)
    if (kv.second.size() > 1) { nDupGroups++; nDupParticles += (int)kv.second.size(); }

  if (nDupGroups == 0) return;  // keep the log file small -- only write when something is found

  FILE *flog = fopen("/tmp/gadgethip_duptrace.log", "a");
  if (!flog) return;
  for (auto &kv : groups)
  {
    if (kv.second.size() < 2) continue;
    fprintf(flog, "iter=%d stage=%s DUP size=%zu ids=[", iter, stage, kv.second.size());
    for (size_t k = 0; k < kv.second.size(); k++)
      fprintf(flog, "%s%llu", k ? "," : "", (unsigned long long) tree.bodies_ids[kv.second[k]]);
    fprintf(flog, "] idxs=[");
    for (size_t k = 0; k < kv.second.size(); k++)
      fprintf(flog, "%s%d", k ? "," : "", kv.second[k]);
    fprintf(flog, "] pos=(%.9g,%.9g,%.9g)\n", posBuf[kv.second[0]].x, posBuf[kv.second[0]].y, posBuf[kv.second[0]].z);
  }
  fprintf(flog, "iter=%d stage=%s SUMMARY dupGroups=%d dupParticles=%d totalN=%d\n",
          iter, stage, nDupGroups, nDupParticles, tree.n);
  fclose(flog);
}

// Ticket T11 (MAP.md, 2026-09-01): Bug B root-cause direct trajectory diff. GADGET_HIP_TRACE_DUP
// (above) tells us *that* and *when* particles collapse to bit-exact positions, but not *why* --
// it never looked at velocity, force magnitude, or the timestep the port actually chose for these
// particles in the run-up to the collapse. This extends the same idiom (env-var gated, zero-cost
// when unset, called from the same two pipeline points as traceDuplicatePositions) into a full
// per-particle trajectory dump for a fixed, caller-specified set of particle IDs -- one line per
// tracked particle per call site per iteration -- so the exact iteration/quantity where the port's
// trajectory first departs from stock Gadget-2's own trajectory for the same IDs (dumped via the
// matching addition to accel.c/forcetree.c on the reference build, see PLAN.md) can be found by a
// straight external diff, rather than inferred after the fact from the collapse's end state.
// GADGET_HIP_TRAJ_IDS: comma-separated particle IDs to track (parsed once, cached in a static set).
// GADGET_HIP_TRAJ_DUMP: output path (default /tmp/gadgethip_trajtrace.log if unset but IDs given).
// C-D-01 probe: dump the values that actually decide the drift for a traced particle, so the
// applied displacement can be checked against BOTH candidate anchors arithmetically rather than
// argued about. In a non-comoving run compute_dt's drift is exactly `dt_drift = tc - t0` (no drift
// table involved), so for each step:
//     pos_after - pos_before  ==  vel * (tc - tb)   -> legacy per-particle Ti_begstep anchor
//     pos_after - pos_before  ==  vel * (tc - tp)   -> global previous-sync anchor (Gadget-2)
// Called after predict() (pos = pre-drift, Ppos = predicted) and after correct() (pos = committed).
// Zero cost unless GADGET_HIP_CD01_TRACE=<id[,id...]> is set.
// C-D-01/03 population probe: why does the clock not advance? Each iteration report how many
// particles are genuinely DUE (time[].y == t_current), how many the group-activation path actually
// marked ACTIVE, and the global minimum Ti_endstep (which is what t_current is reduced from).
// A due particle that is never marked active can never be re-binned, so the minimum -- and hence
// simulation time -- is pinned forever. Gated by GADGET_HIP_CD01_STATS=1.
static void cd01_stats(tree_structure &tree, const char *stage, int iter, double tp, double tc)
{
  static int on = -1;
  if (on < 0) { const char *e = getenv("GADGET_HIP_CD01_STATS"); on = e ? atoi(e) : 0; }
  if (!on) return;

  tree.bodies_time.d2h();
  tree.activePartlist.d2h();
  tree.oriParticleOrder.d2h();

  // activePartlist is indexed in the CURRENT sorted space; bodies_time is not resorted, so it must
  // be reached through oriParticleOrder -- the same translation C-D-03b fixed in setActiveGroups.
  int nDue = 0, nActive = 0, nDueButInactive = 0, nActiveNotDue = 0;
  float minTe = 3.4e38f;
  for (int j = 0; j < tree.n; j++)
  {
    const uint  ti  = tree.oriParticleOrder[j];
    const float te  = tree.bodies_time[ti].y;
    const bool  act = (tree.activePartlist[j] == 1);
    const bool  due = (te == (float) tc);
    if (te < minTe) minTe = te;
    if (due)  { nDue++;    if (!act) nDueButInactive++; }
    if (act)  { nActive++; if (!due) nActiveNotDue++;   }
  }
  fprintf(stderr, "[CD01-STATS] %-4s iter=%d tp=%.9g tc=%.9g nDue=%d nActive=%d "
                  "nDueButInactive=%d nActiveNotDue=%d promotion=%.2f minTe=%.9g\n",
          stage, iter, tp, tc, nDue, nActive, nDueButInactive, nActiveNotDue,
          nDue > 0 ? (double) nActive / nDue : -1.0, minTe);
}

static void cd01_trace(tree_structure &tree, const char *stage, int iter, double tp, double tc)
{
  static bool init = false;
  static std::vector<unsigned long long> ids;
  if (!init)
  {
    init = true;
    const char *e = getenv("GADGET_HIP_CD01_TRACE");
    if (e)
    {
      std::string str(e);
      size_t p = 0;
      while (p <= str.size())
      {
        size_t q = str.find(',', p);
        if (q == std::string::npos) q = str.size();
        if (q > p) ids.push_back(strtoull(str.substr(p, q - p).c_str(), nullptr, 10));
        p = q + 1;
      }
    }
  }
  if (ids.empty()) return;

  tree.bodies_ids.d2h();
  tree.bodies_Ppos.d2h();
  tree.bodies_vel.d2h();
  tree.bodies_time.d2h();
  tree.bodies_Ppos.d2h();

  for (size_t k = 0; k < ids.size(); k++)
    for (int i = 0; i < tree.n; i++)
      if ((unsigned long long) tree.bodies_ids[i] == ids[k])
      {
        const real4  p  = tree.bodies_Ppos[i];
        const real4  pp = tree.bodies_Ppos[i];
        const real4  v  = tree.bodies_vel[i];
        const int2   t  = tree.bodies_time[i];   // Phase 2: (Ti_beg, Ti_end) in ticks
        fprintf(stderr, "[CD01] %-5s iter=%d id=%llu idx=%d tb=%d te=%d tp=%.9g tc=%.9g "
                        "pos=%.9g,%.9g,%.9g Ppos=%.9g,%.9g,%.9g vel=%.9g,%.9g,%.9g\n",
                stage, iter, ids[k], i, t.x, t.y, tp, tc,
                p.x, p.y, p.z, pp.x, pp.y, pp.z, v.x, v.y, v.z);
        break;
      }
}

static void traceParticleTrajectory(tree_structure &tree, const char *stage, int iter, double simTime)
{
  static bool initialized = false;
  static std::vector<ullong> targetIds;
  static const char *dumpPath = nullptr;
  if (!initialized)
  {
    initialized = true;
    const char *idsEnv = getenv("GADGET_HIP_TRAJ_IDS");
    if (idsEnv)
    {
      std::string s(idsEnv);
      size_t pos = 0;
      while (pos < s.size())
      {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        if (comma > pos) targetIds.push_back((ullong) strtoull(s.c_str() + pos, nullptr, 10));
        pos = comma + 1;
      }
      dumpPath = getenv("GADGET_HIP_TRAJ_DUMP");
      if (!dumpPath) dumpPath = "/tmp/gadgethip_trajtrace.log";
    }
  }
  if (targetIds.empty()) return;  // no-op unless GADGET_HIP_TRAJ_IDS is set

  tree.bodies_ids.d2h();
  tree.bodies_Ppos.d2h();
  tree.bodies_vel.d2h();
  tree.bodies_acc0.d2h();
  tree.bodies_time.d2h();

  FILE *flog = fopen(dumpPath, "a");
  if (!flog) return;
  for (int i = 0; i < tree.n; i++)
  {
    ullong id = tree.bodies_ids[i];
    bool tracked = false;
    for (ullong t : targetIds) if (t == id) { tracked = true; break; }
    if (!tracked) continue;

    real4 p  = tree.bodies_Ppos[i];
    real4 v  = tree.bodies_vel[i];
    real4 a  = tree.bodies_acc0[i];
    int2 bt = tree.bodies_time[i];
    double amag = sqrt((double)a.x * a.x + (double)a.y * a.y + (double)a.z * a.z);
    fprintf(flog,
            "iter=%d stage=%s t=%.9g id=%llu pos=(%.9g,%.9g,%.9g) vel=(%.9g,%.9g,%.9g) "
            "amag=%.9g tb=%.9g te=%.9g dt=%.9g\n",
            iter, stage, simTime, (unsigned long long) id,
            p.x, p.y, p.z, v.x, v.y, v.z, amag, bt.x, bt.y, (double)(bt.y - bt.x));
  }
  fclose(flog);
}

// Ticket T12: mechanism-level check for the resort-timing hypothesis (see
// tickets/T11-root-cause-bug-b-freeze.md and T12's own ticket). correct_particles() reads
// pVel/time via tree.oriParticleOrder[idx] ("unsortedIdx") translation. This dumps, right before
// correct() runs (i.e. using the permutation sort_bodies() just computed this same iteration,
// before correct_particles() resets it back to identity at the end of its kernel), the mapping
// idx -> oriParticleOrder[idx] for each GADGET_HIP_TRAJ_IDS-tracked particle's CURRENT-order slot,
// plus an explicit collision check across all tracked ids (a valid permutation must be a bijection
// -- two different current-order idx values mapping to the same oriParticleOrder[idx] is a direct,
// unambiguous smoking gun, not an inference). Zero-cost / no-op unless GADGET_HIP_TRAJ_IDS is set.
// Output: GADGET_HIP_TRAJ_UNSORTED_DUMP (default /tmp/gadgethip_unsorted.log).
static void dumpUnsortedMapping(tree_structure &tree, int iter)
{
  static bool initialized = false;
  static std::vector<ullong> targetIds;
  static const char *dumpPath = nullptr;
  if (!initialized)
  {
    initialized = true;
    const char *idsEnv = getenv("GADGET_HIP_TRAJ_IDS");
    if (idsEnv)
    {
      std::string s(idsEnv);
      size_t pos = 0;
      while (pos < s.size())
      {
        size_t comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        if (comma > pos) targetIds.push_back((ullong) strtoull(s.c_str() + pos, nullptr, 10));
        pos = comma + 1;
      }
      dumpPath = getenv("GADGET_HIP_TRAJ_UNSORTED_DUMP");
      if (!dumpPath) dumpPath = "/tmp/gadgethip_unsorted.log";
    }
  }
  if (targetIds.empty()) return;  // no-op unless GADGET_HIP_TRAJ_IDS is set

  tree.bodies_ids.d2h();
  tree.oriParticleOrder.d2h();
  tree.activePartlist.d2h();
  // Refined hypothesis (found while reading correct_particles/correct()): tree.bodies_pos and
  // tree.bodies_vel are the SAME persistent buffers passed directly into correctParticles (not a
  // scratch view) -- they are written in-kernel only for ACTIVE particles (pos[idx]=pPos[idx],
  // vel[idx]=v), and sort_bodies()'s default (non-doFullShuffle) per-iteration reorder never
  // touches bodies_pos/bodies_vel at all (only bodies_Ppos/bodies_ids/bodies_h get reordered).
  // So for an INACTIVE particle whose new-order idx this iteration differs from its previous
  // slot, bodies_Ppos[idx]/bodies_vel[idx] should still hold whatever a DIFFERENT particle wrote
  // there last -- dump the raw pre-correct() values here (before correct_particles can overwrite
  // them for active particles) to check directly, alongside active_list[idx].
  tree.bodies_Ppos.d2h();
  tree.bodies_vel.d2h();

  FILE *flog = fopen(dumpPath, "a");
  if (!flog) return;

  // idx (current-order) and unsortedIdx (=oriParticleOrder[idx]) for each tracked id present
  // this iteration, plus the id currently sitting at idx (bodies_ids[idx], for context -- this
  // is the id the mapping is FOR, already in new/current order at this point in the pipeline).
  std::vector<std::pair<ullong,int>> found; // (id, idx)
  std::vector<uint> unsortedVals;
  for (int i = 0; i < tree.n; i++)
  {
    ullong id = tree.bodies_ids[i];
    for (ullong t : targetIds)
    {
      if (t == id)
      {
        uint u = tree.oriParticleOrder[i];
        uint act = tree.activePartlist[i];
        real4 rawPos = tree.bodies_Ppos[i];
        real4 rawVel = tree.bodies_vel[i];
        fprintf(flog,
                "iter=%d id=%llu idx=%d unsortedIdx=%u active=%u "
                "preCorrectPos=(%.9g,%.9g,%.9g) preCorrectVel=(%.9g,%.9g,%.9g)\n",
                iter, (unsigned long long) id, i, u, act,
                rawPos.x, rawPos.y, rawPos.z, rawVel.x, rawVel.y, rawVel.z);
        found.push_back({id, i});
        unsortedVals.push_back(u);
        break;
      }
    }
  }
  // Explicit bijection/collision check across just the tracked ids found this iteration.
  for (size_t a = 0; a < found.size(); a++)
    for (size_t b = a + 1; b < found.size(); b++)
      if (unsortedVals[a] == unsortedVals[b])
        fprintf(flog,
                "iter=%d COLLISION: id=%llu (idx=%d) and id=%llu (idx=%d) both map to "
                "unsortedIdx=%u -- oriParticleOrder is NOT a bijection this iteration\n",
                iter, (unsigned long long) found[a].first, found[a].second,
                (unsigned long long) found[b].first, found[b].second, unsortedVals[a]);
  fclose(flog);
}


hipEvent_t startLocalGrav;
hipEvent_t startRemoteGrav;
hipEvent_t endLocalGrav;
hipEvent_t endRemoteGrav;

float runningLETTimeSum, lastTotal, lastLocal;


void octree::makeLET()
{
#ifdef USE_MPI
   //LET code test
  double t00 = get_time();

  //Start copies, while grpTree info is exchanged
  localTree.boxSizeInfo.d2h  (  localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.boxCenterInfo.d2h(  localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.multipole.d2h    (3*localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.boxSizeInfo.waitForCopyEvent();
  localTree.boxCenterInfo.waitForCopyEvent();

  double t10 = get_time();
  //Exchange domain grpTrees, while memory copies take place
  this->sendCurrentInfoGrpTree();

  double t20 = get_time();


  localTree.multipole.waitForCopyEvent();
  double t40 = get_time();
  LOGF(stderr,"MakeLET Preparing data-copy: %lg  sendGroups: %lg Total: %lg \n",
               t10-t00, t20-t10, t40-t00);

  std::vector<real4> topLevelsBuffer;
  std::vector<uint2> treeSizeAndOffset;
  int copyTreeUpToLevel = 0;
  //Start LET kernels
  essential_tree_exchangeV2(localTree,
                            remoteTree,
                            topLevelsBuffer,
                            treeSizeAndOffset,
                            copyTreeUpToLevel);

  letRunning = false;
#endif
}



void octree::iterate_setup() {

  if(execStream == NULL)
  {
      if(execStream == NULL)          execStream          = new my_dev::dev_stream(0);
      if(gravStream == NULL)          gravStream          = new my_dev::dev_stream(0);
      if(copyStream == NULL)          copyStream          = new my_dev::dev_stream(0);
      if(LETDataToHostStream == NULL) LETDataToHostStream = new my_dev::dev_stream(0);

      CU_SAFE_CALL(hipEventCreate(&startLocalGrav));
      CU_SAFE_CALL(hipEventCreate(&endLocalGrav));
      CU_SAFE_CALL(hipEventCreate(&startRemoteGrav));
      CU_SAFE_CALL(hipEventCreate(&endRemoteGrav));

      devContext->writeLogEvent("Start execution\n");
  }

  //Setup of the multi-process particle distribution, initially it should be equal
  #ifdef USE_MPI
    if(nProcs > 1)
    {
      for(int i=0; i < 5; i++)
      {
        double notUsed     = 0;
        int maxN = 0, minN = 0;
        sort_bodies(localTree, true, true); //Initial sort to get global boundaries to compute keys
        parallelDataSummary(localTree, 30, 30, notUsed, notUsed, true); //1 for all process, equal part distribution

        //Check if the min/max are within certain percentage
        MPI_Allreduce(&localTree.n, &maxN, 1, MPI_INT, MPI_MAX, mpiCommWorld);
        MPI_Allreduce(&localTree.n, &minN, 1, MPI_INT, MPI_MIN, mpiCommWorld);

        //Compute difference in percent
        int perc = (int)(100*(maxN-minN)/(double)minN);

        if(procId == 0)
        {
          LOGF(stderr, "Particle setup iteration: %d Min: %d  Max: %d Diff: %d %%\n", i, minN, maxN, perc);
        }
        if(perc < 10) break; //We're happy if difference is less than 10%
      }
    }
  #endif

  sort_bodies(localTree, true, true); //Initial sort to get global boundaries to compute keys
  letRunning      = false;
}

// returns true if this iteration is the last (t_current >= t_end), false otherwise

// C-A-07 probe (contracts/CONTRACTS-A-treebuild.md), enabled with GADGET_HIP_CA07=1.
//
// Gadget-2 maintains its tree between rebuilds with force_update_len() (forcetree.c:940-960),
// which can only ever GROW a node: `if(distmax + distmax > Nodes[no].len) Nodes[no].len = ...`.
// The port instead refits each node's AABB exactly from the current bodies_Ppos, which can also
// SHRINK it. A shrinking node-size proxy is the mechanism behind the prior audit's T24 blowup
// (bJ shrinking pathologically small lets the walk approximate where it should split), so the
// question is whether the refit actually shrinks nodes in practice, and by how much.
//
// This must be settled BEFORE any --rebuild N>1 experiment (C-A-06's probe included), because
// that path is dead at the default rebuild_tree_rate=1 and silently activates when it is raised.
static std::vector<float4> g_ca07_ref;
static int                 g_ca07_refNodes = -1;

static bool ca07_enabled()
{
  static const bool on = (getenv("GADGET_HIP_CA07") != NULL);
  return on;
}

void octree::ca07_probe(tree_structure &tree, bool wasRebuild, int iter)
{
  if (!ca07_enabled()) return;
  tree.boxSizeInfo.d2h();
  const int n = tree.n_nodes;
  if (wasRebuild)
  {
    g_ca07_ref.assign(tree.boxSizeInfo.raw_p(), tree.boxSizeInfo.raw_p() + n);
    g_ca07_refNodes = n;
    fprintf(stderr, "[CA07] iter=%d REBUILD n_nodes=%d (reference stored)\n", iter, n);
    return;
  }
  if (g_ca07_refNodes != n)
  {
    fprintf(stderr, "[CA07] iter=%d node count changed %d -> %d without a rebuild; "
                    "comparison skipped\n", iter, g_ca07_refNodes, n);
    return;
  }
  // Compare the per-node max half-extent, which is what the MAC's node-size proxy is built from.
  int shrunk = 0, grew = 0;
  double worstShrinkFrac = 0.0, sumRef = 0.0, sumNow = 0.0;
  for (int i = 0; i < n; i++)
  {
    const float4 r = g_ca07_ref[i], c = tree.boxSizeInfo[i];
    const float mr = fmaxf(fmaxf(r.x, r.y), r.z);
    const float mc = fmaxf(fmaxf(c.x, c.y), c.z);
    sumRef += mr; sumNow += mc;
    if (mc < mr) { shrunk++; if (mr > 0.0f) worstShrinkFrac = fmax(worstShrinkFrac, (double)(mr - mc) / mr); }
    else if (mc > mr) grew++;
  }
  fprintf(stderr, "[CA07] iter=%d n_nodes=%d shrunk=%d (%.2f%%) grew=%d (%.2f%%) "
                  "worst_shrink=%.4f sum_ratio=%.6f\n",
          iter, n, shrunk, 100.0*shrunk/n, grew, 100.0*grew/n, worstShrinkFrac,
          sumRef > 0 ? sumNow/sumRef : 0.0);
}


// C-A-19 / C-A-21 invariant probe (GADGET_HIP_INVARIANTS=1).
//
// These are the contract set's INVARIANT entries -- "every particle belongs to exactly one leaf,
// none omitted" (C-A-19) and "level 0 is one node spanning all N particles" (C-A-21). Both sat at
// verdict UNKNOWN, unprobed, because they have no Gadget-2 side to read off. Two real bugs this
// session (T21's root box, and the group AABB reductions) landed on exactly these invariants.
//
// The leaf partition is produced by gpuCompact/gpuSplit, which are built on the unsynchronised
// warp-cooperative scans in scanKernels.cu. If those scans are wrong -- ever, even rarely -- the
// partition is the place it shows up: leaves would overlap, leave gaps, or miscount. So this is
// also the direct test of whether those scans are producing correct results.
void octree::invariant_probe(tree_structure &tree, int iter)
{
  static const bool on = (getenv("GADGET_HIP_INVARIANTS") != NULL);
  if (!on) return;

  // Buffer-duplication check (memory work, PLAN 4.1). Two candidates for removal:
  //   bodies_Pvel -- predict_particles does `pVel[idx] = v` with NO transformation (Gadget-2 does
  //     not predict DM velocities at all; its VelPred is gas-only, predict.c:56-65), so this
  //     should be an exact copy of bodies_vel.
  //   bodies_pos  -- correct_particles commits `pos[idx] = pPos[idx]` on BOTH the active and the
  //     inactive branch, and since the T6/C-D-01 fix the drift anchor is the GLOBAL previous sync
  //     time rather than the particle's own Ti_begstep. That makes the pair algebraically an
  //     in-place drift (Gadget-2's move_particles), so pos should equal pPos at every step
  //     boundary.
  // Both claims are checked here rather than assumed -- this is the area that produced four
  // index-space bugs, so "it looks redundant" is not sufficient grounds to delete 4 GB.
  {
    // Only the velocity pair is comparable HERE: bodies_vel and bodies_Pvel are both
    // SPACE_ORIGINAL, so element-wise is valid. bodies_pos (ORIGINAL) against bodies_Ppos
    // (CURRENT) is NOT -- after the sort those indices name different particles, and comparing
    // them straight gives a meaningless "everything differs". That check lives at the top of the
    // iteration instead (see dup_check_pos), before the sort, where the spaces coincide.
    tree.bodies_vel.d2h();  tree.bodies_Pvel.d2h();
    long long velDiff = 0;
    double velMax = 0.0;
    for (int i = 0; i < tree.n; i++)
    {
      const real4 v = tree.bodies_vel[i],  pv = tree.bodies_Pvel[i];
      const double dv = std::max(std::max(fabs(v.x-pv.x), fabs(v.y-pv.y)), fabs(v.z-pv.z));
      if (dv != 0.0) { velDiff++; velMax = std::max(velMax, dv); }
    }
    fprintf(stderr, "[INVARIANT] dup-check iter=%d  bodies_Pvel vs bodies_vel: %lld/%d differ "
                    "(max %.3e)\n", iter, velDiff, tree.n, velMax);
  }

  tree.node_bodies.d2h();
  tree.leafNodeIdx.d2h();

  std::vector<std::pair<uint,uint>> leaves;
  leaves.reserve(tree.n_leafs);
  for (int i = 0; i < tree.n_leafs; i++)
  {
    uint2 nb = tree.node_bodies[tree.leafNodeIdx[i]];
    leaves.push_back(std::make_pair(nb.x & ILEVELMASK, nb.y));
  }
  std::sort(leaves.begin(), leaves.end());

  // C-A-19: the leaves must tile [0, n) exactly -- no gap, no overlap, nothing out of range.
  long long gaps = 0, overlaps = 0, oob = 0, covered = 0;
  uint expect = 0;
  for (size_t i = 0; i < leaves.size(); i++)
  {
    uint s = leaves[i].first, e = leaves[i].second;
    if (e > (uint)tree.n || s > e) { oob++; continue; }
    if (s > expect) gaps++;
    if (s < expect) overlaps++;
    covered += (long long)(e - s);
    expect = std::max(expect, e);
  }
  const bool tiles = (gaps == 0 && overlaps == 0 && oob == 0 &&
                      covered == (long long)tree.n && expect == (uint)tree.n);

  // C-A-21: level 0 is a single node spanning every particle.
  uint2 root = tree.node_bodies[0];
  const bool rootOk = ((root.x & ILEVELMASK) == 0u && root.y == (uint)tree.n);

  fprintf(stderr,
    "[INVARIANT] iter=%d n=%d n_leafs=%d | C-A-19 leaf partition: %s "
    "(gaps=%lld overlaps=%lld oob=%lld covered=%lld/%d) | C-A-21 root spans all: %s (%u..%u)\n",
    iter, tree.n, tree.n_leafs, tiles ? "OK" : "VIOLATED",
    gaps, overlaps, oob, covered, tree.n, rootOk ? "OK" : "VIOLATED",
    root.x & ILEVELMASK, root.y);

  // C-A-20: START_LEVEL_MIN_NODES forces a minimum tree depth that Gadget-2 has no counterpart
  // for. Its probe asks for (level, startLevelMin, n_leafs, n_nodes) per rebuild, and whether
  // any leaf terminates ABOVE startLevelMin -- which would mean the forced floor is actually
  // binding rather than merely advisory.
  {
    int shallow = 0;
    for (int i = 0; i < tree.n_leafs; i++)
    {
      uint2 nb = tree.node_bodies[tree.leafNodeIdx[i]];
      const int lvl = (int)(nb.x >> BITLEVELS);
      if (lvl < tree.startLevelMin) shallow++;
    }
    fprintf(stderr, "[INVARIANT] C-A-20 forced depth: n_levels=%d startLevelMin=%d n_nodes=%d "
                    "n_leafs=%d leaves_above_floor=%d\n",
            tree.n_levels, tree.startLevelMin, tree.n_nodes, tree.n_leafs, shallow);
  }
}




// T14 spike hunt (GADGET_HIP_SPIKE=1). The 512^3 run shows a transient Epot/Ekin excursion
// (de -> 41 then 1774, Ekin up 670x, then a full recovery on a non-advancing step). This reports
// the extreme particles at the moment the forces are final, so the excursion can be attributed to
// specific ids rather than inferred from an aggregate. Reports the top few by |a| and by |Phi|,
// plus how many particles are far out of family, so a single-particle cause and a population-wide
// one are distinguishable.
void octree::spike_probe(tree_structure &tree, int iter, const char *where)
{
  static const bool on = (getenv("GADGET_HIP_SPIKE") != NULL);
  if (!on || tree.n <= 0) return;
  tree.bodies_acc1.d2h(); tree.bodies_Ppos.d2h();
  tree.bodies_ids.d2h();  tree.bodies_vel.d2h();

  const int K = 5;
  std::vector<int> byA(tree.n), byP(tree.n);
  for (int i = 0; i < tree.n; i++) { byA[i] = i; byP[i] = i; }
  auto amag = [&](int i){ const real4 a = tree.bodies_acc1[i];
                          return sqrt((double)a.x*a.x + (double)a.y*a.y + (double)a.z*a.z); };
  auto pmag = [&](int i){ return fabs((double)tree.bodies_acc1[i].w); };
  std::partial_sort(byA.begin(), byA.begin()+K, byA.end(),
                    [&](int x,int y){ return amag(x) > amag(y); });
  std::partial_sort(byP.begin(), byP.begin()+K, byP.end(),
                    [&](int x,int y){ return pmag(x) > pmag(y); });

  // population scale: median |a| via a cheap sample, and counts far above it
  std::vector<double> samp;
  const int stride = std::max(1, tree.n / 20000);
  for (int i = 0; i < tree.n; i += stride) samp.push_back(amag(i));
  std::sort(samp.begin(), samp.end());
  const double medA = samp.empty() ? 0.0 : samp[samp.size()/2];
  long long n100 = 0, n1e4 = 0;
  for (int i = 0; i < tree.n; i++) { const double a = amag(i);
    if (a > 100.0*medA) n100++; if (a > 1e4*medA) n1e4++; }

  fprintf(stderr, "[SPIKE] iter=%d %s median|a|=%.6g  |a|>100x: %lld  |a|>1e4x: %lld  "
                  "n_active=%d n_groups=%d n_active_groups=%d\n",
          iter, where, medA, n100, n1e4, tree.n_active_particles, tree.n_groups,
          tree.n_active_groups);
  for (int k = 0; k < K; k++)
  {
    const int i = byA[k]; const real4 a = tree.bodies_acc1[i];
    const real4 p = tree.bodies_Ppos[i]; const real4 v = tree.bodies_vel[i];
    fprintf(stderr, "[SPIKE]   top|a| #%d id=%llu |a|=%.6g phi=%.6g pos=(%.6g,%.6g,%.6g) "
                    "|v|=%.6g\n", k, (unsigned long long)tree.bodies_ids[i], amag(i), (double)a.w,
            (double)p.x, (double)p.y, (double)p.z,
            sqrt((double)v.x*v.x + (double)v.y*v.y + (double)v.z*v.z));
  }
  for (int k = 0; k < K; k++)
  {
    const int i = byP[k]; const real4 p = tree.bodies_Ppos[i];
    fprintf(stderr, "[SPIKE]   top|phi| #%d id=%llu phi=%.6g |a|=%.6g pos=(%.6g,%.6g,%.6g)\n",
            k, (unsigned long long)tree.bodies_ids[i], (double)tree.bodies_acc1[i].w, amag(i),
            (double)p.x, (double)p.y, (double)p.z);
  }
}

// Phase timing. Every phase below is bracketed by a device sync, because the kernels are queued
// asynchronously and timing the launch would be meaningless.
//
// Two consumers, deliberately separated:
//   * cpu.txt          -- cumulative per-phase seconds in Gadget-2's own format, written every
//                         step like Gadget writes its own. ON by default under --param.
//   * GADGET_HIP_PHASE_TIME -- the per-step [PHASE] lines, for attribution while optimising. Off
//                         by default; it is verbose.
//
// The syncs cost real time. They do not add work -- they only stop the host running ahead of the
// device -- but that is a claim to measure, not assume, and it is measured in tickets/T39.
// GADGET_HIP_NO_CPU_LOG=1 turns the whole thing off, which is also the A/B control for that
// measurement.
struct GadgetCpuAcc {
  double total, domain, potential, predict, timeline, snapshot, treewalk, treebuild, pm, peano;
};
static GadgetCpuAcc s_cpuAcc = {0,0,0,0,0,0,0,0,0,0};
double *gadget_cpu_snapshot_acc() { return &s_cpuAcc.snapshot; }

// ------------------------------------------------------------------------------------------
// Phase 3: GADGET-2's long-range force as a separate kick over the PM interval.
//
// File scope rather than locals of iterate_once() because octree::correct() -- same translation
// unit -- needs all of it: the stored force for the timestep criterion, and the interval state to
// decide whether this step owes every particle a long-range kick.
//
// PM_Ti_begstep/endstep are All.PM_Ti_begstep/All.PM_Ti_endstep. Both the PM solve (in the
// gravity phase) and the kick (in correct()) test `PM_Ti_endstep == Ti_current`, and the interval
// is advanced only in correct(), so the two see the same pre-advance value within one step --
// which is precisely how GADGET-2 orders compute_accelerations() and advance_and_find_timesteps().
// GFreeAcc, not float: the PM solve runs with gravityConstant = 1.0f, so these owe a factor G and
// every consumer must say which one it applies. See include/gadget_units.h; it is what finally
// made T42's missing G, and T47's, impossible to write. The potential stays a plain float -- its
// G travels with the t30 coefficients in compute_energies(), a different path.
static GFreeAcc *s_gravPMx = nullptr, *s_gravPMy = nullptr, *s_gravPMz = nullptr;
static float    *s_gravPMpot = nullptr;

// C-A-07 / Phase 4: the per-node extent floor that makes a non-rebuild step safe. Allocated lazily
// and ONLY when a step actually skips the rebuild, so a run at the default (rebuild every step) pays
// nothing for it -- one float4 per node is roughly 600 MB at 512^3.
static float4 *s_nodeLenFloor      = nullptr;
// OPT-IN (GADGET_HIP_CA07_MONOTONE=1), and the reasoning is worth keeping because the first version
// of this had it on by default, described as a bug fix. It is not one.
//
// GADGET grows node extents monotonically (force_update_len) BECAUSE IT NEVER RECOMPUTES THEM. It
// keeps Nodes[].center fixed at build time and has only Father[i] plus a per-particle distance check
// to work with, so enlarging is the one cheap conservative option available to it; its boxes are
// supersets of the true extent.
//
// This port recomputes instead: compute_properties() rebuilds every node's box exactly from the
// current positions, bottom-up, with each parent's box the UNION of its children's
// (compute_non_leaf -> compute_bounds_node). On a non-rebuild step the port skips sort_bodies and
// build, so node membership is unchanged and that box is the exact extent of the node's own
// particles -- and parent-contains-children, the invariant GADGET maintains by propagating growth up
// the ancestor chain, holds here by construction. A box that is the true extent of its particles is
// not a "pathologically small" box, so T20's identification of the refit with the T24 blowup
// mechanism does not survive inspection.
//
// Measured, 64^3, --rebuild 4, three runs each: floor ON 22.3 +- 0.1 s, OFF 22.1 +- 0.1 s, and the
// tree walk identical at 13.9 s either way -- so the 0.9% is this kernel and its memcpy, not a looser
// MAC. The reason it changes so little: 42.7% of nodes do shrink, but total node size moves by
// 0.007% (sum_ratio 1.000066), i.e. the nodes that shrink carry almost no weight.
//
// Kept, opt-in, for the case where someone wants the port's stale-step geometry to be provably no
// less conservative than GADGET's -- e.g. before changing the opening criterion.
static bool ca07MonotoneWanted()
{
  static const bool on = (getenv("GADGET_HIP_CA07_MONOTONE") != NULL) &&
                         (atoi(getenv("GADGET_HIP_CA07_MONOTONE")) != 0);
  return on;
}
static int     s_nodeLenFloorNodes = 0;
static bool   s_pmCadenceActive = false;   // the mode actually in effect, not what was requested
static gadget_tick_t s_PM_Ti_begstep = 0, s_PM_Ti_endstep = 0;
static GadgetDriftTables s_gadgetTables;
static bool   s_gadgetTablesOk = false;
static unsigned char *s_pmKickAudit = nullptr;   // validation rung 1; NULL unless audited
static void  *s_nullDevPtr = nullptr;            // set_args dereferences its arguments
static long   s_pmKickCount = 0;
static int    s_pmKickNoPerm = 0;   // write-target A/B, see gadget_pm_kick                 // long-range kicks applied, for the banner

void pmstale_reset(void);
void pmstale_read(double *num, double *den);

// T56: see octree.h. Deliberately a lookup over the sorted list rather than an index the caller
// has to keep in step: the snapshot block below can cross several output times in one system step,
// and an index would then need the same catch-up loop the cadence path has.
double octree::nextOutputTimeAfter(double t) const
{
  if (outputListActive())
  {
    for (size_t i = 0; i < gadgetParams.OutputListTimes.size(); ++i)
      if (gadgetParams.OutputListTimes[i] > t) return gadgetParams.OutputListTimes[i];
    return std::numeric_limits<double>::infinity();
  }
  if (haveGadgetParams && gadgetParams.ComovingIntegrationOn)
    return nextGadgetSnapTime * gadgetParams.TimeBetSnapshot;
  return nextGadgetSnapTime + gadgetParams.TimeBetSnapshot;
}

bool octree::iterate_once(IterationData &idata) {
  static const bool phaseTimePrint = (getenv("GADGET_HIP_PHASE_TIME") != NULL);
  // Gadget writes cpu.txt unconditionally; a diagnostic that is off by default is not there when
  // it is needed. It needs an OutputDir to write into, so it follows --param.
  static const bool cpuLogOn = haveGadgetParams && !getenv("GADGET_HIP_NO_CPU_LOG");
  const bool phaseTimeOn = cpuLogOn || phaseTimePrint;
  double phT = 0.0;
  double phStepT0 = 0.0;
  if (phaseTimeOn) { execStream->sync(); gravStream->sync(); phStepT0 = get_time(); }
  #define PHASE_BEGIN() do { if (phaseTimeOn) { execStream->sync(); gravStream->sync(); phT = get_time(); } } while (0)
  #define PHASE_END(label, bucket) do { if (phaseTimeOn) { execStream->sync(); gravStream->sync(); \
        const double dt_ = get_time() - phT; s_cpuAcc.bucket += dt_; \
        if (phaseTimePrint) fprintf(stderr, "[PHASE] iter=%d %-22s %8.4f s\n", iter, label, dt_); } } while (0)

    double t1 = 0;

    //if(t_current < 1) //Clear startup timings
    //if(0)
    if(iter < 32)
    {
      idata.totalGPUGravTimeLocal = 0;
      idata.totalGPUGravTimeLET   = 0;
      idata.totalLETCommTime      = 0;
      idata.totalBuildTime        = 0;
      idata.totalDomTime          = 0;
      idata.lastWaitTime          = 0;
      idata.startTime             = get_time();
      idata.totalGravTime         = 0;
      idata.totalDomUp            = 0;
      idata.totalDomEx            = 0;
      idata.totalDomWait          = 0;
      idata.totalPredCor          = 0;
    }


    LOG("At the start of iterate:\n");

    bool forceTreeRebuild = false;
    bool needDomainUpdate = true;

    double tTempTime = get_time();

    //predict local tree
    devContext->startTiming(execStream->s());
    PHASE_BEGIN();
    predict(this->localTree);
    PHASE_END("predict", predict);

    // T37: map the particles back onto the box, where Gadget does it -- after the drift, before
    // the tree is built and before the PM assignment (domain.c:81, reached from run.c:44 every
    // step the decomposition is due). Off entirely in a non-periodic build, exactly as Gadget's
    // own `#ifdef PERIODIC` around the call.
#ifdef PERIODIC
    {
      float t37Box = pm_get_periodic_boxsize();   // the box the walk itself was set up with
      static const bool t37Off = (getenv("GADGET_HIP_NO_BOX_WRAP") != NULL);
      if (t37Box > 0.0f && !t37Off && this->localTree.n > 0)
      {
        PHASE_BEGIN();
        int t37N = this->localTree.n;
        gadgetBoxWrap.set_args(0, &t37N, &t37Box, this->localTree.bodies_Ppos.p());
        gadgetBoxWrap.setWork(this->localTree.n, 128);
        gadgetBoxWrap.execute2(execStream->s());
        PHASE_END("box_wrap", domain);
      }
    }
#endif
    cd01_trace(this->localTree, "PRE", iter, t_previous, t_current);

    // C-D-26: Gadget-2's per-step diagnostic line, emitted at the same point in the loop it uses
    // (every_timestep_stuff(), run.c:375-392, straight after the sync point is found and the drift
    // applied) and in the SAME format, to both stdout and an `info.txt` in OutputDir.
    //
    // Why this is worth its one line per step: the port previously printed no Systemstep, no
    // Dloga and no redshift, and the one partial line it did have sat behind `--log`
    // (ENABLE_RUNTIME_LOG is false unless the flag is passed). That is the direct reason C-D-08
    // (does the displacement ceiling ever bind?) and C-D-13 (does the bin ladder fragment?) could
    // not be settled from the port's own output -- both are questions about the SYSTEM STEP, which
    // was never reported. `Dloga` in particular makes the C-D-04 ladder and any C-D-13
    // fragmentation visible directly: on a correct ladder its distinct values are a short run of
    // consecutive powers of two, exactly as the reference `info.txt` shows.
    //
    // Emitted unconditionally (not behind --log) precisely because Gadget emits it unconditionally;
    // a diagnostic that is off by default is the failure mode this audit keeps finding.
    if (haveGadgetParams && mpiGetRank() == 0)
    {
      static FILE *s_infoFile = NULL;
      if (!s_infoFile && !gadgetParams.OutputDir.empty())
      {
        std::string ip = gadgetParams.OutputDir;
        if (!ip.empty() && ip[ip.size()-1] != '/') ip += '/';
        ip += "info.txt";
        s_infoFile = fopen(ip.c_str(), "w");
      }
      const double sysStep = (double) t_current - (double) t_previous;
      char line[256];
      if (gadgetParams.ComovingIntegrationOn && t_current > 0.0f && t_previous > 0.0f)
      {
        const double z = 1.0 / (double) t_current - 1.0;
        snprintf(line, sizeof(line),
                 "\nBegin Step %d, Time: %g, Redshift: %g, Systemstep: %g, Dloga: %g\n",
                 iter, (double) t_current, z, sysStep,
                 log((double) t_current) - log((double) t_previous));
      }
      else
      {
        snprintf(line, sizeof(line), "\nBegin Step %d, Time: %g, Systemstep: %g\n",
                 iter, (double) t_current, sysStep);
      }
      fputs(line, stdout);
      if (s_infoFile) { fputs(line, s_infoFile); fflush(s_infoFile); }
    }

    // Gadget-2 refreshes the potential for EVERY particle before an energy statistic
    // (run.c:55 -> compute_potential()). Do the same by activating every group for this one
    // iteration, which makes the walk write a fresh acc.w for all of them. Cadence matches
    // Gadget's TimeBetStatistics; between statistics the cheap (stale-potential) de is still
    // logged, exactly as before, but only the [ENERGY-EXACT] lines are metric-grade.
    double &nextEnergyStat = this->restartNextEnergyStat;   // member, so a restart can restore it
    bool energyStatDue = false;
    if (haveGadgetParams && gadgetParams.TimeBetStatistics > 0.0)
    {
      if (nextEnergyStat < 0.0) nextEnergyStat = (double) t_current;
      energyStatDue = ((double) t_current >= nextEnergyStat);
    }
    energy_set_force_all_active(energyStatDue ? 1 : 0);

    // Hoisted (T35/T36): whether this step will evaluate the global energies. Needed here, before
    // the PM block, because the PM long-range potential feeds Epot -- so any step that samples the
    // energies must sample a FRESH one, exactly as Gadget's compute_potential() does at statistics
    // time rather than reading a stale stored value.
    static const bool energyEveryStep = (getenv("GADGET_HIP_ENERGY_EVERY_STEP") != NULL)
                                          ? (atoi(getenv("GADGET_HIP_ENERGY_EVERY_STEP")) != 0)
                                          : !haveGadgetParams;
    const bool energyDue = energyEveryStep || energyStatDue || (iter == 0) ||
                           (t_current >= tEnd) || (iter >= iterEnd);
    devContext->stopTiming("Predict", 9, execStream->s());
    // Gated 2026-08-30 (user's FRMF-sims N=134M perf test, PLAN.md): this diagnostic does a full
    // .d2h() of bodies_Ppos plus an O(n) std::unordered_map build over every particle every
    // iteration -- at N=32768 (where it was written, LOG.md Sec48) that's negligible; at N=134M
    // it dominates wall-clock (~100s+ of a ~211s/iter total, vs. the real GPU gravity kernel's own
    // ~2ms), which is what actually produced the near-0% GPU utilization the user observed. Kept,
    // not deleted -- still a real, reusable tool for a future close-encounter/duplicate-position
    // investigation -- just no longer unconditional.
    if (getenv("GADGET_HIP_TRACE_DUP"))
      traceDuplicatePositions(this->localTree, this->localTree.bodies_Ppos, "AFTER_PREDICT", iter);
    // Ticket T11: same call pattern, see traceParticleTrajectory's own comment above. No-op
    // unless GADGET_HIP_TRAJ_IDS is set (checked internally, cheap even when unset).
    traceParticleTrajectory(this->localTree, "AFTER_PREDICT", iter, (double) t_current);

    // BUG4-DIAG: dump device address ranges of every suspect scratch buffer each
    // iteration, to test whether tnext's independent allocation is address-adjacent
    // to (or overlapping) generalBuffer1's block -- temporary instrumentation.
    if (gadget_hip_debug_log)
    {
      auto rangeEnd = [](void* p, long bytes){ return (char*)p + bytes; };
      void* gb1 = this->localTree.generalBuffer1.get_devMem();
      long  gb1Bytes = (long)this->localTree.generalBuffer1.get_size() * sizeof(uint);
      void* tn  = this->tnext.get_devMem();
      long  tnBytes = (long)this->tnext.get_size() * sizeof(float);
      void* agl = this->localTree.activeGrpList.get_devMem();
      long  aglBytes = (long)this->localTree.activeGrpList.get_size() * sizeof(uint);
      void* b2g = this->localTree.body2group_list.get_devMem();
      long  b2gBytes = (long)this->localTree.body2group_list.get_size() * sizeof(uint);
      fprintf(stderr, "[BUG4-ADDR] gb1=[%p,%p) tnext=[%p,%p) activeGrpList=[%p,%p) body2group=[%p,%p)\n",
              gb1, rangeEnd(gb1, gb1Bytes),
              tn, rangeEnd(tn, tnBytes),
              agl, rangeEnd(agl, aglBytes),
              b2g, rangeEnd(b2g, b2gBytes));
    }

    idata.totalPredCor += get_time() - tTempTime;

    if(nProcs > 1)
    {
      //if(1) //Always update domain boundaries/particles
      if((iter % rebuild_tree_rate) == 0)
      {
        double domUp =0, domEx = 0;
        double tZ = get_time();
        devContext->startTiming(execStream->s());
        parallelDataSummary(localTree, lastTotal, lastLocal, domUp, domEx, false);
        devContext->stopTiming("UpdateDomain", 6, execStream->s());
        double tZZ = get_time();
        idata.lastDomTime   = tZZ-tZ;
        idata.totalDomTime += idata.lastDomTime;

        idata.totalDomUp += domUp;
        idata.totalDomEx += domEx;

        devContext->startTiming(execStream->s());
        mpiSync();
        devContext->stopTiming("DomainUnbalance", 12, execStream->s());

        idata.totalDomWait += get_time()-tZZ;

        needDomainUpdate    = false; //We did a boundary sync in the parallel decomposition part
        needDomainUpdate    = true; //TODO if I set it to false results degrade. Check why, for now just updte
      }
    }

    if (useDirectGravity)
    {
      devContext->startTiming(gravStream->s());
      direct_gravity(this->localTree);
      devContext->stopTiming("Direct_gravity", 4);
    }
    else
    {
      //Build the tree using the predicted positions
      // bool rebuild_tree = Nact_since_last_tree_rebuild > 4*this->localTree.n;
      bool rebuild_tree = true;

      rebuild_tree = ((iter % rebuild_tree_rate) == 0);

      // C-A-07 (contracts/CONTRACTS-A-treebuild.md, tickets/T20): rebuild_tree_rate > 1 takes the
      // else-branch below, which calls compute_properties() and NOTHING else. Gadget-2 maintains
      // its tree on those steps -- force_update_len() (forcetree.c:940-960, monotone GROWTH only),
      // node-COM drift (predict.c:78-84) and a node-velocity kick (timestep.c:328-339). This port
      // has no counterpart: it refits each node's AABB exactly from the current bodies_Ppos over
      // the STALE node_bodies ranges, which can also SHRINK a node. Measured with
      // GADGET_HIP_CA07=1: 8.5% of nodes shrink on the first stale step, worst case to 8% of the
      // rebuild extent -- the same pathologically-small-node-size mechanism as the prior audit's
      // T24 blowup.
      //
      // Warn rather than refuse: --rebuild N is a legitimate diagnostic (C-A-05's own probe holds
      // the tree fixed for N steps on purpose). But it is never a PERFORMANCE win -- measured at
      // 2M, s/iter goes 0.3553 -> 0.9776 from --rebuild 1 to 16, because stale nodes grow (3.35x
      // over 7 steps) and the walk opens far more of them than the skipped builds save.
      if (rebuild_tree_rate > 1)
      {
        static bool s_ca07Warned = false;
        if (!s_ca07Warned)
        {
          // Both of this warning's original claims are now false, and the history is worth keeping
          // because the measurement that produced them was sound and still mis-generalised.
          //
          // It said node geometry is unmaintained: that was true and is now fixed -- the extent
          // floor below gives GADGET's force_update_len semantics, and the C-A-07 probe reports
          // shrunk=0.00% on every stale step where it previously reported 42.7%.
          //
          // It said the setting is "slower, not faster", from T20's measurement at 2M Plummer
          // (0.3553 -> 0.9776 s/iter from rebuild 1 to 16). That measurement is reproducible and its
          // conclusion does not transfer: Plummer runs a SHARED timestep, so every step is a dense
          // step where the walk dominates and skipping a build loses. A cosmological run with
          // individual timesteps spends most steps with few active particles -- at 512^3, 75% of
          // steps have a walk cheaper than the rebuild, and the rebuild is flat at 1.077 s +- 0.045
          // regardless of the active set, because it processes all N particles either way.
          // Re-measured at 64^3 cosmological with the floor active, three runs each:
          // rebuild 1 = 26.6 +- 0.1 s, rebuild 4 = 22.3 +- 0.3 s, a 16.2% saving at 19.7 sd.
          fprintf(stderr,
            "\n[C-A-07] rebuild_tree_rate=%d (>1). Node geometry IS maintained between rebuilds "
            "(monotone extent floor, GADGET force_update_len semantics; GADGET_HIP_CA07_MONOTONE=0 "
            "disables it and restores the shrinking behaviour). Measured FASTER than every-step on "
            "cosmological ICs with individual timesteps -- 16.2%% at 64^3 with rebuild 4 -- and "
            "slower on a shared-timestep Plummer run, which is what tickets/T20 measured. It "
            "changes trajectories, so validate against your own reference before using it for "
            "science. See contracts/CONTRACTS-A-treebuild.md C-A-07 and tickets/T20.\n\n",
            rebuild_tree_rate);
          s_ca07Warned = true;
        }
      }
      if(rebuild_tree)
      {
        //Rebuild the tree
        t1 = get_time();
        STAGE_TRACE("[BUG4-STAGE] before sort_bodies iter=%d\n", iter);
        PHASE_BEGIN();
        this->sort_bodies(this->localTree, needDomainUpdate);
        PHASE_END("sort_bodies", peano);
        STAGE_TRACE("[BUG4-STAGE] after sort_bodies / before build iter=%d\n", iter);
        PHASE_BEGIN();
        this->build(this->localTree);
        PHASE_END("build_tree", treebuild);
        STAGE_TRACE("[BUG4-STAGE] after build n_groups=%d iter=%d\n", this->localTree.n_groups, iter);
        LOGF(stderr, " done in %g sec : %g Mptcl/sec\n", get_time()-t1, this->localTree.n/1e6/(get_time()-t1));

        PHASE_BEGIN();
        this->allocateTreePropMemory(this->localTree);
        PHASE_END("allocTreeProp", domain);

        // Phase 5 ticket 03 (PLAN.md): tree-rebuild granularity, matching Gadget-2's own
        // set_softenings() call site (gravtree.c:51 -- before each new force-tree build, not
        // every timestep); deliberately inside this branch only, not the no-rebuild `else` below.
        //
        // T28/C-A-04 ORDERING: this MUST run before compute_properties(). The per-node softening
        // reduction lives in compute_properties, and its input is tree.bodies_forceSoftening, which
        // is written here -- from tree.bodies_type, itself rebuilt here from the freshly-resorted
        // tree.bodies_ids. Called after compute_properties (as it was), the node reduction would
        // read the PREVIOUS rebuild's softenings in the PREVIOUS particle order: a silent
        // wrong-particle read of exactly the kind C-B-13/C-C-18/C-D-08 record.
        // Safe to hoist: recomputeSoftening depends only on bodies_ids (resorted by sort_bodies,
        // which ran above) and get_t_current(). It reads nothing compute_properties produces.
        PHASE_BEGIN();
        this->recomputeSoftening(this->localTree);
        PHASE_END("recomputeSoftening", timeline);

        STAGE_TRACE("[BUG4-STAGE] before compute_properties iter=%d\n", iter);
        // C-A-01(b) PERF (GADGET_HIP_PROPS_TIME=1): wall time of the tree-properties phase, which
        // is where the fixed-cell-centre block lives. Syncs the stream on BOTH sides -- the kernels
        // are launched asynchronously, so without the trailing sync this would time the launch, not
        // the work. The sync itself is why this is env-gated and off by default.
        static const bool propsTimeOn = (getenv("GADGET_HIP_PROPS_TIME") != NULL);
        double propsT0 = 0.0;
        if (propsTimeOn) { execStream->sync(); propsT0 = get_time(); }
        this->compute_properties(this->localTree);
        if (propsTimeOn)
        {
          execStream->sync();
          fprintf(stderr, "[PROPS-TIME] iter=%d nodes=%d compute_properties=%.6f s\n",
                  iter, this->localTree.n_nodes, get_time() - propsT0);
        }
        STAGE_TRACE("[BUG4-STAGE] after compute_properties iter=%d\n", iter);

        // T28/C-A-04 probe (GADGET_HIP_T28_NODESOFT=1). The predecessor of this field was
        // identically 0.0f for every node in every run (it read body_vel[].w, which no IC path
        // writes a softening into), so "it is populated at all" is the first thing to establish.
        // For a single-species run every node must be NON-mixed with |value| == 2.8*eps.
        if (getenv("GADGET_HIP_T28_NODESOFT") && iter <= 1)
        {
          this->localTree.nodeSoftInfo.d2h();
          const int nn = this->localTree.n_nodes;
          long nz = 0, nmix = 0; float lo = 3.4e38f, hi = -1.0f;
          for (int i = 0; i < nn; i++)
          {
            const float v = this->localTree.nodeSoftInfo[i];
            if (v == 0.0f) { nz++; continue; }
            if (v < 0.0f) nmix++;
            const float a = fabsf(v);
            if (a < lo) lo = a; if (a > hi) hi = a;
          }
          fprintf(stderr, "[T28] iter=%d nodes=%d  zero(no-contributor)=%ld  mixed=%ld  "
                          "|maxSoft| range=[%.9g, %.9g]\n", iter, nn, nz, nmix, lo, hi);
        }

#ifdef GADGET_HIP_HIGHRES
        // Phase 5 ticket 06 (PLAN.md): initial region build (first rebuild) + reactive
        // recompute-on-out-of-range check, same tree-rebuild granularity as recomputeSoftening()
        // above -- matches Gadget-2's own reactive-only recomputation cadence
        // (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.3: not every step, only on demand). Deliberately runs
        // BEFORE recomputeTimestepGlobals() below (Ticket 07, PLAN.md): that function now reads
        // gadgetZoomRegion.asmth1 for the displacement-constraint's own Asmth[1] refinement, which
        // must already be valid (not the default-constructed 0.0f) the very first time it's used --
        // ordering them the other way around would silently feed a zero Asmth1 into the timestep
        // formula on the very first rebuild, forcing dt_displacement to 0.
        if (haveGadgetZoom)
        {
          float pmBoxSize = 0.0f; int pmGridSize = 0;
          if (pm_get_dump_context(pmBoxSize, pmGridSize))
          {
#ifdef PERIODIC
            const double asmth0 = 1.25 * pmBoxSize / pmGridSize;
            const float  boxSizeForBoundsCheck = pmBoxSize;
#else
            const double asmth0 = 1.25 * pmBoxSize / (2.0 * pmGridSize); // pmBoxSize holds meshSize here
            const float  boxSizeForBoundsCheck = 0.0f; // no periodic box to cross
#endif
            const double rcut0 = 4.5 * asmth0;

            // Dedicated timer, NOT PHASE_BEGIN: see the zoomFineGridPM pair below for why.
            double zrsT = 0.0;
            if (phaseTimeOn) { execStream->sync(); gravStream->sync(); zrsT = get_time(); }
            if (!gadgetZoomSolverReady)
            {
              this->recomputeZoomRegion(this->localTree, asmth0, rcut0, pmGridSize,
                                         boxSizeForBoundsCheck);
            }
            else
            {
              // T50 section 3A: on the DEVICE. This used to d2h every position (1.84 GB at 115M),
              // build a 1.38 GB std::vector<Vec3f>, and scan it single-threaded, every rebuild step
              // -- 0.58 s/step steady, 3.74 s on the first -- to answer a masked bounds reduction.
              // recomputeZoomRegion() below still does the host copy, but it needs the actual
              // min/max to size the new region and it only runs when the answer is yes.
              // T50 item 3F: `corner` is inset BELOW the data span, so the containment window
              // starts at corner + cornerInset. Passing `corner` here would shift the test by two
              // fine cells and no longer match pm_zoom_region_out_of_range()'s host version.
              const float3 zCorner = make_float3(
                  gadgetZoomRegion.corner[0] + gadgetZoomRegion.cornerInset,
                  gadgetZoomRegion.corner[1] + gadgetZoomRegion.cornerInset,
                  gadgetZoomRegion.corner[2] + gadgetZoomRegion.cornerInset);
              if (pm_zoom_any_out_of_range(this->localTree.bodies_Ppos.raw_p(),
                                            this->localTree.bodies_typeDevice.raw_p(),
                                            gadgetZoomMask, zCorner,
                                            gadgetZoomRegion.totalMeshSize,
                                            this->localTree.n, gravStream->s()))
              {
                fprintf(stderr, "[ZOOM] high-res particle left the current region -- recomputing.\n");
                this->recomputeZoomRegion(this->localTree, asmth0, rcut0, pmGridSize,
                                           boxSizeForBoundsCheck);
              }
            }
            if (phaseTimeOn)
            {
              execStream->sync(); gravStream->sync();
              const double dt_ = get_time() - zrsT;
              s_cpuAcc.timeline += dt_;
              if (phaseTimePrint)
                fprintf(stderr, "[PHASE] iter=%d %-22s %8.4f s\n", iter, "zoomRegionScan", dt_);
            }
          }
        }
#endif

        // Phase 5 ticket 04 (PLAN.md): same tree-rebuild granularity as recomputeSoftening()
        // above -- a real d2h sync, not something to do every timestep.
        PHASE_BEGIN();
        this->recomputeTimestepGlobals(this->localTree);
        PHASE_END("recomputeTimestepGlobals", timeline);

        #ifdef DO_BLOCK_TIMESTEP
                devContext->startTiming(execStream->s());
                setActiveGrpsFunc(this->localTree);
                devContext->stopTiming("setActiveGrpsFunc", 10, execStream->s());
                idata.Nact_since_last_tree_rebuild = 0;
        #endif

        idata.lastBuildTime   = get_time() - t1;
        idata.totalBuildTime += idata.lastBuildTime;
        // C-A-07: seed the extent floor from the freshly built geometry. Monotonicity is enforced
        // only BETWEEN rebuilds -- a rebuild is entitled to shrink a node, because its index ranges
        // are no longer stale.
        //
        // POSITION MATTERS TWICE OVER. It must be at the rebuild rather than lazily on the first
        // stale step, and it must be AFTER compute_properties(), which is what actually writes
        // boxSizeInfo. Seeded before that call it captures the PREVIOUS iteration's geometry and
        // enforces nothing: measured, 42.73% of nodes still shrank on the first stale step with
        // worst_shrink identical to the unmaintained run. Seeded on the first stale step instead,
        // one full unconstrained refit is baked in -- the same 42.73% -- and the floor locks it in. Seeding on the first
        // stale step bakes in one full step of unconstrained refit, and that is the step that
        // matters: measured at 64^3, 42.7% of nodes shrink in it, the worst to 0.09% of its rebuild
        // extent, and the floor then locks that in. Monotonicity has to start from the geometry a
        // rebuild produced -- the only geometry whose index ranges are not stale.
        //
        // Allocated only when some step will actually skip a rebuild, so the default configuration
        // (rebuild every step) never pays the ~600 MB this costs at 512^3.
        if (ca07MonotoneWanted() && rebuild_tree_rate > 1)
        {
          const size_t cap = this->localTree.boxSizeInfo.get_size();
          if (s_nodeLenFloorNodes < (int) cap)
          {
            if (s_nodeLenFloor) hipFree(s_nodeLenFloor);
            CU_SAFE_CALL(hipMalloc((void **) &s_nodeLenFloor, cap * sizeof(float4)));
            s_nodeLenFloorNodes = (int) cap;
            fprintf(stderr, "[C-A-07] node-extent floor ACTIVE (GADGET force_update_len "
                            "semantics): %.1f MB for %d nodes\n",
                    cap * sizeof(float4) / (1024.0*1024.0), s_nodeLenFloorNodes);
          }
          CU_SAFE_CALL(hipMemcpyAsync(s_nodeLenFloor, this->localTree.boxSizeInfo.raw_p(),
                                      (size_t) this->localTree.n_nodes * sizeof(float4),
                                      hipMemcpyDeviceToDevice, execStream->s()));
        }

        ca07_probe(this->localTree, true, iter);
        invariant_probe(this->localTree, iter);
      }
      else
      {
        #ifdef DO_BLOCK_TIMESTEP
          devContext->startTiming(execStream->s());
          setActiveGrpsFunc(this->localTree);
          devContext->stopTiming("setActiveGrpsFunc", 10, execStream->s());
          idata.Nact_since_last_tree_rebuild = 0;
        #endif
        //Don't rebuild only update the current boxes
        this->compute_properties(this->localTree);

        // C-A-07 / T20 / Phase 4. compute_properties has just refitted every node's AABB EXACTLY
        // from the current positions over the STALE node_bodies ranges, which can shrink a node --
        // 8.5% of them on the first stale step, the worst to 8% of its rebuild extent. GADGET-2
        // never shrinks a node between rebuilds (force_update_len, forcetree.c:940-960), and that
        // monotonicity is what keeps the opening criterion conservative; a too-small box lets the
        // walk approximate where it must split, which is the T24 blowup mechanism.
        //
        // GADGET_HIP_CA07_MONOTONE=0 restores the unmaintained behaviour, for A/B only. Same idiom
        // as GADGET_HIP_T4_FIX_G / T45_KEEP_ACC0_D2H / T47_NO_G.
        {
          const int nn = this->localTree.n_nodes;
          if (ca07MonotoneWanted() && s_nodeLenFloor && nn > 0 && s_nodeLenFloorNodes >= nn)
          {
            gadgetNodeLenMonotone.set_args(0, &this->localTree.n_nodes,
                                          this->localTree.boxSizeInfo.p(),
                                          (void *) &s_nodeLenFloor);
            gadgetNodeLenMonotone.setWork(nn, 128);
            gadgetNodeLenMonotone.execute2(execStream->s());
          }
          else if (!ca07MonotoneWanted())
          {
            static bool noted = false;
            if (!noted)
            {
              noted = true;
              fprintf(stderr, "[C-A-07] node boxes are recomputed exactly on non-rebuild steps "
                              "(parent = union of children). GADGET instead grows them "
                              "monotonically; GADGET_HIP_CA07_MONOTONE=1 matches that, at about "
                              "+0.9%% wall clock and no measured accuracy change.\n");
            }
          }
        }
        ca07_probe(this->localTree, false, iter);

      }//end rebuild tree

      //Approximate gravity
      t1 = get_time();
      //devContext.startTiming(gravStream->s());
      STAGE_TRACE("[BUG4-STAGE] before approximate_gravity iter=%d\n", iter);

      // Transverse-force bug trace (user request 2026-08-30): see gravkernel's own g_trace*
      // globals doc comment (dev_approximate_gravity_warp_new.cu). traceY_reset()/traceY_read()
      // are plain host wrapper functions defined in that same .cu file (hipMemcpyToSymbol needs
      // same-translation-unit visibility of the __device__ symbols, per pm.h's own convention
      // note -- gpu_iterate.cpp can't touch them directly). Zero-cost when GADGET_HIP_TRACE_Y unset.
      const bool traceYEnabled = getenv("GADGET_HIP_TRACE_Y") != nullptr;
      if (traceYEnabled) traceY_reset();

      // T4 (contract C-B-02): census of WHICH test opens each node in the Springel MAC.
      // Window-gated (default iters 1-3; iter 0 never reaches split_node_grav_springel at all,
      // since groupMaxAcc==0 routes it to the geometric bootstrap per T23's fix) because the
      // counters are plain per-lane atomics -- see t4_tally. GADGET_HIP_T4_COUNT=1 uses the
      // default window; GADGET_HIP_T4_COUNT=lo,hi overrides it.
      static bool t4_init = false;
      static int  t4_lo = -1, t4_hi = -1;
      if (!t4_init)
      {
        t4_init = true;
        const char *e = getenv("GADGET_HIP_T4_COUNT");
        if (e && sscanf(e, "%d,%d", &t4_lo, &t4_hi) != 2) { t4_lo = 1; t4_hi = 3; }
      }
      const bool t4On = (t4_lo >= 0 && iter >= t4_lo && iter <= t4_hi);
      if (t4On) { t4_counters_reset(); t4_counters_set(1); }

      PHASE_BEGIN();
      approximate_gravity(this->localTree);
      PHASE_END("approximate_gravity", treewalk);
      STAGE_TRACE("[BUG4-STAGE] after approximate_gravity iter=%d\n", iter);

      if (t4On)
      {
        t4_counters_set(0);
        unsigned long long p = 0, x = 0, b = 0, r = 0;
        t4_counters_read(p, x, b, r);
        const unsigned long long opens = p + x + b;
        const unsigned long long total = opens + r;
        fprintf(stderr, "[T4-MAC-CENSUS] iter=%d primary=%llu prox=%llu bJzero=%llu reject=%llu "
                        "opens=%llu total=%llu primary_frac_of_opens=%.9f\n",
                iter, p, x, b, r, opens, total,
                opens ? (double) p / (double) opens : 0.0);
      }

      if (traceYEnabled)
      {
        double approxY = 0.0, directY = 0.0;
        int approxN = 0, directN = 0;
        traceY_read(approxY, directY, approxN, directN);
        fprintf(stderr, "[TRACE-Y] iter=%d approxY=%.9g (n=%d, avg=%.9g)  directY=%.9g (n=%d, avg=%.9g)  total=%.9g\n",
                iter, approxY, approxN, approxN ? approxY/approxN : 0.0,
                directY, directN, directN ? directY/directN : 0.0, approxY + directY);
        const char *traceDumpPath = getenv("GADGET_HIP_TRACE_Y_DUMP");
        if (traceDumpPath) traceY_dump(traceDumpPath);

        // Transverse-force bug trace, group-boundary check (user request 2026-08-30): the near-
        // field neighbor set found asymmetric around the traced target (extends one extra grid
        // row in +y with no -y counterpart) might be explained by the GOTHIC-style group-shared
        // MAC simply not being centered on this specific target (a documented, accepted trade-off
        // of sharing one MAC decision across NCRIT=32 targets) rather than a genuine bug -- if the
        // group's own bounding box (groupCenterInfo/groupSizeInfo) is offset from the target in a
        // way that matches the observed asymmetry AND is wide enough that the MAC's own
        // conservatism should have caught the missing -y row anyway, that points to a real bug in
        // groupSize itself rather than an accepted architectural cost.
        const char *traceGroupPath = getenv("GADGET_HIP_TRACE_Y_GROUP");
        if (traceGroupPath)
        {
          this->localTree.bodies_ids.d2h();
          this->localTree.body2group_list.d2h();
          this->localTree.groupCenterInfo.d2h();
          this->localTree.groupSizeInfo.d2h();
          FILE *gf = fopen(traceGroupPath, "w");
          if (gf)
          {
            for (int i = 0; i < this->localTree.n; i++)
            {
              if ((unsigned long long) this->localTree.bodies_ids[i] == 1ULL)
              {
                const unsigned grpID = this->localTree.body2group_list[i];
                const real4 gc = this->localTree.groupCenterInfo[grpID];
                const real4 gs = this->localTree.groupSizeInfo[grpID];
                fprintf(gf, "target_idx=%d grpID=%u groupCenter=(%.9g,%.9g,%.9g) groupSize=(%.9g,%.9g,%.9g)\n",
                        i, grpID, gc.x, gc.y, gc.z, gs.x, gs.y, gs.z);
              }
            }
            fclose(gf);
          }
        }
      }

      // Group-corner statistics (user request 2026-08-30 follow-up, PLAN.md's Close-encounter
      // root-cause audit): quantifies what fraction of particles sit near the "corner" of their
      // own group's shared-MAC bounding box, across ALL particles rather than the single hardcoded
      // target above. Purely geometric -- body2group_list/groupCenterInfo/groupSizeInfo/bodies_pos
      // exist unconditionally in every build, no PMGRID/PERIODIC involved -- so this doubles as a
      // periodicity-independence check: if the corner-score distribution looks the same on a plain
      // non-periodic, non-PMGRID build, that's direct evidence periodicity isn't a driver of the
      // bug, not just an architectural argument from code inspection. score=1.0 reproduces
      // PLAN.md's exact zero-margin condition (sitting on the group box boundary on every axis at
      // once); score=0.0 is sitting exactly at the group's own center.
      const char *cornerStatsPath = getenv("GADGET_HIP_GROUP_CORNER_STATS");
      // Per-particle (id, corner-score, tree-force xyz) dump (2026-08-30 follow-up): the single
      // hardcoded GADGET_HIP_TRACE_Y target no longer shows the large spurious y-force it once did
      // (re-checked same session: ~1.4e-7, noise-level) even though the REAL aggregate diagnostic
      // (comparing real snapshot y/z-velocity growth against stock Gadget-2 on the same 32^3
      // Zel'dovich IC) still shows a genuine, reproducible ~87x excess by snapshot 9 -- so whatever
      // drives the aggregate effect is not concentrated in that one historically-traced particle.
      // This dumps every particle's own corner-score alongside its own real tree force so that can
      // be checked directly (population-wide correlation) instead of assumed from one example.
      const char *cornerDumpPath = getenv("GADGET_HIP_GROUP_CORNER_DUMP");
      if (cornerStatsPath || cornerDumpPath)
      {
        this->localTree.bodies_Ppos.d2h();
        this->localTree.bodies_ids.d2h();
        this->localTree.body2group_list.d2h();
        this->localTree.groupCenterInfo.d2h();
        this->localTree.groupSizeInfo.d2h();
        if (cornerDumpPath) this->localTree.bodies_acc1.d2h();

        const int nBins = 10;
        std::vector<long> hist(nBins, 0);
        double sumScore = 0.0, maxScore = 0.0;
        int maxScoreIdx = -1;
        long nCorner90 = 0, nCorner95 = 0, nCorner99 = 0;

        for (int i = 0; i < this->localTree.n; i++)
        {
          const unsigned grpID = this->localTree.body2group_list[i];
          const real4 gc = this->localTree.groupCenterInfo[grpID];
          const real4 gs = this->localTree.groupSizeInfo[grpID];
          const real4 p  = this->localTree.bodies_Ppos[i];

          const double fx = gs.x > 0 ? fabs(p.x - gc.x) / gs.x : 0.0;
          const double fy = gs.y > 0 ? fabs(p.y - gc.y) / gs.y : 0.0;
          const double fz = gs.z > 0 ? fabs(p.z - gc.z) / gs.z : 0.0;
          const double score = std::min(fx, std::min(fy, fz));

          sumScore += score;
          if (score > maxScore) { maxScore = score; maxScoreIdx = i; }
          if (score >= 0.90) nCorner90++;
          if (score >= 0.95) nCorner95++;
          if (score >= 0.99) nCorner99++;

          int bin = (int)(score * nBins);
          if (bin >= nBins) bin = nBins - 1;
          if (bin < 0) bin = 0;
          hist[bin]++;

          if ((unsigned long long) this->localTree.bodies_ids[i] == 1ULL)
            fprintf(stderr, "[CORNER-DEBUG-ID1] iter=%d i=%d grpID=%u p=(%.6g,%.6g,%.6g) gc=(%.6g,%.6g,%.6g) gs=(%.6g,%.6g,%.6g) score=%.6g\n",
                    iter, i, grpID, p.x, p.y, p.z, gc.x, gc.y, gc.z, gs.x, gs.y, gs.z, score);
        }

        if (cornerDumpPath)
        {
          char pathIter[512];
          snprintf(pathIter, sizeof(pathIter), "%s.iter%d", cornerDumpPath, iter);
          FILE *cdf = fopen(pathIter, "w");
          if (cdf)
          {
            for (int i = 0; i < this->localTree.n; i++)
            {
              const unsigned grpID = this->localTree.body2group_list[i];
              const real4 gc = this->localTree.groupCenterInfo[grpID];
              const real4 gs = this->localTree.groupSizeInfo[grpID];
              const real4 p  = this->localTree.bodies_Ppos[i];
              const double fx = gs.x > 0 ? fabs(p.x - gc.x) / gs.x : 0.0;
              const double fy = gs.y > 0 ? fabs(p.y - gc.y) / gs.y : 0.0;
              const double fz = gs.z > 0 ? fabs(p.z - gc.z) / gs.z : 0.0;
              const double score = std::min(fx, std::min(fy, fz));
              const real4 a1 = this->localTree.bodies_acc1[i];
              fprintf(cdf, "%llu %u %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
                      (unsigned long long) this->localTree.bodies_ids[i], grpID, score,
                      p.x, p.y, p.z, a1.x, a1.y, a1.z, a1.w,
                      gc.x, gc.y, gc.z, gs.x, gs.y, gs.z);
            }
            fclose(cdf);
          }
        }
        if (maxScoreIdx >= 0)
        {
          const unsigned grpID = this->localTree.body2group_list[maxScoreIdx];
          const real4 gc = this->localTree.groupCenterInfo[grpID];
          const real4 gs = this->localTree.groupSizeInfo[grpID];
          const real4 p  = this->localTree.bodies_Ppos[maxScoreIdx];
          fprintf(stderr, "[CORNER-DEBUG-MAX] iter=%d i=%d grpID=%u p=(%.6g,%.6g,%.6g) gc=(%.6g,%.6g,%.6g) gs=(%.6g,%.6g,%.6g) score=%.6g\n",
                  iter, maxScoreIdx, grpID, p.x, p.y, p.z, gc.x, gc.y, gc.z, gs.x, gs.y, gs.z, maxScore);
        }

        FILE *cf = fopen(cornerStatsPath, iter == 0 ? "w" : "a");
        if (cf)
        {
          fprintf(cf, "iter=%d n=%d mean=%.6f max=%.6f n_ge_90=%ld(%.4f%%) n_ge_95=%ld(%.4f%%) n_ge_99=%ld(%.4f%%) hist=[",
                  iter, this->localTree.n, sumScore / this->localTree.n, maxScore,
                  nCorner90, 100.0 * nCorner90 / this->localTree.n,
                  nCorner95, 100.0 * nCorner95 / this->localTree.n,
                  nCorner99, 100.0 * nCorner99 / this->localTree.n);
          for (int b = 0; b < nBins; b++) fprintf(cf, "%ld%s", hist[b], b + 1 < nBins ? "," : "");
          fprintf(cf, "]\n");
          fclose(cf);
        }
      }
//      devContext.stopTiming("Approximation", 4, gravStream->s());

      runningLETTimeSum = 0;

      if(nProcs > 1) makeLET();
    }//else if useDirectGravity

    gravStream->sync(); //Syncs the gravity stream, including any gravity computations due to LET actions

#if defined(PMGRID) && defined(PERIODIC)
    // Phase 4 (PLAN.md): the real per-step PM long-range force, added on top of the tree's
    // already-computed short-range force in bodies_acc1 -- previously PM only ran inside one-shot
    // diagnostics (--pm-test-*, GADGET_HIP_DUMP_ACC), never in an actual simulation step. Only
    // for the tree-walk path (not --direct, which is a validation-only N^2 mode never meant to
    // run a real periodic box) and only for nProcs==1 -- this port's PM solver is deliberately
    // single-GPU with no MPI slab decomposition (LOG.md §18/PLAN.md Phase 4 architecture note),
    // so silently running it under multiple ranks would double/miscount the density grid rather
    // than erroring loudly; refuse instead of guessing. context comes from
    // pm_set_dump_context() (main.cpp, set once right after IC load from the same
    // --boxsize/PMGRID the tree's own periodic wrap already uses) so this doesn't invent a
    // second, possibly-inconsistent source of truth for boxSize/PMGRID.
    if (!useDirectGravity)
    {
      PHASE_BEGIN();
      float pmBoxSize; int pmGridSize;
      if (pm_get_dump_context(pmBoxSize, pmGridSize))
      {
        static bool s_pmWarnedMultiProc = false;
        if (nProcs > 1)
        {
          if (!s_pmWarnedMultiProc)
          {
            fprintf(stderr, "[PM] WARNING: nProcs=%d > 1 -- this port's PM solver is single-GPU "
                             "only (no MPI slab decomposition), skipping PM long-range force "
                             "every step rather than silently computing it wrong. Long-range "
                             "gravity will be MISSING from this multi-process run.\n", nProcs);
            s_pmWarnedMultiProc = true;
          }
        }
        else
        {
          static PMPeriodicSolver s_pmSolver;
          static bool  s_pmSolverInit = false;
          static int   s_pmSolverN = 0;
          static float *s_d_pmfx = nullptr, *s_d_pmfy = nullptr, *s_d_pmfz = nullptr, *s_d_pmpot = nullptr;

          if (!s_pmSolverInit)
          {
            pm_fft_init();
            s_pmSolver = pm_solver_create_periodic(pmGridSize, pmBoxSize);
            s_pmSolverInit = true;
          }
          if (s_pmSolverN != localTree.n)
          {
            if (s_d_pmfx) { hipFree(s_d_pmfx); hipFree(s_d_pmfy); hipFree(s_d_pmfz); hipFree(s_d_pmpot); }
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmfx, localTree.n * sizeof(float)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmfy, localTree.n * sizeof(float)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmfz, localTree.n * sizeof(float)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmpot, localTree.n * sizeof(float)));
            s_pmSolverN = localTree.n;
          }

          // bodies_Ppos: the same predicted positions approximate_gravity() itself just used,
          // matching the tree walk's own input exactly rather than a second, possibly-stale copy.
          // Also requests PM's own potential (LOG.md §30) -- previously only the force was added
          // here (LOG.md §29), leaving bodies_acc1.w (and therefore the run's own Etot/energy-
          // drift diagnostic) short-range-only even though the dynamics were already using the
          // correct total force.
          // T36: PM CADENCE. Gadget-2 recomputes the long-range force only on PM steps and applies
          // the stored P[i].GravPM on every intervening step (accel.c). The PM step is the largest
          // power-of-two subdivision of the timeline not exceeding dt_displacement -- exactly the
          // ladder compute_dt() already uses for particle timesteps (timestep.c:189-193 on
          // TIMEBASE). With MaxSizeTimestep=0.03 and a log-timeline span of 4.605 that lands at
          // ~0.018 in dloga, while the global sync step late in the run is ~1.4e-4: PM runs about
          // once per hundred steps rather than every one.
          //
          // The stored force is kept in ID space (pm_scatter_by_id) because these arrays are
          // indexed by current sort order and the tree re-sorts every step -- see pm_cic.cu.
          static double s_pmNextTime = -1.0;
          static int    s_pmStoredN  = 0;
          // Phase 3: these ARE GravPM. Bound by reference to the file-scope buffers so correct()
          // can reach them; the code below is unchanged.
          GFreeAcc *&s_d_pmIdX = s_gravPMx; GFreeAcc *&s_d_pmIdY = s_gravPMy;
          GFreeAcc *&s_d_pmIdZ = s_gravPMz; float *&s_d_pmIdPot = s_gravPMpot;
          // T36 IS OFF BY DEFAULT -- it cost accuracy, measured. See the block below and
          // tickets/T35-sparse-step-cost.md "Round 3 RETRACTED".
          // GADGET_HIP_PM_CADENCE=1 opts back in for experiments; PM_EVERY_STEP is kept as the
          // explicit spelling of the default.
          static const bool pmCadenceOn = (getenv("GADGET_HIP_PM_CADENCE") != NULL) &&
                                          (atoi(getenv("GADGET_HIP_PM_CADENCE")) != 0);
          static const bool pmEveryStep = !pmCadenceOn ||
                                          (getenv("GADGET_HIP_PM_EVERY_STEP") != NULL);

          if (s_pmStoredN != localTree.n)
          {
            if (s_d_pmIdX) { hipFree(s_d_pmIdX); hipFree(s_d_pmIdY); hipFree(s_d_pmIdZ); hipFree(s_d_pmIdPot); }
            // sizeof(GFreeAcc) == sizeof(float): a single-float struct, so the allocation, the
            // scatter and every memcpy below are byte-for-byte what they were.
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmIdX,   localTree.n * sizeof(GFreeAcc)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmIdY,   localTree.n * sizeof(GFreeAcc)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmIdZ,   localTree.n * sizeof(GFreeAcc)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmIdPot, localTree.n * sizeof(float)));
            s_pmStoredN = localTree.n;
            s_pmNextTime = -1.0;      // force a recompute after any resize
          }

          // Dense ids are what makes id-space storage valid; without them fall back to recomputing
          // every step, which is what the port did before and is always correct.
          // GADGET_HIP_PM_NO_IDSPACE=1 bypasses the id-space store entirely and uses the original
          // per-index add. The id-space path only EXISTS to let the force survive between PM steps;
          // with PM recomputed every step (the default since T36 was retracted) it buys nothing and
          // costs a 134M-element scatter over 2.1 GB every step. Kept as a switch rather than
          // deleted because the cadence remains available behind GADGET_HIP_PM_CADENCE.
          static const bool pmNoIdSpace = (getenv("GADGET_HIP_PM_NO_IDSPACE") != NULL);
          const bool pmIdSpaceOk = !pmNoIdSpace && gadgetTypeByIdOk &&
                                   (gadgetTypeByIdMax == (ullong)(localTree.n - 1));
          // WHY THIS IS OFF BY DEFAULT.
          //
          // Skipping the PM solve between PM steps looks like it just reproduces Gadget, which
          // also keeps the long-range force fixed between them. It does not, because Gadget applies
          // that force DIFFERENTLY: a separate long-range kick over the whole PM interval, applied
          // to EVERY particle exactly once (`advance_and_find_timesteps`, the
          // `All.PM_Ti_endstep == All.Ti_Current` block). This port instead folds the stored force
          // into bodies_acc1 for ACTIVE particles and lets each particle's own kick integrate it
          // over its own timestep. The two agree in the time-integral to first order -- which is
          // exactly why a short test could not tell them apart -- but they are not the same
          // operator, and the difference accumulates.
          //
          // Measured: a full 512^3 run to z=0 with the cadence on drifted from -2.0% in Ekin at
          // a=0.06 to -10.9% at a=0.96 against the CPU reference, where PM-every-step gives
          // +0.06%. A 60-step bisect isolated it to this switch alone (the two T35 device kernels
          // were bit-exact over the same interval).
          //
          // Getting the 0.73 s/step back means implementing Gadget's actual structure -- a separate
          // long-range kick over the PM interval for all particles -- not skipping the solve.
          //
          // The cadence also depends ONLY on time, never on whether this step samples the energies.
          // `energyDue` is driven by TimeBetStatistics, a DIAGNOSTIC parameter, and I wired it in
          // at first: that was wrong, because it makes the trajectory depend on how often you ask
          // for statistics. Gadget keeps them separate too -- its statistics call
          // compute_potential() for a fresh pass that never touches the GravPM used for dynamics.
          // Phase 3: the PM step is chosen on the integer timeline exactly as GADGET-2 chooses it
          // (timestep.c:345-357) -- the largest power-of-two tick count not exceeding
          // dt_displacement -- instead of the float wall-clock comparison the retracted T36 used.
          // The cadence needs the id-space store to survive the re-sort between PM steps, so it
          // cannot be active without it.
          s_pmCadenceActive = !pmEveryStep && pmIdSpaceOk && haveGadgetComovingTables &&
                              s_gadgetTablesOk;
          // ">=", not "==". GADGET-2 tests All.PM_Ti_endstep == All.Ti_Current and is entitled
          // to, because its run always begins at Ti_Current == 0 and every sync point lands on
          // the tick lattice. This port can begin mid-timeline -- restarting from a snapshot
          // whose stored time differs from TimeBegin in the last few digits puts Ti_current at
          // some nonzero tick on step 0 -- and then exact equality is a DEADLOCK: PM_Ti_endstep
          // starts at 0 and is only advanced inside the branch the equality guards, while
          // Ti_current only grows, so the condition can never become true and the long-range
          // force is never computed or applied at all. Measured: restarting at a=0.68935 from
          // snapshot_006, the cadence silently did nothing for the whole run.
          const bool pmDue = !s_pmCadenceActive || (Ti_current >= s_PM_Ti_endstep);

          if (pmDue)
          {
            pm_compute_forces_periodic(s_pmSolver, localTree.bodies_Ppos.raw_p(), localTree.n,
                                        s_d_pmfx, s_d_pmfy, s_d_pmfz, gravStream->s(),
                                        /*gravityConstant=*/1.0f, s_d_pmpot);
#ifdef GADGET_HIP_HIGHRES
            // T58: the zoom (PLACEHIGHRESREGION) fine mesh, solved on the SAME cadence as the
            // coarse one and accumulated into the SAME arrays -- Gadget's long_range_force()
            // structure (longrange.c:54-82): zero GravPM, `+=` the periodic mesh, `+=` the
            // isolated high-res mesh, and let every consumer read the sum.
            //
            // The fine mesh's kernel is the DIFFERENTIAL Green's function
            // (erfc(u*asmthRatio) - erfc(u), pm_zoom.cu), not a second copy of the full potential,
            // so this is pure superposition rather than double counting. The accumulate is masked
            // by particle type because the force READOUT is type-filtered in Gadget too
            // (pm_nonperiodic.c:930-932) -- the DEPOSIT is not, in either code.
            if (haveGadgetZoom && gadgetZoomSolverReady)
            {
              double zpmT = 0.0;
              if (phaseTimeOn) { execStream->sync(); gravStream->sync(); zpmT = get_time(); }
              static float *s_d_zoomfx = nullptr, *s_d_zoomfy = nullptr, *s_d_zoomfz = nullptr,
                           *s_d_zoompot = nullptr;
              static int    s_zoomSolverN = 0;
              if (s_zoomSolverN != localTree.n)
              {
                if (s_d_zoomfx) { hipFree(s_d_zoomfx); hipFree(s_d_zoomfy); hipFree(s_d_zoomfz);
                                  hipFree(s_d_zoompot); }
                CU_SAFE_CALL(hipMalloc((void**)&s_d_zoomfx,  localTree.n * sizeof(float)));
                CU_SAFE_CALL(hipMalloc((void**)&s_d_zoomfy,  localTree.n * sizeof(float)));
                CU_SAFE_CALL(hipMalloc((void**)&s_d_zoomfz,  localTree.n * sizeof(float)));
                CU_SAFE_CALL(hipMalloc((void**)&s_d_zoompot, localTree.n * sizeof(float)));
                s_zoomSolverN = localTree.n;
              }
              // Same bodies_Ppos the coarse solve just used -- the walk's own predicted positions.
              pm_compute_forces_finegrid(gadgetZoomSolver, gadgetZoomRegion,
                                          localTree.bodies_Ppos.raw_p(), localTree.n,
                                          s_d_zoomfx, s_d_zoomfy, s_d_zoomfz, gravStream->s(),
                                          /*gravityConstant=*/1.0f, s_d_zoompot);
              // T54's per-component dump (GADGET_HIP_DUMP_SPLIT), rewritten by T58 as one
              // self-contained block. It has to sit HERE, before the accumulate below: this is the
              // only point in the step where all three components exist separately --
              // bodies_acc1 is tree-only (the walk wrote it, no PM has been added yet),
              // s_d_pmf{x,y,z} is coarse-only, and s_d_zoomf{x,y,z} is the fine mesh. One line
              // later the accumulate makes the last two indistinguishable, which is the whole point
              // of T58 and is exactly what the dump must be read before.
              //
              // It used to be two halves -- a capture in the coarse block and a writer in the
              // old standalone zoom block -- bridged by file-scope statics. Moving the fine mesh
              // in here deleted the writer and left the capture running, so the dump would have
              // silently produced no file at all. A diagnostic that quietly stops working is worse
              // than one that was never written.
              {
                const char *dumpSplit = getenv("GADGET_HIP_DUMP_SPLIT");
                const int   splitIter = getenv("GADGET_HIP_DUMP_ACC_ITER")
                                      ? atoi(getenv("GADGET_HIP_DUMP_ACC_ITER")) : 0;
                if (dumpSplit && iter == splitIter)
                {
                  gravStream->sync(); execStream->sync();
                  const int n = localTree.n;
                  std::vector<float> cx(n), cy(n), cz(n), fx(n), fy(n), fz(n);
                  CU_SAFE_CALL(hipMemcpy(cx.data(), s_d_pmfx,   (size_t)n*sizeof(float), hipMemcpyDeviceToHost));
                  CU_SAFE_CALL(hipMemcpy(cy.data(), s_d_pmfy,   (size_t)n*sizeof(float), hipMemcpyDeviceToHost));
                  CU_SAFE_CALL(hipMemcpy(cz.data(), s_d_pmfz,   (size_t)n*sizeof(float), hipMemcpyDeviceToHost));
                  CU_SAFE_CALL(hipMemcpy(fx.data(), s_d_zoomfx, (size_t)n*sizeof(float), hipMemcpyDeviceToHost));
                  CU_SAFE_CALL(hipMemcpy(fy.data(), s_d_zoomfy, (size_t)n*sizeof(float), hipMemcpyDeviceToHost));
                  CU_SAFE_CALL(hipMemcpy(fz.data(), s_d_zoomfz, (size_t)n*sizeof(float), hipMemcpyDeviceToHost));
                  localTree.bodies_acc1.d2h();
                  localTree.bodies_Ppos.d2h();
                  localTree.bodies_ids.d2h();
                  FILE *f = fopen(dumpSplit, "w");
                  if (f)
                  {
                    const double Gv = (haveGadgetParams && gadgetParams.G > 0.0) ? gadgetParams.G : 1.0;
                    fprintf(f, "# gadget-hip force split  iter=%d  G=%.17g  zoomMask=0x%x  "
                               "asmth1=%.17g rcut1=%.17g corner=%.17g,%.17g,%.17g inset=%.17g "
                               "totalMeshSize=%.17g meshSize=%.17g  haveCoarse=1\n",
                            iter, Gv, gadgetZoomMask,
                            (double) gadgetZoomRegion.asmth1, (double) gadgetZoomRegion.rcut1,
                            (double) gadgetZoomRegion.corner[0], (double) gadgetZoomRegion.corner[1],
                            (double) gadgetZoomRegion.corner[2], (double) gadgetZoomRegion.cornerInset,
                            (double) gadgetZoomRegion.totalMeshSize, (double) gadgetZoomRegion.meshSize);
                    fprintf(f, "# id type x y z mass treeX treeY treeZ coarseX coarseY coarseZ "
                               "fineX fineY fineZ\n");
                    for (int i = 0; i < n; ++i)
                    {
                      const float4 pp = localTree.bodies_Ppos[i];
                      const real4  a  = localTree.bodies_acc1[i];
                      const int    ty = (i < (int) localTree.bodies_type.size())
                                      ? (int) localTree.bodies_type[i] : -1;
                      // Record the fine force the code ACTUALLY APPLIES, i.e. after the zoom mask
                      // -- the accumulate below is masked, and Gadget skips the readout for
                      // non-PLACEHIGHRESREGION types too (pm_nonperiodic.c:930-932). Dumping the
                      // unmasked value once made type 2 read as 38% wrong against Gadget when
                      // neither code uses it.
                      const bool  ap = (ty >= 0) && (((1u << ty) & gadgetZoomMask) != 0u);
                      fprintf(f, "%llu %d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g "
                                 "%.9g %.9g %.9g\n",
                              (unsigned long long) localTree.bodies_ids[i], ty,
                              pp.x, pp.y, pp.z, pp.w,
                              a.x, a.y, a.z,
                              cx[i], cy[i], cz[i],
                              ap ? fx[i] : 0.0f, ap ? fy[i] : 0.0f, ap ? fz[i] : 0.0f);
                    }
                    fclose(f);
                    fprintf(stderr, "[T54] force split dumped: %d particles -> %s\n", n, dumpSplit);
                  }
                }
              }
              pm_accumulate_masked_longrange(s_d_pmfx, s_d_pmfy, s_d_pmfz, s_d_pmpot,
                                              s_d_zoomfx, s_d_zoomfy, s_d_zoomfz, s_d_zoompot,
                                              localTree.bodies_typeDevice.raw_p(),
                                              gadgetZoomMask, localTree.n, gravStream->s());
              gravStream->sync();
              // T54's per-component dump needs the fine force on its own, before the accumulate
              // folds it in; it reads these through the pointers it captured, so nothing to do here
              // beyond keeping the arrays alive, which the statics above do.
              if (phaseTimeOn)
              {
                execStream->sync(); gravStream->sync();
                const double dt_ = get_time() - zpmT;
                s_cpuAcc.pm += dt_;
                if (phaseTimePrint)
                  fprintf(stderr, "[PHASE] iter=%d %-22s %8.4f s\n", iter, "zoomFineGridPM", dt_);
              }
            }
#endif
            // Phase 3 validation rung 2. Right here -- and only here -- the fresh long-range force
            // and the one the run has been using since the last PM step both exist, so the drift
            // across the interval can be measured without keeping a second copy of GravPM.
            // Deliberately before the scatter below, which overwrites GravPM.
            static const bool pmStaleness = (getenv("GADGET_HIP_PM_STALENESS") != NULL);
            if (pmStaleness && s_pmCadenceActive && s_pmKickCount > 0)
            {
              gravStream->sync();
              pmstale_reset();
              gadgetPMStaleness.set_args(0, &localTree.n,
                                         (void *) localTree.bodies_ids.as_const<CurrentOrder>(),
                                         (void *) &s_d_pmfx, (void *) &s_d_pmfy, (void *) &s_d_pmfz,
                                         (void *) &s_gravPMx, (void *) &s_gravPMy, (void *) &s_gravPMz);
              gadgetPMStaleness.setWork(localTree.n, 128);
              gadgetPMStaleness.execute2(gravStream->s());
              gravStream->sync();
              double num = 0.0, den = 0.0;
              pmstale_read(&num, &den);
              fprintf(stderr, "[PM-STALENESS] interval %ld  rms|da|/|a| = %.6e  "
                              "(sum|da|^2=%.6e sum|a|^2=%.6e)\n",
                      s_pmKickCount, (den > 0.0) ? sqrt(num / den) : 0.0, num, den);
            }
            if (pmIdSpaceOk)
            {
              pm_scatter_by_id(localTree.bodies_ids.raw_p(), s_d_pmfx, s_d_pmfy, s_d_pmfz, s_d_pmpot,
                               s_d_pmIdX, s_d_pmIdY, s_d_pmIdZ, s_d_pmIdPot,
                               localTree.n, gravStream->s());
              // Next PM step: the ladder-snapped displacement constraint, in the same units the
              // integrator uses (dloga under comoving integration, dt otherwise).
              const bool comoving = haveGadgetParams && gadgetParams.ComovingIntegrationOn != 0;
              const double span = comoving ? (gadgetLogTimeMax - gadgetLogTimeBegin)
                                           : (gadgetParams.TimeMax - gadgetParams.TimeBegin);
              double q = span;
              if (gadgetDtDisplacement > 0.0 && span > 0.0)
                while (q > gadgetDtDisplacement && q > 1e-12) q *= 0.5;
              else
                q = 0.0;
              s_pmNextTime = (q > 0.0)
                  ? (comoving ? (double) t_current * exp(q) : (double) t_current + q)
                  : -1.0;   // no usable constraint -> recompute next step
              static bool s_pmCadenceLogged = false;
              if (!s_pmCadenceLogged)
              {
                // Say which mode is ACTIVE. This used to print the cadence unconditionally, which
                // reads as "the cadence is on" even when PM is recomputed every step -- the
                // banner is the only runtime evidence of the mode, so it must not imply the
                // opposite of what is happening.
                if (pmEveryStep)
                  // T58: this message used to read "it cost 11% in Ekin, see tickets/T35", which
                  // blamed the SHIPPING cadence for its RETRACTED predecessor's cost. The 11%
                  // belonged to T36's withdrawn operator -- folding the stored long-range force
                  // into each active particle's own kick (T36:125, T35:317 "RETRACTED") -- not to
                  // the T42/T45 cadence that replaced it, whose worst Ekin deviation at 512^3 is
                  // 0.133% once the energy diagnostic and the Springel aold were given the
                  // long-range force too (T45). Leaving the old number in the runtime log made the
                  // cadence look 80x worse than it is, and it is the first thing anyone reads.
                  fprintf(stderr, "[T42] PM recomputed EVERY STEP (cadence off by default; enable "
                                  "with GADGET_HIP_PM_CADENCE=1 -- worst Ekin deviation 0.133%% at "
                                  "512^3, see tickets/T45. The 11%% in tickets/T35/T36 was the "
                                  "RETRACTED predecessor, not this cadence). The cadence this run "
                                  "WOULD have used: dt_displacement=%.6g -> PM step %.6g (%s).\n",
                          gadgetDtDisplacement, q, comoving ? "dloga" : "dt");
                else
                  fprintf(stderr, "[T36] PM CADENCE ACTIVE (opt-in): dt_displacement=%.6g -> PM step "
                                  "%.6g (%s), first recompute at t=%.9g\n",
                          gadgetDtDisplacement, q, comoving ? "dloga" : "dt", s_pmNextTime);
                s_pmCadenceLogged = true;
              }
            }
          }
          // Probe (GADGET_HIP_PM_ADD_PROBE=1): rms|acc1| immediately BEFORE and AFTER the PM add,
          // plus rms of the array actually handed to it, all in the same run at the same instant.
          // Subtracting rms values across separate runs implied a PM contribution of ~441 while
          // the array measures ~9.4; this measures the contribution instead of inferring it.
          static const bool addProbe = (getenv("GADGET_HIP_PM_ADD_PROBE") != NULL);
          static int addCount = 0;
          double rmsBefore = 0.0;
          if (addProbe && addCount < 3)
          {
            gravStream->sync(); execStream->sync();
            localTree.bodies_acc1.d2h();
            double sa = 0.0;
            for (int i = 0; i < localTree.n; ++i)
            { const real4 a = localTree.bodies_acc1[i];
              sa += (double)a.x*a.x + (double)a.y*a.y + (double)a.z*a.z; }
            rmsBefore = sqrt(sa/localTree.n);
          }
          // T23: masked. bodies_acc1 is refreshed by the tree walk for ACTIVE particles only, so
          // adding PM to the whole buffer double-counts it for every particle that stayed inactive.
          // tree.activePartlist is the walk's own output mask (zeroed before it, set to 1 for each
          // particle it writes), so it marks exactly the entries that are fresh.
          // Phase 3 bisect (GADGET_HIP_PM_KEEP_IN_ACC=1): fold the long-range force into acc1 the
          // way the every-step path does, while still solving PM on the cadence. Combined with
          // GADGET_HIP_PM_NO_KICK=1 this separates the three things the cadence changes at once:
          // acc1 content, the timestep criterion, and the separate kick.
          static const bool pmKeepInAcc = (getenv("GADGET_HIP_PM_KEEP_IN_ACC") != NULL);
          if (s_pmCadenceActive && !pmKeepInAcc)
          {
            // Phase 3: DO NOT fold the long-range force into bodies_acc1. That is the retracted
            // T36 operator, and it is wrong twice over: it reaches only active particles, and it
            // lets each of them integrate the long-range force over its own timestep instead of
            // over the PM interval. acc1 stays short-range only; the long-range part is applied
            // once per PM interval, to every particle, by gadget_pm_kick in correct().
            //
            // The POTENTIAL is a different matter -- it is a diagnostic, not dynamics -- so the
            // id-space potential is still folded in for the active particles that were just
            // refreshed, keeping the reported Etot a total rather than a short-range-only number.
            pm_add_by_id_masked(localTree.bodies_acc1.raw_p(), localTree.bodies_ids.raw_p(),
                                nullptr, nullptr, nullptr, s_d_pmIdPot, localTree.n,
                                (const int *) localTree.activePartlist.raw_p(), gravStream->s());
          }
          else if (pmIdSpaceOk)
          {
            // One masked gather-add for force AND potential together, from the id-space store.
            pm_add_by_id_masked(localTree.bodies_acc1.raw_p(), localTree.bodies_ids.raw_p(),
                                s_d_pmIdX, s_d_pmIdY, s_d_pmIdZ, s_d_pmIdPot, localTree.n,
                                (const int *) localTree.activePartlist.raw_p(), gravStream->s());
          }
          else
          {
            pm_add_force_to_acc_masked(localTree.bodies_acc1.raw_p(), s_d_pmfx, s_d_pmfy, s_d_pmfz,
                                       nullptr, localTree.n,
                                       (const int *) localTree.activePartlist.raw_p(),
                                       gravStream->s());
            // T23: masked, same reason as the force add above.
            pm_add_force_to_acc_masked(localTree.bodies_acc1.raw_p(), nullptr, nullptr, nullptr,
                                       s_d_pmpot, localTree.n,
                                       (const int *) localTree.activePartlist.raw_p(),
                                       gravStream->s());
          }
          gravStream->sync();
          if (addProbe && addCount < 3)
          {
            execStream->sync();
            localTree.bodies_acc1.d2h();
            double sa = 0.0;
            for (int i = 0; i < localTree.n; ++i)
            { const real4 a = localTree.bodies_acc1[i];
              sa += (double)a.x*a.x + (double)a.y*a.y + (double)a.z*a.z; }
            std::vector<float> gx(localTree.n);
            double sg = 0.0;
            if (s_gravPMx)
            {
              // Diagnostic only: GFreeAcc is layout-identical to float, so this reads the raw
              // G-free components into a float vector for an rms.
              CU_SAFE_CALL(hipMemcpy(&gx[0], s_gravPMx, (size_t)localTree.n*sizeof(GFreeAcc),
                                     hipMemcpyDeviceToHost));
              for (int i = 0; i < localTree.n; ++i) sg += (double)gx[i]*gx[i];
            }
            std::vector<float> fxh(localTree.n);
            double sf = 0.0;
            if (s_d_pmfx)
            {
              CU_SAFE_CALL(hipMemcpy(&fxh[0], s_d_pmfx, (size_t)localTree.n*sizeof(float),
                                     hipMemcpyDeviceToHost));
              for (int i = 0; i < localTree.n; ++i) sf += (double)fxh[i]*fxh[i];
            }
            fprintf(stderr, "[PM-ADD-PROBE] iter=%d  acc1 before=%.6e after=%.6e   "
                            "rms(GravPM_x)=%.6e  rms(pmfx)=%.6e  cadence=%d\n",
                    iter, rmsBefore, sqrt(sa/localTree.n),
                    sqrt(sg/localTree.n), sqrt(sf/localTree.n), (int) s_pmCadenceActive);
            addCount++;
          }
        }
      }
    }
#elif defined(PMGRID) && !defined(PERIODIC)
    // Isolated-TreePM counterpart to the PERIODIC block above (LOG.md §34-35): the real per-step
    // PM long-range force + potential for a non-periodic (isolated/vacuum-boundary) run, added on
    // top of the tree's already-computed (now correctly erfc-suppressed, §34) short-range force
    // in bodies_acc1. Same nProcs==1-only restriction and same reasoning as periodic -- this
    // port's isolated PM solver is also single-GPU with no MPI slab decomposition. Context comes
    // from pm_get_dump_context() (main.cpp's isolated setup block, storing meshSize/PMGRID via
    // the same generic function the periodic path uses for boxSize/PMGRID -- no parallel API).
    //
    // Caller contract carried over from pm.h's PMIsolatedSolver doc comment: particles must lie
    // within [0, meshSize/2) in each axis (the doubled grid's "occupied first half") for correct
    // isolated-boundary behavior -- this is the IC's responsibility, not enforced here, exactly
    // matching how the periodic path also doesn't enforce positions lying within [0, boxSize).
    if (!useDirectGravity)
    {
      float pmMeshSize; int pmGridSize;
      if (pm_get_dump_context(pmMeshSize, pmGridSize))
      {
        static bool s_pmIsoWarnedMultiProc = false;
        if (nProcs > 1)
        {
          if (!s_pmIsoWarnedMultiProc)
          {
            fprintf(stderr, "[PM] WARNING: nProcs=%d > 1 -- this port's isolated PM solver is "
                             "single-GPU only (no MPI slab decomposition), skipping PM long-range "
                             "force every step rather than silently computing it wrong. "
                             "Long-range gravity will be MISSING from this multi-process run.\n",
                             nProcs);
            s_pmIsoWarnedMultiProc = true;
          }
        }
        else
        {
          static PMIsolatedSolver s_pmIsoSolver;
          static bool  s_pmIsoSolverInit = false;
          static int   s_pmIsoSolverN = 0;
          static float *s_d_pmIsofx = nullptr, *s_d_pmIsofy = nullptr, *s_d_pmIsofz = nullptr, *s_d_pmIsopot = nullptr;

          if (!s_pmIsoSolverInit)
          {
            pm_fft_init();
            s_pmIsoSolver = pm_solver_create_isolated(pmGridSize, pmMeshSize);
            s_pmIsoSolverInit = true;
          }
          if (s_pmIsoSolverN != localTree.n)
          {
            if (s_d_pmIsofx) { hipFree(s_d_pmIsofx); hipFree(s_d_pmIsofy); hipFree(s_d_pmIsofz); hipFree(s_d_pmIsopot); }
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmIsofx, localTree.n * sizeof(float)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmIsofy, localTree.n * sizeof(float)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmIsofz, localTree.n * sizeof(float)));
            CU_SAFE_CALL(hipMalloc((void**)&s_d_pmIsopot, localTree.n * sizeof(float)));
            s_pmIsoSolverN = localTree.n;
          }

          pm_compute_forces_isolated(s_pmIsoSolver, localTree.bodies_Ppos.raw_p(), localTree.n,
                                      s_d_pmIsofx, s_d_pmIsofy, s_d_pmIsofz, gravStream->s(),
                                      /*gravityConstant=*/1.0f, s_d_pmIsopot);
          // T23: masked. bodies_acc1 is refreshed by the tree walk for ACTIVE particles only, so
          // adding PM to the whole buffer double-counts it for every particle that stayed inactive.
          // tree.activePartlist is the walk's own output mask (zeroed before it, set to 1 for each
          // particle it writes), so it marks exactly the entries that are fresh.
          pm_add_force_to_acc_masked(localTree.bodies_acc1.raw_p(), s_d_pmIsofx, s_d_pmIsofy,
                                     s_d_pmIsofz, nullptr, localTree.n,
                                     (const int *) localTree.activePartlist.raw_p(),
                                     gravStream->s());
          // T23: masked, same reason as the force add above.
          pm_add_force_to_acc_masked(localTree.bodies_acc1.raw_p(), nullptr, nullptr, nullptr,
                                     s_d_pmIsopot, localTree.n,
                                     (const int *) localTree.activePartlist.raw_p(),
                                     gravStream->s());
          gravStream->sync();
        }
      }
    }
#endif

    // T58: the zoom fine mesh used to be solved HERE, every step, outside the PM cadence that
    // governs the coarse mesh. That was a divergence from Gadget, which puts both meshes behind the
    // single `if(All.PM_Ti_endstep == All.Ti_Current)` guard in accel.c:35-40 and solves them
    // together inside long_range_force() (longrange.c:54-82). Measured at 115M particles with
    // GADGET_HIP_PM_CADENCE=1: the coarse mesh fired once in ten steps and the fine mesh ten times
    // out of ten, costing 1.586 s/step at GADGET_HIP_PMGRID=512 -- 39% of the step, and the whole
    // of why 512 looked 41% slower than 256 in T57.
    //
    // It now lives inside the coarse mesh's own `pmDue` branch above, accumulating into the same
    // s_d_pmf{x,y,z}/s_d_pmpot arrays, which is Gadget's structure exactly: both solves `+=` into
    // GravPM and every consumer downstream reads the sum. That single move gives the fine mesh the
    // cadence, the id-space store that survives the re-sort, the once-per-interval kick, the
    // drift-corrected energy diagnostic and the Springel `aold` -- all of which T42/T45 had already
    // built for the coarse mesh and none of which the fine mesh was reaching.

    // Phase 5 ticket 08 (PLAN.md): a real, serious bug found via this ticket's own end-to-end
    // comparison against stock Gadget-2 -- structure growth was suppressed by orders of magnitude
    // (P(k) grew ~7x over the run instead of stock's ~4700x at the largest scale) because NEITHER
    // the tree kernel (add_acc(), dev_approximate_gravity_warp_new.cu -- mirrors Gadget-2's own
    // forcetree.c convention of a plain mass/r^2 force with `All.G` multiplied in separately,
    // gravtree.c:328) NOR either PM call above (`/*gravityConstant=*/1.0f`, hardcoded, both
    // periodic and isolated) NOR the zoom fine-grid PM call ever actually used the real
    // `gadgetParams.G` (43007 for this project's own phase0 test, vastly different from Bonsai's
    // original G=1 toy-unit convention every prior Phase 3/4 test ran at) -- this port's whole
    // per-step dynamics were silently running at G=1 regardless of what `--param` said. Invisible
    // through every earlier ticket because they all validated at G=1 (Plummer spheres, the Ewald
    // ground-truth test, etc.), where "forgot to multiply by G" and "multiplied by 1" are the same
    // bug or no bug at all. Fixed here with ONE combined scale of the fully-summed bodies_acc1
    // (tree + periodic/isolated coarse PM + zoom fine PM, force .xyz AND potential .w together,
    // since PM's own output is linearly proportional to its gravityConstant argument, currently
    // fixed at 1.0f above) -- deliberately not fixed by threading gadgetParams.G into the tree
    // kernel and both PM calls separately, which would be three separate edits with more
    // duplication risk for a value that's exactly linear in all of them. Gated off for --direct
    // (an N^2 validation-only mode never meant to run a real cosmological box, existing
    // convention) and for every non-`--param` run (preserves G=1 exactly for every existing
    // diagnostic/dev test that depends on it, e.g. --pm-test-*, --plummer).
    // T14 spike hunt: sample |a| immediately BEFORE the G scaling as well as after. The 512^3
    // excursion multiplies EVERY particle's |a| and phi by ~43 -- and G is 43.0070779 for this
    // run -- so the question is whether acc1 arrives here already carrying a G factor from the
    // previous iteration (i.e. the walk did not recompute it, but this scale ran anyway), which
    // would compound. Comparing pre- and post-scale medians answers that directly.
    PHASE_END("PM + long-range", pm);
    spike_probe(localTree, iter, "pre-Gscale");
    if (!useDirectGravity && haveGadgetParams && gadgetParams.G > 0.0)
    {
      // T23: masked. This scale used to run over the WHOLE buffer every iteration, on the
      // assumption stated in T4-FIX-G's own comment ("ONE combined scale of the fully-summed
      // bodies_acc1") -- which holds only if the walk recomputes acc1 for every particle. With
      // individual timesteps it does not: measured at 512^3, 3 of 4,194,801 groups were active and
      // 134M particles had G applied to values that already carried it, twice in a row (43^2).
      pm_scale_acc_masked(localTree.bodies_acc1.raw_p(), localTree.n,
                          (float) gadgetParams.G,
                          (const int *) localTree.activePartlist.raw_p(), gravStream->s());
      gravStream->sync();
    }

    spike_probe(localTree, iter, "post-walk");

    // Phase 2/3 accuracy validation: dump predicted positions + accelerations on a chosen
    // iteration (default 0, override with GADGET_HIP_DUMP_ACC_ITER) so a tree-code run and a
    // --direct (N^2) run on the same IC can be diffed offline. Gated by an env var so it's
    // zero-cost/no-op in normal operation. Iteration 0 is only meaningful for the improved-BH MAC
    // (theta-based, no history needed); the Springel MAC needs a real prior-step bodies_acc0 to
    // test its actual steady-state behavior -- bodies_acc0 is still all-zero on iter 0, which
    // makes the MAC trivially always-open (not a real test) -- so use iter 1+ for that comparison.
    const int dumpIter = getenv("GADGET_HIP_DUMP_ACC_ITER") ? atoi(getenv("GADGET_HIP_DUMP_ACC_ITER")) : 0;
    if (iter == dumpIter)
    {
      const char *dumpPath = getenv("GADGET_HIP_DUMP_ACC");
      if (dumpPath)
      {
        // approximate_gravity() writes its result into bodies_acc1 (Bonsai's old/new
        // double-buffer, combined later in correct()); direct_gravity() writes straight into
        // bodies_acc0. Dump both plus the original particle id, since the tree build's SFC sort
        // reorders bodies_Ppos/bodies_ids relative to the unsorted direct-gravity run -- indices
        // alone can't be used to match particles between a tree run and a direct run.
        localTree.bodies_Ppos.d2h();
        localTree.bodies_acc0.d2h();
        localTree.bodies_acc1.d2h();
        localTree.bodies_ids.d2h();

        // Phase 4 (PLAN.md) combined tree+PM validation: if this is a PMGRID build and the run's
        // boxSize/PMGRID context was recorded (main.cpp's pm_set_dump_context, called once right
        // after IC load for PERIODIC builds), also compute the PM long-range force for these same
        // (already tree-force-computed) particles and append it as 3 extra columns -- an external
        // harness combines this with the tree columns above (bodies_acc1 for the improved-BH/
        // Springel MAC path, bodies_acc0 for --direct) and checks the SUM against a reference,
        // closing the "combined tree+PM force never validated against anything" gap noted in
        // PLAN.md/LOG.md. Deliberately a totally separate one-shot solver here (create+use+destroy)
        // rather than reusing any long-lived solver -- this diagnostic path is not performance
        // sensitive and keeping it self-contained avoids threading solver lifetime through the
        // rest of octree/gpu_iterate.cpp for a debug-only feature.
        std::vector<float> h_pmfx, h_pmfy, h_pmfz;
#ifdef PMGRID
        float pmBoxSize; int pmGridSize;
        bool havePm = pm_get_dump_context(pmBoxSize, pmGridSize);
        if (havePm)
        {
          pm_fft_init();
          PMPeriodicSolver pmSolver = pm_solver_create_periodic(pmGridSize, pmBoxSize);
          float *d_pmfx = nullptr, *d_pmfy = nullptr, *d_pmfz = nullptr;
          CU_SAFE_CALL(hipMalloc((void**)&d_pmfx, localTree.n * sizeof(float)));
          CU_SAFE_CALL(hipMalloc((void**)&d_pmfy, localTree.n * sizeof(float)));
          CU_SAFE_CALL(hipMalloc((void**)&d_pmfz, localTree.n * sizeof(float)));
          pm_compute_forces_periodic(pmSolver, localTree.bodies_Ppos.raw_p(), localTree.n,
                                      d_pmfx, d_pmfy, d_pmfz, 0);
          CU_SAFE_CALL(hipDeviceSynchronize());
          h_pmfx.resize(localTree.n); h_pmfy.resize(localTree.n); h_pmfz.resize(localTree.n);
          CU_SAFE_CALL(hipMemcpy(h_pmfx.data(), d_pmfx, localTree.n * sizeof(float), hipMemcpyDeviceToHost));
          CU_SAFE_CALL(hipMemcpy(h_pmfy.data(), d_pmfy, localTree.n * sizeof(float), hipMemcpyDeviceToHost));
          CU_SAFE_CALL(hipMemcpy(h_pmfz.data(), d_pmfz, localTree.n * sizeof(float), hipMemcpyDeviceToHost));
          hipFree(d_pmfx); hipFree(d_pmfy); hipFree(d_pmfz);
          pm_solver_destroy_periodic(pmSolver);
        }
#endif

        FILE *dumpF = fopen(dumpPath, "w");
        for (int i = 0; i < localTree.n; i++)
        {
          fprintf(dumpF, "%llu %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g",
                  (unsigned long long)localTree.bodies_ids[i],
                  localTree.bodies_Ppos[i].x, localTree.bodies_Ppos[i].y, localTree.bodies_Ppos[i].z,
                  localTree.bodies_Ppos[i].w,
                  localTree.bodies_acc0[i].x, localTree.bodies_acc0[i].y, localTree.bodies_acc0[i].z,
                  localTree.bodies_acc0[i].w,
                  localTree.bodies_acc1[i].x, localTree.bodies_acc1[i].y, localTree.bodies_acc1[i].z,
                  localTree.bodies_acc1[i].w);
          if (!h_pmfx.empty())
            fprintf(dumpF, " %.9g %.9g %.9g", h_pmfx[i], h_pmfy[i], h_pmfz[i]);
          fprintf(dumpF, "\n");
        }
        fclose(dumpF);
      }
    }

    // Phase 3 warm-start accuracy validation: the plain iter==0 comparison above is degenerate
    // for the Springel MAC (bodies_acc0 is still all-zero, so the MAC trivially always-opens --
    // see AMD_PORT_PLAN.md Phase 2 §10/Phase 3 §11) and comparing at iter>=1 across independently
    // -run binaries is invalid (each run's own iter-0 forces already sent its particles down a
    // different leapfrog trajectory before iter 1's forces are even computed). This block sidesteps
    // both problems: seed bodies_acc0 with the *just-computed* gravity result (mimicking what
    // correct() normally does at the end of a step) WITHOUT calling correct()/predict() -- so no
    // integration happens and every particle's position is byte-identical to what a plain iter-0
    // run (tree-code or --direct) would see -- then immediately recompute gravity a second time on
    // those same, unmoved positions. The Springel MAC's second pass now sees a real, non-zero
    // |a_old|, and the dumped positions are still directly comparable to a --direct run's iter-0
    // dump (GADGET_HIP_DUMP_ACC, same IC, same seed). Exits immediately after dumping -- this is a
    // diagnostic-only code path, not a real simulation step (gravity has been evaluated twice with
    // the same positions, so nothing downstream should treat this as normal iteration state).
    if (iter == 0)
    {
      const char *warmDumpPath = getenv("GADGET_HIP_DUMP_ACC_WARM");
      if (warmDumpPath)
      {
        if (useDirectGravity)
        {
          // bodies_acc0 already holds the just-computed direct-gravity result -- nothing to copy.
        }
        else
        {
          localTree.bodies_acc0.copy_devonly(localTree.bodies_acc1, localTree.n);
        }

        if (useDirectGravity)
          direct_gravity(this->localTree);
        else
          approximate_gravity(this->localTree);
        gravStream->sync();

        localTree.bodies_Ppos.d2h();
        localTree.bodies_acc0.d2h();
        localTree.bodies_acc1.d2h();
        localTree.bodies_ids.d2h();
        FILE *warmDumpF = fopen(warmDumpPath, "w");
        for (int i = 0; i < localTree.n; i++)
        {
          fprintf(warmDumpF, "%llu %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
                  (unsigned long long)localTree.bodies_ids[i],
                  localTree.bodies_Ppos[i].x, localTree.bodies_Ppos[i].y, localTree.bodies_Ppos[i].z,
                  localTree.bodies_Ppos[i].w,
                  localTree.bodies_acc0[i].x, localTree.bodies_acc0[i].y, localTree.bodies_acc0[i].z,
                  localTree.bodies_acc0[i].w,
                  localTree.bodies_acc1[i].x, localTree.bodies_acc1[i].y, localTree.bodies_acc1[i].z,
                  localTree.bodies_acc1[i].w);
        }
        fclose(warmDumpF);
        exit(0);
      }
    }

    idata.lastGravTime      = get_time() - t1;
    idata.totalGravTime    += idata.lastGravTime;
    idata.lastLETCommTime   = thisPartLETExTime;
    idata.totalLETCommTime += thisPartLETExTime;


    //Compute the total number of interactions that we executed
    tTempTime = get_time();
// Paired with INTCOUNT in dev_approximate_gravity_warp_new.cu and with the tree.interactions
// allocation in build.cpp: all three must be re-enabled together. The buffer is 1 element while
// this is off, so reading it here would run off the end.
#if 0
   localTree.interactions.d2h();

   long long directSum = 0;
   long long apprSum = 0;

   for(int i=0; i < localTree.n; i++)
   {
     apprSum     += localTree.interactions[i].x;
     directSum   += localTree.interactions[i].y;
   }
   char buff2[512];
   sprintf(buff2, "INT Interaction at (rank= %d ) iter: %d\tdirect: %llu\tappr: %llu\tavg dir: %f\tavg appr: %f\n",
                   procId,iter, directSum ,apprSum, directSum / (float)localTree.n, apprSum / (float)localTree.n);
   devContext->writeLogEvent(buff2);
#endif
   LOGF(stderr,"Stats calculation took: %lg \n", get_time()-tTempTime);


    float ms=0, msLET=0;
#if 1 //enable when load-balancing, gets the accurate GPU time from events
    CU_SAFE_CALL(hipEventElapsedTime(&ms, startLocalGrav, endLocalGrav));
    // T40: remember the walk's own GPU time so the next step can size its chunks from measurement
    // rather than a guess. These events are already recorded and already read here, so this costs
    // nothing and adds no synchronisation.
    g_t40LastWalkMs = ms;
    if(nProcs > 1)  CU_SAFE_CALL(hipEventElapsedTime(&msLET,startRemoteGrav, endRemoteGrav));

    msLET += runningLETTimeSum;

    char buff[512];
    sprintf(buff,  "APPTIME [%d]: Iter: %d\t%g \tn: %d EventTime: %f  and %f\tSum: %f\n",
        procId, iter, idata.lastGravTime, this->localTree.n, ms, msLET, ms+msLET);
    LOGF(stderr,"%s", buff);
    devContext->writeLogEvent(buff);
#else
    ms    = 1;
    msLET = 1;
#endif

    idata.lastGPUGravTimeLocal   = ms;
    idata.lastGPUGravTimeLET     = msLET;
    idata.totalGPUGravTimeLocal += ms;
    idata.totalGPUGravTimeLET   += msLET;

    //Different options for basing the load balance on
    lastLocal = ms;
    lastTotal = ms + msLET;

    // T19 (2026-09-02): PRODUCTION energy diagnostic, sampled HERE -- after gravity (bodies_acc1
    // already holds this step's fresh force, tree/PM-combined) but strictly BEFORE correct()
    // overwrites bodies_vel with the post-kick velocity and resets oriParticleOrder -- to match
    // Gadget-2's own energy_statistics() call order (run.c:47-59, called BEFORE
    // advance_and_find_timesteps()'s kick). Previously this call lived after correct() and read
    // the POST-kick velocity paired with the new position/potential; Gadget-2 itself always pairs
    // the freshly-drifted position/potential with the OLD (pre-kick) velocity instead -- a genuine
    // half-step sampling-point mismatch in the driver, found by direct comparison against
    // Gadget2/run.c, not assumed. See compute_energies()'s own `usePreKickState` doc and
    // compute_energy_double_prekick (timestep.cu) for the exact mechanism.
    tTempTime = get_time();
    devContext->startTiming(execStream->s());
    // T56: an output list drives the snapshots on its own; TimeBetSnapshot is then
    // irrelevant and need not be set at all (Gadget only validates it on the else
    // branch of find_next_outputtime, run.c:273-290).
    if (haveGadgetParams && (outputListActive() || gadgetParams.TimeBetSnapshot > 0.0) &&
        t_current >= nextGadgetSnapTime)
    {
      const double requestedSnapTime = nextGadgetSnapTime;   // C-D-11
      writeGadgetSnapshot(/*preKick=*/true);
      // A while-loop, not a single bump: this port always recomputes PM every step (LOG.md
      // ticket 05), so a single physical step can cross several TimeBetSnapshot boundaries at
      // once (e.g. early in a comoving run, when MaxSizeTimestep's additive-in-`a` step is large
      // relative to the still-small multiplicative cadence gap) -- a single bump would leave
      // nextGadgetSnapTime behind t_current and fire again next step, writing far more snapshots
      // than intended. Matches the *intent* of Gadget-2's own run.c ~279-330 (find the next
      // output time still >= the current one, skipping past any already in the past), simplified
      // for this port's own no-look-ahead per-step loop.
      int nCrossed = 0;
      do
      {
        nextGadgetSnapTime = nextOutputTimeAfter(nextGadgetSnapTime);
        nCrossed++;
      } while (nextGadgetSnapTime <= t_current);

      // C-D-11: Gadget-2 emits ONE snapshot per crossed output time, partially drifting to each
      // (run.c:206-227: move_particles(Ti_Current, Ti_nextoutput); savepositions; repeat). This
      // port cannot: it has no partial drift, so the extra files would be byte-identical copies at
      // one epoch, which is worse than useless. The loop above therefore skips them -- but that
      // silently drops outputs the parameter file asked for, so SAY SO rather than let a missing
      // file be discovered later.
      if (nCrossed > 1)
        fprintf(stderr, "[C-D-11] WARNING: one system step crossed %d output times; %d were "
                        "SKIPPED (this port writes one snapshot per step, not one per crossed "
                        "output time -- it has no partial drift). Increase TimeBetSnapshot or "
                        "reduce MaxSizeTimestep if every output is required.\n",
                nCrossed, nCrossed - 1);

      // C-D-11: and report the epoch error even for the one we did write. Gadget drifts to the
      // requested time before writing, so its residual is <= 1 timeline tick; this port writes at
      // whatever t_current the step landed on. C-D-10's truncation makes the FINAL snapshot exact,
      // but intermediate ones carry up to one system step of error.
      {
        const double relErr = (requestedSnapTime > 0.0)
                                ? fabs((double) t_current - requestedSnapTime) / requestedSnapTime
                                : 0.0;
        if (relErr > 1e-6)
          fprintf(stderr, "[C-D-11] snapshot written at t=%.9g, requested t=%.9g "
                          "(%.3g relative); this port does not drift to the output time.\n",
                  (double) t_current, requestedSnapTime, relErr);
      }
    }

    // ENERGY CADENCE (contract C-D-15 neighbourhood; measured optimisation, see tickets/T35).
    //
    // This is a DIAGNOSTIC, and the port computed it on every single step. Gadget-2 computes the
    // system's global energies only at `TimeBetStatistics` cadence -- `every_timestep_stuff()` /
    // `compute_global_quantities_of_system()` from run.c, which is also the only thing that writes
    // energy.txt. Measured at 512^3 on a sparse step (1.6% of groups active): 9.48 s of a 14.88 s
    // step, 64% of the entire step, against 0.42 s for the gravity walk it exists to check. Nine
    // of every ten steps in a 9557-step run were spending two thirds of their time on a number
    // nobody reads.
    //
    // Default is now Gadget's cadence. GADGET_HIP_ENERGY_EVERY_STEP=1 restores per-step evaluation
    // (what the acceptance baselines were recorded with, and what any run comparing `de` per
    // iteration needs). The first and last steps always evaluate, so a run's endpoints are always
    // exact regardless of cadence.
    double de = 0.0;   // energyDue was hoisted above the PM block -- see there for why
    if (energyDue)
    {
      PHASE_BEGIN();
      de = compute_energies(this->localTree, /*usePreKickState=*/true);
      PHASE_END("compute_energies", potential);
    }
    if (energyStatDue)
    {
      // This sample used a freshly-computed potential for every particle, so it is directly
      // comparable to Gadget-2's own energy.txt. The per-iteration `de` above is not.
      fprintf(stderr, "[ENERGY-EXACT] iter=%d time=%.9g de=%.9g Ekin=%.9g Epot=%.9g\n",
              iter, t_current, de, this->Ekin, this->Epot);
      nextEnergyStat += gadgetParams.TimeBetStatistics;
      energy_set_force_all_active(0);
    }
    devContext->stopTiming("Energy", 7, execStream->s());
    idata.totalPredCor += get_time() - tTempTime;

    //Corrector
    tTempTime = get_time();
    devContext->startTiming(execStream->s());
    STAGE_TRACE("[BUG4-STAGE] before correct iter=%d\n", iter);
    // Ticket T12: capture this iteration's freshly-computed oriParticleOrder permutation
    // (the "unsorted" array correct_particles() is about to consume) before correct_particles()
    // resets it back to identity at the end of its own kernel.
    dumpUnsortedMapping(this->localTree, iter);
    cd01_stats(this->localTree, "KICK", iter, t_previous, t_current);
    PHASE_BEGIN();
    correct(this->localTree);
    PHASE_END("correct", timeline);
    cd01_trace(this->localTree, "POST", iter, t_previous, t_current);
    cd01_stats(this->localTree, "POST", iter, t_previous, t_current);

    if (iter == 0) reportDeviceMemory("after first completed step");
    STAGE_TRACE("[BUG4-STAGE] after correct iter=%d\n", iter);
    // Gated 2026-08-30 -- see the matching AFTER_PREDICT call site's comment above for why.
    if (getenv("GADGET_HIP_TRACE_DUP"))
      traceDuplicatePositions(this->localTree, this->localTree.bodies_Ppos, "AFTER_CORRECT", iter);
    // Ticket T11: post-correct dump -- final committed pos/vel/acc/timestep for this iteration.
    traceParticleTrajectory(this->localTree, "AFTER_CORRECT", iter, (double) t_current);
    devContext->stopTiming("Correct", 8, execStream->s());
    idata.totalPredCor += get_time() - tTempTime;



    if(nProcs > 1)
    {
      #ifdef USE_MPI
      //Wait on all processes and time how long the waiting took
      t1 = get_time();
      devContext->startTiming(execStream->s());
      //Gather info about the load-balance, used to decide if we need to refine the domains
      MPI_Allreduce(&lastTotal, &maxExecTimePrevStep, 1, MPI_FLOAT, MPI_MAX, mpiCommWorld);
      MPI_Allreduce(&lastTotal, &avgExecTimePrevStep, 1, MPI_FLOAT, MPI_SUM, mpiCommWorld);
      avgExecTimePrevStep /= nProcs;

      devContext->stopTiming("Unbalance", 12, execStream->s());
      idata.lastWaitTime  += get_time() - t1;
      idata.totalWaitTime += idata.lastWaitTime;
      #endif
    }

    idata.Nact_since_last_tree_rebuild += this->localTree.n_active_particles;

    // T19: production energy diagnostic already computed above, BEFORE correct() -- see that call
    // site's comment for why (Gadget-2 timing match). Not recomputed here.

    if(statisticsIter > 0)
    {
      if(t_current >= nextStatsTime)
      {
        nextStatsTime += statisticsIter;
        double tDens0 = get_time();
        localTree.bodies_Ppos.d2h();
        localTree.bodies_vel.d2h();
        localTree.bodies_ids.d2h();

        double tDens1 = get_time();
        // DENSITY/DISKSTATS each hold two [N_MESH][4*N_MESH] float arrays (32MB total,
        // see postProcessModules.h) -- heap-allocated rather than stack-constructed so
        // iterate_once's frame doesn't reserve 32MB on every call even when this
        // statisticsIter-gated block never runs (it caused a stack-overflow SIGSEGV on
        // the default 8MB thread stack).
        const std::unique_ptr<DENSITY> dens(new DENSITY(mpiCommWorld, procId, nProcs, localTree.n,
                           &localTree.bodies_Ppos[0],
                           &localTree.bodies_vel[0],
                           &localTree.bodies_ids[0],
                           1, 2.33e9, 20, "density", t_current));

        double tDens2 = get_time();
        if(procId == 0) LOGF(stderr,"Density took: Copy: %lg Create: %lg \n", tDens1-tDens0, tDens2-tDens1);

        double tDisk1 = get_time();
        const std::unique_ptr<DISKSTATS> diskstats(new DISKSTATS(mpiCommWorld, procId, nProcs, localTree.n,
                           &localTree.bodies_Ppos[0],
                           &localTree.bodies_vel[0],
                           &localTree.bodies_ids[0],
                           1, 2.33e9, "diskstats", t_current));

        double tDisk2 = get_time();
        if(procId == 0) LOGF(stderr,"Diskstats took: Create: %lg \n", tDisk2-tDisk1);
      }
    }//Statistics dumping


    if (useMPIIO)
    {
#ifdef USE_MPI
      if (mpiRenderMode) dumpDataMPI(); //To renderer process
      else               dumpData();    //To disk
#endif
    }
    else if (snapshotIter > 0)
    {
      if((t_current >= nextSnapTime))
      {
        nextSnapTime += snapshotIter;

        while(!ioSharedData.writingFinished)
        {
          fprintf(stderr,"Waiting till previous snapshot has been written\n");
          usleep(100); //Wait till previous snapshot is written
        }

        ioSharedData.t_current  = t_current;

        //TODO JB, why do we do malloc here?
        assert(ioSharedData.nBodies == 0);
        ioSharedData.malloc(localTree.n);


        localTree.bodies_Ppos.d2h(localTree.n, ioSharedData.Pos);
        localTree.bodies_vel.d2h(localTree.n, ioSharedData.Vel);
        localTree.bodies_ids.d2h(localTree.n, ioSharedData.IDs);
        ioSharedData.writingFinished = false;
        if(nProcs <= 16) while (!ioSharedData.writingFinished);
      }
    }

    // Phase 5 ticket 08 (PLAN.md): independent of the useMPIIO/snapshotIter dump mechanism above
    // (a --param run typically sets neither) -- see octree.h's nextGadgetSnapTime declaration
    // comment for why this exists and how the multiplicative-vs-additive cadence rule is chosen.

    // Restart: dump at CpuTimeBetRestartFile intervals, and once unconditionally when the run
    // ends. Placed here, at the end of a completed iteration, because that is the only point where
    // positions, velocities, times and acc0 all describe the SAME instant -- mid-iteration the
    // predictor has advanced positions but not yet applied the kick.
    if (!restartPath.empty() && restartCpuInterval > 0.0)
    {
      const double now = get_time();
      if (lastRestartWrite == 0.0) lastRestartWrite = now;   // arm on the first iteration
      if ((now - lastRestartWrite) >= restartCpuInterval)
      {
        writeRestartFile(restartPath.c_str());
        lastRestartWrite = now;
      }
    }

    // Graceful stop, the other half of Gadget-2's stop/restart pair (run.c:143-193). Two triggers,
    // both checked here at the same safe point as the periodic dump:
    //   1. a file named `stop` in OutputDir -- the user's way to end a run cleanly at the next
    //      step boundary rather than killing it mid-iteration. Removed once acted on, exactly as
    //      Gadget-2 does, so the next run is not stopped by a leftover file.
    //   2. 0.85 * TimeLimitCPU of wall time consumed -- Gadget-2's own margin (run.c:161), which
    //      leaves room to finish the dump before a batch scheduler kills the job.
    // Both write a restart file first if one is configured; without --restart-file there is
    // nothing to write, so the run just stops and says so rather than pretending it saved state.
    if (haveGadgetParams)
    {
      const std::string stopFile = gadgetParams.OutputDir.empty()
                                     ? std::string("stop") : gadgetParams.OutputDir + "/stop";
      bool stopNow = false;
      const char *why = NULL;
      if (access(stopFile.c_str(), F_OK) == 0) { stopNow = true; why = "stop file found"; remove(stopFile.c_str()); }
      else if (gadgetParams.TimeLimitCPU > 0.0 &&
               (get_time() - idata.startTime) > 0.85 * gadgetParams.TimeLimitCPU)
      { stopNow = true; why = "0.85 * TimeLimitCPU reached"; }

      if (stopNow)
      {
        if (!restartPath.empty())
        {
          writeRestartFile(restartPath.c_str());
          fprintf(stderr, "[STOP] %s -- restart file written, resume with --resume --restart-file %s\n",
                  why, restartPath.c_str());
        }
        else
        {
          fprintf(stderr, "[STOP] %s -- NO restart file configured (--restart-file was not given), "
                          "so this run cannot be resumed.\n", why);
        }
        return true;
      }
    }

    if (iter >= iterEnd)
    {
      if (!restartPath.empty()) writeRestartFile(restartPath.c_str());
      return true;
    }

    if(t_current >= tEnd)
    {
      if (!restartPath.empty()) writeRestartFile(restartPath.c_str());
      compute_energies(this->localTree);
      // Phase 5 ticket 08 (PLAN.md): a forced final snapshot, matching Gadget-2's own run.c:136
      // ("write a last snapshot file") -- unconditional so the run's true endpoint is always
      // captured even if it doesn't land exactly on a TimeBetSnapshot boundary.
      if (haveGadgetParams) writeGadgetSnapshot();
      double totalTime = get_time() - idata.startTime;
      LOG("Finished: %f > %f \tLoop alone took: %f\n", t_current, tEnd, totalTime);
      my_dev::base_mem::printMemUsage();
      return true;
    }
    if (phaseTimeOn)
    {
      execStream->sync(); gravStream->sync();
      const double stepTotal = get_time() - phStepT0;
      s_cpuAcc.total += stepTotal;
      if (phaseTimePrint)
        fprintf(stderr, "[PHASE] iter=%d %-22s %8.4f s\n", iter, "STEP TOTAL", stepTotal);

      // Gadget-2's own cpu.txt, column for column (run.c:394-401), so the two files can be diffed
      // directly. Columns this port has no analogue for are 0, and that is informative rather than
      // missing: Hydro/Hyd*/EnsureNgb are 0 because there is no SPH, and CommSum/Imbalance are 0
      // because there is one GPU and no MPI -- the CPU reference spent 28% of its total in
      // Imbalance. CPU_Gravity is the whole gravity pipeline (walk + tree build + Peano sort + PM),
      // matching Gadget's own use of it as the umbrella figure; the columns overlap there in
      // Gadget too, so Total is not the sum of the columns in either code.
      if (cpuLogOn && mpiGetRank() == 0)
      {
        static FILE *s_cpuFile = NULL;
        if (!s_cpuFile && !gadgetParams.OutputDir.empty())
        {
          std::string cp = gadgetParams.OutputDir;
          if (!cp.empty() && cp[cp.size()-1] != '/') cp += '/';
          cp += "cpu.txt";
          s_cpuFile = fopen(cp.c_str(), "w");
        }
        if (s_cpuFile)
        {
          const double gravity = s_cpuAcc.treewalk + s_cpuAcc.treebuild + s_cpuAcc.peano + s_cpuAcc.pm;
          fprintf(s_cpuFile, "Step %d, Time: %g, CPUs: %d\n", iter, (double) t_current, 1);
          fprintf(s_cpuFile,
                  "%10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f "
                  "%10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n",
                  s_cpuAcc.total, gravity, 0.0, s_cpuAcc.domain, s_cpuAcc.potential,
                  s_cpuAcc.predict, s_cpuAcc.timeline, s_cpuAcc.snapshot,
                  s_cpuAcc.treewalk, s_cpuAcc.treebuild,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  s_cpuAcc.pm, s_cpuAcc.peano);
          fflush(s_cpuFile);
        }
      }
    }
    iter++;

    return false;
}



void octree::iterate_teardown(IterationData &idata) {
  if(execStream != NULL) {
    delete execStream;
    execStream = NULL;
  }

  if(gravStream != NULL) {
    delete gravStream;
    gravStream = NULL;
  }

  if(copyStream != NULL) {
    delete copyStream;
    copyStream = NULL;
  }

  if(LETDataToHostStream != NULL)  {
    delete LETDataToHostStream;
    LETDataToHostStream = NULL;
  }
}

void octree::iterate(bool amuse) {
  IterationData idata;
  if(!amuse) iterate_setup();
  idata.startTime = get_time();


  while(true)
  {
    bool stopRun = iterate_once(idata);

    double totalTime = get_time() - idata.startTime;

    static char textBuff[16384];
    sprintf(textBuff,"TIME [%02d] TOTAL: %g\t Grav: %g (GPUgrav %g , LET Com: %g)\tBuild: %g\tDomain: %g\t Wait: %g\tdomUp: %g\tdomEx: %g\tdomWait: %g\ttPredCor: %g\n",
                      procId, totalTime, idata.totalGravTime,
                      (idata.totalGPUGravTimeLocal+idata.totalGPUGravTimeLET) / 1000,
                      idata.totalLETCommTime,
                      idata.totalBuildTime, idata.totalDomTime, idata.lastWaitTime,
                      idata.totalDomUp, idata.totalDomEx, idata.totalDomWait, idata.totalPredCor);

    if (procId == 0)
    {
      LOGF(stderr,"%s", textBuff);
      LOGF(stdout,"%s", textBuff);
    }

    devContext->writeLogEvent(textBuff);
    this->writeLogToFile();     //Write the logdata to file

    if(stopRun) break;
  } //end while

  if(!amuse) iterate_teardown(idata);
} //end iterate


void octree::predict(tree_structure &tree)
{
  //Functions that predicts the particles to the next timestep

  //tend is time per particle
  //tnext is reduce result

  //First we get the minimum time, which is the next integration time
  #ifdef DO_BLOCK_TIMESTEP
    getTNext.set_args(sizeof(int)*128, &tree.n, tree.bodies_time.p(), tnext.p());
    getTNext.setWork(-1, 128, NBLOCK_REDUCE);
    getTNext.execute2(execStream->s());

    //TODO
    //This will not work in block-step! Only shared- time step
    //in block step we need syncs and global communication
    if(tree.n == 0)
    {
      Ti_previous = Ti_current;
      Ti_current  = std::min<gadget_tick_t>(Ti_current + 1, GADGET_TIMEBASE);
      t_previous  = (float) gadgetTimeline.toTime(Ti_previous);
      t_current   = (float) gadgetTimeline.toTime(Ti_current);
    }
    else
    {
      //Reduce the last parts on the host
      tnext.d2h();
      // Phase 2: the sync point is an INTEGER minimum over Ti_endstep. Every particle that should
      // meet here holds the identical tick, so the meeting is exact -- the float32 version missed
      // it by 1 ULP and burned a whole system step on the miss (T24/T38).
      Ti_previous = Ti_current;
      Ti_current  = tnext[0];
      for (int i = 1; i < NBLOCK_REDUCE ; i++)
      {
          Ti_current = std::min(Ti_current, tnext[i]);
      }
      // The derived scale factors, for everything downstream that wants one.
      t_previous = (float) gadgetTimeline.toTime(Ti_previous);
      t_current  = (float) gadgetTimeline.toTime(Ti_current);
      // C-D-23 invariant probe (GADGET_HIP_INVARIANTS=1): t_current must be EXACTLY the minimum
      // over every particle's bodies_time[].y. This contract is worth checking rather than
      // assuming: its own probe notes that tnext[] had already been caught holding denormal bit
      // patterns that are not any particle's time -- i.e. this reduction (get_Tnext in
      // timestep.cu) has returned a wrong answer before. It is one of the four unsynchronised
      // warp-reduction tails fixed in 9986e8f, so this is the direct test of that fix.
      {
        static const bool invOn = (getenv("GADGET_HIP_INVARIANTS") != NULL);
        if (invOn)
        {
          tree.bodies_time.d2h();
          int trueMin = tree.bodies_time[0].y;
          int nAtMin  = 0;
          for (int i = 1; i < tree.n; i++) trueMin = std::min(trueMin, tree.bodies_time[i].y);
          for (int i = 0; i < tree.n; i++) if (tree.bodies_time[i].y == trueMin) nAtMin++;
          fprintf(stderr,
            "[INVARIANT] C-D-23 active-set argmin: %s  reduced=%d true_min=%d "
            "(particles at min: %d/%d)\n",
            (Ti_current == trueMin) ? "OK" : "VIOLATED", Ti_current, trueMin,
            nAtMin, tree.n);
        }
      }

      // BUG4-DIAG: unconditional per-call tnext dump (gated to a small call-index window)
      // to compare a known-good iteration against the crash iteration directly.
      {
        static int bug4CallIdx = -1;
        bug4CallIdx++;
        if (gadget_hip_debug_log && bug4CallIdx >= 68 && bug4CallIdx <= 74) {
          fprintf(stderr, "[BUG4-ITERDUMP] callIdx=%d tree.n=%d t_previous=%.9g t_current=%.9g bodies_time_devptr=%p bodies_time_size=%d\n",
                  bug4CallIdx, tree.n, t_previous, t_current, tree.bodies_time.get_devMem(), tree.bodies_time.get_size());
          #define BUG4_MAP(name, elemsize) \
            fprintf(stderr, "[BUG4-MAP] callIdx=%d %-22s ptr=%p bytes=%ld end=%p\n", bug4CallIdx, #name, \
                    tree.name.get_devMem(), (long)tree.name.get_size()*(elemsize), \
                    (char*)tree.name.get_devMem() + (long)tree.name.get_size()*(elemsize))
          BUG4_MAP(bodies_pos, 16); BUG4_MAP(bodies_key, 16); BUG4_MAP(bodies_vel, 16);
          BUG4_MAP(bodies_acc0, 16); BUG4_MAP(bodies_acc1, 16); BUG4_MAP(bodies_time, 8);
          BUG4_MAP(bodies_ids, 8); BUG4_MAP(bodies_Ppos, 16); BUG4_MAP(bodies_Pvel, 16);
          BUG4_MAP(bodies_h, 4); BUG4_MAP(bodies_dens, 8); BUG4_MAP(bodies_forceSoftening, 4);
          BUG4_MAP(bodies_typeDevice, 4); BUG4_MAP(oriParticleOrder, 4);
          BUG4_MAP(level_list, 8); BUG4_MAP(n_children, 4); BUG4_MAP(node_bodies, 8);
          BUG4_MAP(leafNodeIdx, 4); BUG4_MAP(node_level_list, 4); BUG4_MAP(body2group_list, 4);
          BUG4_MAP(group_list, 8); BUG4_MAP(multipole, 16); BUG4_MAP(activeGrpList, 4);
          BUG4_MAP(active_group_list, 4); BUG4_MAP(activePartlist, 4); BUG4_MAP(ngb, 4);
          BUG4_MAP(interactions, 8); BUG4_MAP(boxSizeInfo, 16); BUG4_MAP(groupSizeInfo, 16);
          BUG4_MAP(boxCenterInfo, 16); BUG4_MAP(groupCenterInfo, 16); BUG4_MAP(groupMaxAccInfo, 4);
          BUG4_MAP(groupMaxSofteningInfo, 4);
          BUG4_MAP(parallelBoundaries, 16); BUG4_MAP(generalBuffer1, 4); BUG4_MAP(fullRemoteTree, 16);
          #undef BUG4_MAP
          fprintf(stderr, "[BUG4-MAP] callIdx=%d %-22s ptr=%p bytes=%ld end=%p\n", bug4CallIdx, "tnext",
                  this->tnext.get_devMem(), (long)this->tnext.get_size()*4, (char*)this->tnext.get_devMem() + (long)this->tnext.get_size()*4);
          for (int i = 0; i < NBLOCK_REDUCE; i++) {
            uint32_t bits;
            float v = tnext[i];
            memcpy(&bits, &v, sizeof(bits));
            fprintf(stderr, "[BUG4-ITERDUMP] callIdx=%d tnext[%d]=0x%08x (%.9g)\n", bug4CallIdx, i, bits, v);
          }
          // Sample bodies_time directly (before the reduction reads it, to see whether
          // the corruption is upstream of getTNext or specific to tnext itself).
          tree.bodies_time.d2h();
          tree.oriParticleOrder.d2h();
          int sampleIdx[6] = {0, 1, 1000, 16384, 32000, tree.n - 1};
          for (int k = 0; k < 6; k++) {
            int idx = sampleIdx[k];
            if (idx < 0 || idx >= tree.n) continue;
            int2 bt = tree.bodies_time[idx];
            uint32_t xb, yb;
            memcpy(&xb, &bt.x, sizeof(xb));
            memcpy(&yb, &bt.y, sizeof(yb));
            uint32_t orderVal = tree.oriParticleOrder[idx];
            fprintf(stderr, "[BUG4-ITERDUMP] callIdx=%d bodies_time[%d]=(x=0x%08x(%.9g), y=0x%08x(%.9g)) oriParticleOrder[%d]=%u (n=%d)\n",
                    bug4CallIdx, idx, xb, bt.x, yb, bt.y, idx, orderVal, tree.n);
          }
          // Also scan the ENTIRE oriParticleOrder array for any out-of-range entry.
          {
            int badCount = 0;
            int firstBad = -1;
            uint32_t firstBadVal = 0;
            for (int i = 0; i < tree.n; i++) {
              uint32_t v = tree.oriParticleOrder[i];
              if (v >= (uint32_t)tree.n) {
                if (badCount == 0) { firstBad = i; firstBadVal = v; }
                badCount++;
              }
            }
            fprintf(stderr, "[BUG4-ITERDUMP] callIdx=%d oriParticleOrder OOB scan: badCount=%d firstBadIdx=%d firstBadVal=%u\n",
                    bug4CallIdx, badCount, firstBad, firstBadVal);
          }
        }
      }

      // Phase 5 ticket 08 (PLAN.md): a real, reproducible failure mode found via this ticket's own
      // end-to-end runs (both the plain and zoom tests), root-caused down to its exact mechanism
      // but NOT to a specific fixable line: `tnext[]` (this reduction's own scratch buffer) was
      // observed holding a clean arithmetic progression of denormalized floats whose bit patterns
      // are small integers linear in block index (0x80000000 | ((i+1)*256)) -- the signature of a
      // GPU memory-aliasing bug (this codebase's shared/reused scratch-buffer pools, e.g.
      // generalBuffer1, letting some other kernel's integer-typed data bleed into tnext's own
      // memory), not of this reduction's own logic (which is a plain, correctly-sentineled
      // fminf() over real `bodies_time[i].y` values). Finding the exact overlapping allocation is
      // out of this ticket's scope (needs either a HIP memory sanitizer or a full scratch-pool
      // audit) -- deliberately left as a known, well-characterized open issue (LOG.md) rather than
      // guessed at further. This check turns the alternative -- t_current silently corrupting to a
      // denormal/NaN and the sim then hanging forever, indistinguishable from still-running work --
      // into a clear, fast, diagnosable failure, the same instinct as Gadget-2's own real
      // `endrun(818)` guard against an unrepresentable timestep (timestep.c ~528-541).
      // C-D-14: this tested `<` only, so the exact freeze -- t_current == t_previous, the global
      // minimum end-time failing to move, which is precisely C-D-07's case and Gadget-2's
      // endrun(818) -- passed it silently and the run hung instead of aborting.
      // A plain `<=` is WRONG here, and measurably so: on the first call no timestep has been
      // assigned yet, every bodies_time[].y is still 0, so the reduction legitimately yields
      // t_current == t_previous == 0 and `<=` aborts a healthy run before iteration 0 (observed on
      // the A1 Plummer run). Equality is a freeze only once the clock has actually started moving.
      // The startup window is not left unguarded: a non-advancing step there is caught directly,
      // per particle, by the C-D-14 check on compute_dt's own output (timestep.cu, g_t7_*), which
      // is the stronger of the two tests. The two are complementary.
      static bool clockHasAdvanced = false;
      const bool  clockFrozen = clockHasAdvanced && (t_current == t_previous);
      if (t_current < t_previous || t_current != t_current || clockFrozen)
      {
        fprintf(stderr, "FATAL: t_current failed to advance or is corrupted (t_previous=%.9g -> "
                         "t_current=%.9g). Either the global minimum end-of-step did not move "
                         "(C-D-07/C-D-14, Gadget-2's endrun(818) condition) or tnext[] was "
                         "corrupted by the known scratch-buffer aliasing issue (PLAN.md/LOG.md). "
                         "Aborting instead of hanging.\n", t_previous, t_current);
        // BUG4-DIAG: raw dump of every tnext[] entry (bit pattern + float) at the
        // moment of detection, plus tree.n/n_groups for context.
        fprintf(stderr, "[BUG4-DUMP] tree.n=%d tree.n_groups=%d NBLOCK_REDUCE=%d\n",
                tree.n, tree.n_groups, NBLOCK_REDUCE);
        for (int i = 0; i < NBLOCK_REDUCE; i++) {
          uint32_t bits;
          float v = tnext[i];
          memcpy(&bits, &v, sizeof(bits));
          fprintf(stderr, "[BUG4-DUMP] tnext[%d]=0x%08x (%.9g)\n", i, bits, v);
        }
        ::exit(1);
      }
      if (t_current > t_previous) clockHasAdvanced = true;
    }
  #else
    static int temp = 0;
    t_previous =  t_current;
    if(temp > 0) t_current  += timeStep;
    else	      temp 		 = 1;
  #endif


    //Set valid list to zero, TODO should we act on this comment?

    // Phase 5 ticket 05 (PLAN.md): int comovingIntegrationOn passed by address like every other
    // scalar in this legacy wrapper's set_args() convention -- must be a non-const local, not a
    // literal/temporary (see LOG.md's own note on the const-pointer set_args() compile error this
    // caught during ticket 04).
    int predictComovingFlag = haveGadgetComovingTables ? 1 : 0;
  // MEMORY (PLAN 4.1): bodies_pos is bound to bodies_Ppos, so the drift happens IN PLACE -- which
  // is what Gadget-2 does (move_particles drifts P[].Pos itself; there is no second predicted
  // array). This is a pointer rebinding ONLY: no kernel logic changes, because predict reads
  // pos[idx] into a register before writing pPos[idx] at the SAME index, and correct's
  // `pos[idx] = pPos[idx]` becomes a harmless self-assignment. No __restrict__ anywhere in
  // timestep.cu, so the aliasing is well-defined.
  //
  // The premise was verified, not assumed (GADGET_HIP_INVARIANTS=1, dup_check_pos): bodies_pos and
  // bodies_Ppos are bit-identical at the start of EVERY step, on both the synchronous and the
  // individual-timestep paths. That only became true with the T6/C-D-01 fix -- under the old
  // Ti_begstep-anchored drift the un-drifted anchor position was load-bearing and this would have
  // been wrong.
  //
  // NOTE bodies_Pvel is NOT removable the same way: correct_particles reads pVel[unsortedIdx] (a
  // TRANSLATED index) while writing vel[idx], so a read-only snapshot is what keeps that race-free,
  // and computeEnergyPreKick needs the pre-kick velocity after vel has been overwritten.
    predictParticles.set_args(0, &tree.n, &t_current, &t_previous, tree.bodies_Ppos.p(), tree.bodies_vel.p(),
                    tree.bodies_acc0.p(), tree.bodies_time.p(), tree.bodies_Ppos.p(), tree.bodies_Pvel.p(),
                    &predictComovingFlag, gadgetDriftTable.p(), gadgetGravKickTable.p(),
                    &gadgetLogTimeBegin, &gadgetDPerTick, &Ti_previous, &gadgetLogTimeMax);
    predictParticles.setWork(tree.n, 128);
    predictParticles.execute2(execStream->s());

} //End predict


void octree::setActiveGrpsFunc(tree_structure &tree)
{
  //Moved to compute_properties
}

void octree::direct_gravity(tree_structure &tree)
{
    std::vector<size_t> localWork  = {256, 1};
    std::vector<size_t> globalWork = {static_cast<size_t>(256 * ((tree.n + 255) / 256)), 1};

    // C-C-03: the direct path now uses the SAME spline kernel and the SAME per-particle
    // ForceSoftening as the tree walk, plus minimum imaging under PERIODIC. It previously used a
    // flat Plummer with the CLI `eps` -- a different kernel shape, a length scale 2.8x off, and no
    // periodicity -- which silently corrupted every tree-vs-direct comparison.
    // Shared tile carries a softening per source alongside the position.
    float dg_boxSize = 0.0f;   // <= 0 selects the non-periodic branch in the kernel
#ifdef PERIODIC
    dg_boxSize = pm_get_periodic_boxsize();   // the value the walk itself was set up with
#endif
    directGrav.set_args(sizeof(float4)*256 + sizeof(float)*256,
                        tree.bodies_acc0.p(), tree.bodies_Ppos.p(), tree.bodies_Ppos.p(),
                        &tree.n, &tree.n,
                        tree.bodies_forceSoftening.p(), tree.bodies_forceSoftening.p(),
                        &dg_boxSize);
    directGrav.setWork(globalWork, localWork);

    // iterate_once() unconditionally reads back elapsed time on startLocalGrav/endLocalGrav
    // (CU_SAFE_CALL(hipEventElapsedTime(...))) regardless of which gravity path ran; only
    // approximate_gravity() used to record these events, so --direct runs hit "invalid resource
    // handle" reading unrecorded events. Record them here too so both paths behave the same way.
    hipEventRecord(startLocalGrav, gravStream->s());
    directGrav.execute2(gravStream->s());
    hipEventRecord(endLocalGrav, gravStream->s());
}

// Phase 3: Springel(2005)/Gadget-2 MAC input -- one conservative |a_old| per group (the max over
// the group's members), computed host-side from last step's bodies_acc0. Groups are contiguous
// blocks of the CURRENT (this-iteration, freshly SFC-sorted) particle order -- groupSizeInfo is
// built from build()/compute_properties() against whatever order sort_bodies() just produced this
// same iteration (bodies_Ppos/bodies_ids/bodies_h). This function's start/nb_i decode
// (groupSizeInfo[g].w, CRITMASK/INVCMASK, node_specs.h) replicates the kernel's own group-layout
// decode to guarantee IT can't silently desync from the group layout.
//
// Ticket 23 fix: the comment this replaced claimed "bodies_acc0 already uses [the same SFC-sorted
// order]" and indexed it directly with the current-order start/i -- that claim is false on the
// normal per-iteration path. Ticket 12 already established (timestep.cu, sort_bodies_gpu.cpp:206-
// 223, the "!doFullShuffle" branch every ordinary step takes) that bodies_pos/bodies_vel/bodies_acc0
// /bodies_acc1/bodies_time are NOT among the buffers sort_bodies() resorts every step (only
// bodies_Ppos/bodies_ids/bodies_h are) -- they stay in whatever order they were left by the LAST
// correct_particles() call, i.e. the PREVIOUS iteration's current-order, not this iteration's. T12's
// own resolution explicitly flagged this as a latent risk for "any future kernel added to the
// per-step pipeline that reads bodies_Ppos[idx]/bodies_vel[idx] before correct_particles has run that
// iteration" -- this function is exactly that future kernel for bodies_acc0, added later by the
// (until Ticket 23, never runtime-exercised) MAC_SPRINGEL work, and it hit precisely the bug T12
// predicted: reading bodies_acc0[start+i] with a current-order index silently attributes a
// DIFFERENT particle's old acceleration to this group whenever the SFC order changed since
// bodies_acc0 was last written (by default, sort_bodies()/rebuild_tree_rate=1 means this is every
// single iteration, not a rare event) -- a real, silent, per-group MAC-input corruption bug on the
// same class already found and fixed by Tickets 11/12 elsewhere in this codebase.
//
// Fix: translate each group member's current-order index through tree.oriParticleOrder (the exact
// "unsorted[]" permutation correct_particles()/compute_energies() already use for this identical
// buffer-ordering mismatch, timestep.cu/gpu_iterate.cpp:2408-2412) to get the OLD-order index that
// bodies_acc0 is actually still indexed by, before reading it. oriParticleOrder is guaranteed fresh
// by this point: sort_bodies() (which (re)computes it unconditionally, sort_bodies_gpu.cpp:183-184,
// before the doFullShuffle branch) always runs earlier in this same iteration, before
// approximate_gravity()/computeGroupMaxAccel() are reached.
//
// Deliberately host-side and unconditional-copy for a first, correctness-first implementation --
// see AMD_PORT_PLAN.md Phase 3 for the note that this should move to a GPU reduction kernel
// (mirroring compute_propertiesD.cu's existing per-group AABB reduction) once profiling shows it
// matters; only called when built with _MAC_SPRINGEL_.
void octree::computeGroupMaxAccel(tree_structure &tree)
{
  // T45: the Springel MAC's aold must include the long-range force, as Gadget's does
  // (gravtree.c:309-311). With the cadence ON, GravPM is not in bodies_acc0, so the magnitude is
  // formed on the device by gadget_aold_mag -- see that kernel for why it is not done here.
  const bool aoldWithPM = s_pmCadenceActive && s_gravPMx != nullptr && tree.n > 0;
  static const bool t45KeepAcc0 = (getenv("GADGET_HIP_T45_KEEP_ACC0_D2H") != NULL);
  if (aoldWithPM)
  {
    if (tree.bodies_aoldMag.get_size() < (size_t) tree.n)
      tree.bodies_aoldMag.cmalloc(tree.n, true);
    // Not const: set_args stores a non-const void* and dereferences it at launch.
    GCarrier aoldG = GCarrier::just_G(
        (!useDirectGravity && haveGadgetParams && gadgetParams.G > 0.0) ? (float) gadgetParams.G : 1.0f);
    // bodies_acc0 and the output are both in the unpermuted (original) space; bodies_ids is in the
    // post-sort space. See include/gadget_index_spaces.h and buffer_registry.h.
    gadgetAoldMag.set_args(0, &tree.n,
                           (void *) tree.bodies_acc0.as_const<OriginalOrder>(),
                           (void *) tree.oriParticleOrder.as_const<CurrentOrder>(),
                           (void *) tree.bodies_ids.as_const<CurrentOrder>(),
                           (void *) &s_gravPMx, (void *) &s_gravPMy,
                           (void *) &s_gravPMz, &aoldG,
                           (void *) tree.bodies_aoldMag.as<OriginalOrder>());
    gadgetAoldMag.setWork(tree.n, 128);
    gadgetAoldMag.execute2(execStream->s());
    execStream->sync();
    tree.bodies_aoldMag.d2h();
    static bool s_aoldReported = false;
    if (!s_aoldReported)
    {
      s_aoldReported = true;
      fprintf(stderr, "[T45-AOLD-PM] %s (G = %.6f)\n",
              t45KeepAcc0 ? "kernel runs but the reduction uses the OLD tree-only vector "
                            "(GADGET_HIP_T45_KEEP_ACC0_D2H=1)"
                          : "Springel aold includes the long-range force",
              (double) aoldG.v);
    }
  }
  else if (s_pmCadenceActive)
  {
    // T44 item 3: a feature that declines to engage must say so.
    static bool s_aoldOffReported = false;
    if (!s_aoldOffReported)
    {
      s_aoldOffReported = true;
      fprintf(stderr, "[T45-AOLD-PM] OFF -- the cadence is on but GravPM is not in aold; "
                      "the MAC will open more nodes than Gadget's\n");
    }
  }
  // The vector is only needed on the host for the non-PM path; the device path reads acc0 on the
  // device. GADGET_HIP_T45_KEEP_ACC0_D2H=1 restores the unconditional transfer for A/B work.
  if (!aoldWithPM || t45KeepAcc0) tree.bodies_acc0.d2h();
  tree.groupSizeInfo.d2h();
  tree.oriParticleOrder.d2h();

  // T24: diagnostic instrumentation for the MAC_SPRINGEL N=32768/PMGRID64 blowup (iter~235).
  // Host-side fprintf only (device printf unreliable near a blowup/fault per this project's own
  // repeated experience). Env-gated window plus an always-on huge-value safety net so we don't
  // miss the actual trigger iteration even if our estimate of the window is slightly off.
  static bool t24_init = false;
  static int t24_lo = 220, t24_hi = 245;
  static float t24_prevGlobalMax = -1.0f;
  if (!t24_init) {
    t24_init = true;
    const char *e = getenv("GADGET_HIP_T24_WINDOW");
    if (e) sscanf(e, "%d,%d", &t24_lo, &t24_hi);
  }
  const bool t24_windowIter = (iter >= t24_lo && iter <= t24_hi);
  if (t24_windowIter) tree.bodies_ids.d2h();

  // T4 (contract C-B-02): Gadget-2 forms `aold` from the un-G'd acceleration -- OldAcc is computed
  // at gravtree.c:317, BEFORE `GravAccel *= All.G` at :325-328 -- so at forcetree.c:1279 both the
  // node `mass` and `aold` are in G-free units. This port G-scales the entire accumulated
  // acceleration buffer (see the pm_scale_buffer call above) before it is copied into acc0 by
  // correct_particles (timestep.cu), so the value reduced below carries a factor of G that the
  // kernel's node mass `mJ` (a bare summed multipole mass) does not. The walk therefore behaves as
  // though ErrTolForceAcc were ErrTolForceAcc*G -- 0.005*43007.1 ~= 215 for this project's own
  // cosmological parameter file. Dividing G back out here restores Gadget-2's own convention at the
  // single point where the MAC input is formed, rather than touching the per-interaction kernel.
  // Env-gated (GADGET_HIP_T4_FIX_G=1) so corrected and uncorrected walks come from one binary; the
  // gate mirrors the pm_scale_buffer call's own condition, so a G=1 run is bit-unchanged either way.
  // T1 (contract C-B-01): which reduction forms the group's shared |a_old|.
  // Gadget-2 uses each particle's OWN OldAcc (forcetree.c:1156). A group must collapse that to one
  // value, and the direction matters: the test opens when `m*l^2 > r^4 * ErrTolForceAcc * aold`, so a
  // LARGER aold raises the right-hand side and opens FEWER nodes -- a coarser force. Taking the MAX
  // therefore grants every member the error budget of its most strongly accelerated member and
  // systematically under-resolves the low-acceleration tail; MIN is the conservative direction.
  // (The kernel's own comment at dev_approximate_gravity_warp_new.cu:1058-1062 asserts the opposite,
  // carried over from the geometric MAC's group-AABB generalisation where it does hold.)
  //   GADGET_HIP_T1_AOLD_REDUCE = 0 max (legacy) | 1 min | 2 mean
  // A group whose reduced value is 0 falls back to the geometric criterion, as before.
  static int t1_reduce = -1;
  if (t1_reduce < 0)
  {
    const char *e = getenv("GADGET_HIP_T1_AOLD_REDUCE");
    t1_reduce = e ? atoi(e) : 1;   // default: min (see ticket T1; max is the defect)
    fprintf(stderr, "[T1-AOLD-REDUCE] group |a_old| = %s\n",
            t1_reduce == 1 ? "min (conservative)" : t1_reduce == 2 ? "mean" : "max (legacy)");
  }

  static bool  t4g_init = false;
  static float t4g_aoldScale = 1.0f;
  if (!t4g_init)
  {
    t4g_init = true;
    // Default ON as of the T4 resolution: the G-inflated `aold` was a unit error, not a tuning
    // knob. GADGET_HIP_T4_FIX_G=0 restores the old (buggy) behaviour for A/B work.
    const char *e = getenv("GADGET_HIP_T4_FIX_G");
    const bool  want = e ? (atoi(e) != 0) : true;
    if (want && !useDirectGravity && haveGadgetParams && gadgetParams.G > 0.0)
      t4g_aoldScale = (float) (1.0 / gadgetParams.G);
    fprintf(stderr, "[T4-FIX-G] aold scale = %.9g (G = %.6f)\n",
            t4g_aoldScale, haveGadgetParams ? gadgetParams.G : 1.0);
  }

  float t24_globalMax2 = 0.0f;
  int   t24_globalMaxGroup = -1;
  uint  t24_globalMaxOldIdx = 0;
  uint  t24_globalMaxStart = 0, t24_globalMaxNbi = 0;

  // C-B-03/T31 diagnostic: this host loop is the only Springel-specific per-particle host work.
  // Print its actual extent so a stall here is attributable rather than inferred.
  unsigned long long t31_visited = 0;
  const double t31_t0 = get_time();
  static bool t31_reported = false;
  static const bool t31_verbose = (getenv("GADGET_HIP_T31_STATS") != NULL);
  // This host loop walks the GROUP space on the outside and the PARTICLE spaces on the inside, which
  // is exactly the mixture T23 Bug 2 got wrong: it read bodies_acc0 (unpermuted) with the current
  // slot index, so every group's MAC input silently used a different set of particles. The spaces are
  // named here for the same reason they are named in the kernels.
  for (GroupIdx g = {0}; g.v < (unsigned int) tree.n_groups; g.v++)
  {
    const int groupData = *reinterpret_cast<const int*>(&tree.groupSizeInfo[g.v].w);
    const uint start = groupData & CRITMASK;
    const uint nb_i   = ((groupData & INVCMASK) >> CRITBIT) + 1;
    if (t31_verbose && !t31_reported && g.v < 3)
      fprintf(stderr, "[T31] group %u: raw=0x%08x start=%u nb_i=%u  (n_groups=%d, tree.n=%d, CRITBIT=%d)\n",
              g.v, (unsigned) groupData, start, nb_i, tree.n_groups, tree.n, CRITBIT);
    t31_visited += nb_i;

    float maxAcc2 = 0.0f;
    float minAcc2 = 3.4e38f;   // T1
    double sumAcc = 0.0;       // T1
    float maxSoftening = 0.0f;  // T27: per-group max Gadget-2 ForceSoftening, softening-based bJ floor input
    for (uint i = 0; i < nb_i; i++)
    {
      // start + i is a CURRENT slot; oldIdx is the matching ORIGINAL slot. Both are particle
      // spaces, and neither is the group index g driving the loop.
      const CurrentIdx slot = { start + i };
      const uint oldIdx = tree.oriParticleOrder[slot.v];
      // T45: with the cadence on this is |tree + PM| (gadget_aold_mag); otherwise PM is already
      // inside acc0 and the vector is read directly, exactly as before.
      float a2;
      if (aoldWithPM && !t45KeepAcc0)
      {
        const float m = tree.bodies_aoldMag[oldIdx];
        a2 = m * m;
      }
      else
      {
        const real4 a = tree.bodies_acc0[oldIdx];
        a2 = a.x*a.x + a.y*a.y + a.z*a.z;
      }
      if (a2 > maxAcc2) maxAcc2 = a2;
      if (a2 < minAcc2) minAcc2 = a2;                 // T1
      sumAcc += sqrt((double) a2);                    // T1
      if (a2 > t24_globalMax2) {
        t24_globalMax2 = a2;
        t24_globalMaxGroup = (int) g.v;
        t24_globalMaxOldIdx = oldIdx;
        t24_globalMaxStart = start;
        t24_globalMaxNbi = nb_i;
      }
      // T27: bodies_forceSoftening is likewise never resorted by sort_bodies()'s default
      // per-iteration path (confirmed via sort_bodies_gpu.cpp) -- same oldIdx translation as
      // bodies_acc0 above is required, not a new lookup pattern.
      // CORRECTED: bodies_forceSoftening is written by recomputeSoftening() AFTER sort_bodies
      // (gpu_iterate.cpp:592), indexed by bodies_type, which is itself rebuilt from the resorted
      // bodies_ids at :2120. It is therefore in CURRENT order, and translating it through
      // oriParticleOrder here reads a DIFFERENT particle's softening. T27 added that translation
      // by analogy with bodies_acc0 -- which genuinely IS original-order, one line above -- but
      // without checking where this buffer is written. Verified inert to date: every IC in this
      // project is single-species, so all forceSoftening values are equal and an A/B of the two
      // forms is bit-identical (Etot -0.2007262861 either way).
      const float sft = tree.bodies_forceSoftening[start + i];
      if (sft > maxSoftening) maxSoftening = sft;
    }
    float t1_aold;                                     // T1
    if      (t1_reduce == 1) t1_aold = (nb_i > 0) ? sqrtf(minAcc2) : 0.0f;
    else if (t1_reduce == 2) t1_aold = (nb_i > 0) ? (float)(sumAcc / (double) nb_i) : 0.0f;
    else                     t1_aold = sqrtf(maxAcc2);
    tree.groupMaxAccInfo[g.v] = t1_aold * t4g_aoldScale;
    tree.groupMaxSofteningInfo[g.v] = maxSoftening;
  }
  if (t31_verbose && !t31_reported)
  {
    // Which MAC branch will the walk actually take? The kernel picks springel iff
    // groupMaxAcc > 0; on the bootstrap iteration bodies_acc0 is supposed to be all-zero so it
    // falls back to the geometric test. Verify that rather than assume it.
    unsigned long long nPos = 0; float gmax = 0.0f;
    for (int g = 0; g < tree.n_groups; g++) {
      const float v = tree.groupMaxAccInfo[g];
      if (v > 0.0f) nPos++;
      if (v > gmax) gmax = v;
    }
    fprintf(stderr, "[T31] computeGroupMaxAccel: %d groups, %llu particle-visits, %.3f s "
                    "(tree.n=%d -- visits should be ~= tree.n)\n"
                    "[T31] groupMaxAcc > 0 in %llu / %d groups (max=%.6g) -> walk takes %s branch\n",
            tree.n_groups, t31_visited, get_time() - t31_t0, tree.n,
            nPos, tree.n_groups, gmax, nPos ? "SPRINGEL" : "geometric-bootstrap");
    t31_reported = true;
  }

  const float t24_globalMax = sqrtf(t24_globalMax2);
  const bool t24_hugeJump = (t24_prevGlobalMax > 0.0f) &&
                             (t24_globalMax > 1000.0f * t24_prevGlobalMax || t24_globalMax > 1e6f);
  if (gadget_hip_debug_log && (t24_windowIter || t24_hugeJump)) {
    ullong t24_pid = 0;
    if (t24_windowIter && t24_globalMaxGroup >= 0) {
      // bodies_ids is resorted every iteration along with bodies_Ppos/bodies_h (current sorted
      // order); the group's own start/i indices (from groupSizeInfo) are already in that same
      // current-sorted order, so index bodies_ids directly at start+i (the slot that produced the
      // max), NOT via oriParticleOrder (which only applies to the acc0 buffer's own, different,
      // stale-cross-iteration ordering).
      for (uint i = 0; i < t24_globalMaxNbi; i++) {
        const uint oldIdx = tree.oriParticleOrder[t24_globalMaxStart + i];
        if (oldIdx == t24_globalMaxOldIdx) { t24_pid = tree.bodies_ids[t24_globalMaxStart + i]; break; }
      }
    }
    fprintf(stderr, "[T24-GROUPMAX] iter=%d n_groups=%d globalMax=%.6g (prevMax=%.6g ratio=%.6g) "
                     "group=%d start=%u nb_i=%u oldIdx=%u pid=%llu%s\n",
            iter, tree.n_groups, t24_globalMax, t24_prevGlobalMax,
            t24_prevGlobalMax > 0.0f ? t24_globalMax / t24_prevGlobalMax : 0.0,
            t24_globalMaxGroup, t24_globalMaxStart, t24_globalMaxNbi, t24_globalMaxOldIdx,
            (unsigned long long) t24_pid, t24_hugeJump ? " HUGE-JUMP" : "");
  }
  t24_prevGlobalMax = t24_globalMax;

  tree.groupMaxAccInfo.h2d();
  tree.groupMaxSofteningInfo.h2d();
}

// Bug fix (Phase 5 ticket 06, LOG.md): see octree.h's declaration for the full rationale --
// records the STABLE particle-ID -> type association from the original (pre-sort) IC load order,
// so recomputeSoftening() can rebuild tree.bodies_type from the always-correctly-reordered
// tree.bodies_ids every rebuild instead of trusting bodies_type's own (never reordered) index.
void octree::setGadgetTypeMap(const std::vector<ullong> &ids, const std::vector<int> &types)
{
  gadgetIdToType.clear();
  gadgetIdToType.reserve(ids.size());
  for (size_t i = 0; i < ids.size(); ++i)
    gadgetIdToType[ids[i]] = types[i];

  // T35: the same association as a flat table, so the per-rebuild refresh can run on the device.
  // Only built when the id space is dense enough to index directly -- Gadget numbers particles
  // 1..N, but an unusual IC need not, and a sparse id space would blow the allocation up. The
  // threshold is deliberately generous (2x the particle count plus slack): beyond it the host+map
  // path still runs and is still correct, just slower.
  gadgetTypeByIdOk = false;
  gadgetTypeById.clear();
  for (int t = 0; t < 6; ++t) gadgetTypeUsed[t] = 0;
  for (size_t i = 0; i < types.size(); ++i)
    if (types[i] >= 0 && types[i] < 6) gadgetTypeUsed[types[i]] = 1;
  ullong maxId = 0;
  for (size_t i = 0; i < ids.size(); ++i) maxId = std::max(maxId, ids[i]);
  const ullong cap = (ullong) ids.size() * 2ull + 1024ull;
  bool typesFit = true;
  for (size_t i = 0; i < types.size(); ++i)
    if (types[i] < 0 || types[i] > 255) { typesFit = false; break; }

  if (!ids.empty() && maxId <= cap && typesFit)
  {
    gadgetTypeById.assign((size_t) maxId + 1, (unsigned char) 0);
    for (size_t i = 0; i < ids.size(); ++i)
      gadgetTypeById[(size_t) ids[i]] = (unsigned char) types[i];
    gadgetTypeByIdMax = maxId;
    gadgetTypeByIdOk  = true;
    fprintf(stderr, "[T35] device softening refresh ENABLED (max id %llu, table %.1f MB)\n",
            (unsigned long long) maxId, (double)(maxId + 1) / (1024.0 * 1024.0));
  }
  else
  {
    fprintf(stderr, "[T35] device softening refresh DISABLED (max id %llu vs cap %llu, typesFit=%d)"
                    " -- falling back to the host (id -> type) map, correct but slower\n",
            (unsigned long long) maxId, (unsigned long long) cap, (int) typesFit);
  }
}

// Phase 5 tickets 03/04 (PLAN.md): see octree.h's declarations for the full rationale.
void octree::setGadgetParams(const GadgetParams &params)
{
  gadgetParams        = params;
  haveGadgetParams    = true;
  // T56: with an output list, the first wanted epoch is the list's own first in-range entry, not
  // TimeOfFirstSnapshot -- Gadget ignores TimeOfFirstSnapshot entirely when OutputListOn is set
  // (find_next_outputtime(), run.c:250-271, never reads it on that branch). An entry exactly at
  // TimeBegin therefore fires a snapshot at the IC epoch, which is what Gadget does too.
  nextGadgetSnapTime  = (params.OutputListOn != 0 && !params.OutputListTimes.empty())
                      ? params.OutputListTimes.front()
                      : params.TimeOfFirstSnapshot;
  gadgetSnapshotCount = 0;
}

// Phase 5 ticket 08 (PLAN.md): see octree.h's own declaration comment for why this exists.
// Snapshot cadence, matched to Gadget-2. Gadget writes its output inside
// find_next_sync_point_and_drift() (run.c:18-22) -- i.e. AFTER drifting every particle to the
// output time but BEFORE compute_accelerations() and before advance_and_find_timesteps()'s kick.
// The written state is therefore post-drift positions with PRE-KICK velocities. The port wrote
// from the end of iterate_once, after correct() had already applied the kick, so every snapshot
// carried velocities half a step ahead of the reference's. Visible directly: the port's
// snapshot_000 had RMS(vy,vz) = 2.7e-3 on the Zel'dovich plane wave where the IC and Gadget-2's
// own snapshot_000 are both exactly 0.
//
// preKick=true reads the same buffer pair the pre-kick energy diagnostic already uses:
// bodies_Ppos (drifted to t_current, current sorted order) and bodies_Pvel (never resorted, so
// reached through oriParticleOrder). This must be called BEFORE correct(), which resets
// oriParticleOrder to identity and so destroys that translation.
// Deliverable 1 of PLAN section 4.1: device-buffer inventory.
//
// The production target is 512^3 (1.34e8 particles) minimum and ideally ~3e8, in 128 GB shared.
// An older measurement put the port at ~38-40 GB for 512^3 against Gadget-2's 16 GB -- roughly
// 290 B/particle against ~120 -- which extrapolates to ~87 GB at 3e8 and leaves little headroom.
// The requirement is not "optimise now" but "do not silently inflate", so this reports the actual
// per-buffer footprint and the derived bytes/particle, which can then be tracked as a regression
// number. Walks the named members rather than instrumenting the allocator, so it cannot perturb
// allocation behaviour. Zero cost unless GADGET_HIP_MEMREPORT is set.
//
// Note this is also a correctness surface, not only a performance one: Bug 5 (node_level_list
// overflow), Bug 8 (tempmem undersized at NLEAF=1) and Bug 9 (int32 overflow past ~53.7M nodes)
// were all wrong-answer or crash defects in buffer sizing.
void octree::reportDeviceMemory(const char *stage)
{
  static int on = -1;
  if (on < 0) { const char *e = getenv("GADGET_HIP_MEMREPORT"); on = e ? atoi(e) : 0; }
  if (!on) return;

  struct Row { const char *name; size_t n, esz; };
  std::vector<Row> rows;
  tree_structure &tree = localTree;
#define MEMREP(b) rows.push_back(Row{#b, (size_t) tree.b.get_size(), sizeof(*tree.b.raw_p())})
  MEMREP(bodies_pos);
  MEMREP(bodies_key);
  MEMREP(bodies_vel);
  MEMREP(bodies_acc0);
  MEMREP(bodies_acc1);
  MEMREP(bodies_time);
  MEMREP(bodies_ids);
  MEMREP(bodies_Ppos);
  MEMREP(bodies_Pvel);
  MEMREP(bodies_h);
  MEMREP(bodies_dens);
  MEMREP(bodies_forceSoftening);
  MEMREP(bodies_typeDevice);
  MEMREP(oriParticleOrder);
  MEMREP(level_list);
  MEMREP(n_children);
  MEMREP(node_bodies);
  MEMREP(leafNodeIdx);
  MEMREP(node_level_list);
  MEMREP(body2group_list);
  MEMREP(group_list);
  MEMREP(multipole);
  MEMREP(activeGrpList);
  MEMREP(active_group_list);
  MEMREP(activePartlist);
  MEMREP(ngb);
  MEMREP(interactions);
  MEMREP(boxSizeInfo);
  MEMREP(groupSizeInfo);
  MEMREP(boxCenterInfo);
  MEMREP(groupCenterInfo);
  MEMREP(groupMaxAccInfo);
  MEMREP(groupMaxSofteningInfo);
  MEMREP(cellSizeInfo);
  MEMREP(cellCenterInfo);   // C-A-01(b); size 1 in a geometric-MAC build, n_nodes in a Springel one
  MEMREP(parallelBoundaries);
  MEMREP(generalBuffer1);
  MEMREP(fullRemoteTree);
#undef MEMREP
  // These two live on the octree, not on tree_structure.
#define MEMREPO(b) rows.push_back(Row{#b, (size_t) this->b.get_size(), sizeof(*this->b.raw_p())})
  MEMREPO(gadgetDriftTable);
  MEMREPO(gadgetGravKickTable);
  MEMREPO(devMemRMIN);
  MEMREPO(devMemRMAX);
  MEMREPO(devMemCounts);
  MEMREPO(devMemCountsx);
#undef MEMREPO

  size_t total = 0;
  for (size_t i = 0; i < rows.size(); i++) total += rows[i].n * rows[i].esz;
  std::sort(rows.begin(), rows.end(),
            [](const Row &a, const Row &b){ return a.n*a.esz > b.n*b.esz; });

  // Separate buffers that scale with the particle count from ones that do not. Extrapolating a
  // fixed-size buffer per particle is meaningless, and generalBuffer1 in particular is pinned to a
  // GPU-occupancy floor (treeWalkStackSize = 2*(LMEM_STACK_SIZE*NTHREAD + LMEM_EXTRA_SIZE)*nBlocks)
  // at small N, then grows as ~13.2*n above the crossover. Classify by element count relative to n:
  // real per-particle/per-node/per-group buffers all sit within ~64x of n.
  size_t fixedBytes = 0, scalingBytes = 0;
  for (size_t i = 0; i < rows.size(); i++)
  {
    const size_t b = rows[i].n * rows[i].esz;
    const bool scales = tree.n > 0 && rows[i].n <= (size_t) tree.n * 64;
    if (scales) scalingBytes += b; else fixedBytes += b;
  }
  const double perPart = tree.n > 0 ? (double) scalingBytes / (double) tree.n : 0.0;
  fprintf(stderr, "[MEMREPORT] === %s ===  n=%d  n_nodes=%d  n_groups=%d\n",
          stage, tree.n, tree.n_nodes, tree.n_groups);
  fprintf(stderr, "[MEMREPORT] total %.3f GB   scaling %.3f GB (%.1f B/particle)   fixed %.3f GB\n",
          total / 1073741824.0, scalingBytes / 1073741824.0, perPart, fixedBytes / 1073741824.0);
  fprintf(stderr, "[MEMREPORT] model: fixed + %.1f B/part  ->  512^3 %.1f GB,  3e8 %.1f GB"
                  "   (Gadget-2 ~120 B/part -> 16 / 36 GB)\n",
          perPart,
          (fixedBytes + perPart * 134217728.0) / 1073741824.0,
          (fixedBytes + perPart * 3.0e8) / 1073741824.0);
  int unregistered = 0;
  for (size_t i = 0; i < rows.size(); i++)
  {
    const size_t b = rows[i].n * rows[i].esz;
    if (b == 0) continue;
    const BufferSpec *spec = bufferSpecFor(rows[i].name);
    if (!spec) unregistered++;
    fprintf(stderr, "[MEMREPORT]   %-24s %11zu x %2zu = %9.2f MB  %7.1f B/part  %-8s%s\n",
            rows[i].name, rows[i].n, rows[i].esz, b / 1048576.0,
            tree.n > 0 ? (double) b / tree.n : 0.0,
            spec ? bufferSpaceName(spec->space) : "UNREGISTERED",
            spec && spec->evidence == EV_INFERRED ? " (inferred)" : "");
  }
  // Recoverable capacity: bytes that exist only for features the default build does not use.
  {
    double quadDead = 0.0;
    for (size_t i = 0; i < rows.size(); i++)
    {
      const BufferSpec *sp = bufferSpecFor(rows[i].name);
      if (sp && sp->quadrupoleDeadFraction > 0.0f)
        quadDead += rows[i].n * rows[i].esz * (double) sp->quadrupoleDeadFraction;
    }
    if (quadDead > 0.0)
      fprintf(stderr, "[MEMREPORT] recoverable if quadrupole support is removed: %.2f MB now "
                      "(%.1f B/particle -> %.2f GB at 512^3)\n",
              quadDead / 1048576.0,
              tree.n > 0 ? quadDead / tree.n : 0.0,
              tree.n > 0 ? (quadDead / tree.n) * 134217728.0 / 1073741824.0 : 0.0);
  }

  // Coverage check: a buffer with no registry entry is a buffer whose index space nobody has
  // decided. That is exactly how C-D-03b got in, so make it loud rather than silent.
  if (unregistered > 0)
    fprintf(stderr, "[MEMREPORT] WARNING: %d buffer(s) have no buffer_registry.h entry -- "
                    "their index space is undeclared\n", unregistered);

  // And the reverse: registry entries naming buffers that no longer exist.
  fprintf(stderr, "[MEMREPORT] registry: %d entries, %d ORIGINAL-space (must be read via "
                  "oriParticleOrder in phase 2)\n", kBufferRegistryCount,
          [](){ int c = 0; for (int i = 0; i < kBufferRegistryCount; i++)
                             if (kBufferRegistry[i].space == SPACE_ORIGINAL) c++; return c; }());
}

void octree::writeGadgetSnapshot(bool preKick)
{
  const double snapT0 = get_time();   // Gadget-2 CPU_Snapshot

  if (!haveGadgetParams) return;

  if (preKick)
  {
    localTree.bodies_Ppos.d2h();
    localTree.bodies_Pvel.d2h();
    localTree.oriParticleOrder.d2h();
  }
  localTree.bodies_Ppos.d2h();
  localTree.bodies_vel.d2h();
  localTree.bodies_ids.d2h();
  // bodies_type is host-only and kept fresh by recomputeSoftening() every tree rebuild (see that
  // function's own comment) -- no d2h needed, and already index-aligned with bodies_pos/ids here.

  const int n = localTree.n;
  GadgetParticleData data;
  data.pos.resize(3 * n);
  data.vel.resize(3 * n);
  data.id.resize(n);
  data.mass.resize(n);
  data.type.resize(n);
  const bool haveTypes = (int) localTree.bodies_type.size() == n;
  for (int i = 0; i < n; ++i)
  {
    const real4 sp = localTree.bodies_Ppos[i];   // pos is bound to Ppos; identical post-correct
    const real4 sv = preKick ? localTree.bodies_Pvel[localTree.oriParticleOrder[i]]
                             : localTree.bodies_vel[i];
    data.pos[3 * i + 0] = sp.x;
    data.pos[3 * i + 1] = sp.y;
    data.pos[3 * i + 2] = sp.z;
#ifdef GADGET_HIP_HIGHRES
    // T51: undo the recentring translation so the snapshot is in the IC's own frame. Without this
    // the output would be silently offset and every downstream analysis tool would need to know the
    // shift -- which is exactly the manual bookkeeping this feature exists to remove.
    if (gadgetZoomShifted && gadgetParams.BoxSize > 0.0)
      for (int a = 0; a < 3; ++a)
      {
        double x = (double) data.pos[3 * i + a] - gadgetZoomShift[a];
        while (x >= gadgetParams.BoxSize) x -= gadgetParams.BoxSize;
        while (x < 0.0)                   x += gadgetParams.BoxSize;
        data.pos[3 * i + a] = (float) x;
      }
#endif
    data.vel[3 * i + 0] = sv.x;
    data.vel[3 * i + 1] = sv.y;
    data.vel[3 * i + 2] = sv.z;
    data.id[i]          = localTree.bodies_ids[i];
    data.mass[i]         = localTree.bodies_Ppos[i].w;
    // Falls back to type 1 (Halo) -- the same default Gadget-2 DM-only convention this port
    // already uses elsewhere -- when no --param IC ever populated bodies_type (e.g. a
    // --plummer/--cube dev run somehow reaching a --param-configured output path).
    data.type[i]         = haveTypes ? localTree.bodies_type[i] : 1;
  }

  GadgetSnapshotHeader header;
  header.time        = (double) t_current;
  header.redshift    = gadgetParams.ComovingIntegrationOn ? (1.0 / header.time - 1.0) : 0.0;
  header.BoxSize     = gadgetParams.BoxSize;
  header.Omega0      = gadgetParams.Omega0;
  header.OmegaLambda = gadgetParams.OmegaLambda;
  header.HubbleParam = gadgetParams.HubbleParam;

#ifdef GADGET_HIP_LONGIDS
  const bool longIds = true;
#else
  const bool longIds = false;
#endif

  std::string path = gadgetParams.OutputDir;
  if (!path.empty() && path.back() != '/') path += "/";
  char numbuf[16];
  snprintf(numbuf, sizeof(numbuf), "_%03d", gadgetSnapshotCount);
  path += gadgetParams.SnapshotFileBase + numbuf;

  std::string err;
  if (!gadget_snapshot_write(path, gadgetParams.SnapFormat, longIds,
                              gadgetParams.ComovingIntegrationOn != 0, header, data, err))
  {
    if (procId == 0)
      fprintf(stderr, "[GADGET-SNAPSHOT] write failed for '%s': %s\n", path.c_str(), err.c_str());
  }
  else if (procId == 0)
  {
    fprintf(stderr, "[GADGET-SNAPSHOT] wrote '%s' (N=%d, time=%g)\n", path.c_str(), n, header.time);
  }
  *gadget_cpu_snapshot_acc() += get_time() - snapT0;
  gadgetSnapshotCount++;
}

void octree::recomputeSoftening(tree_structure &tree)
{
  // Bug fix (Phase 5 ticket 06, LOG.md): tree.bodies_type is never reordered by sort_bodies()'s
  // SFC sort, so it can't be trusted as index-aligned with the current bodies_pos/bodies_acc1
  // order across rebuilds -- rebuild it fresh from the always-correctly-reordered bodies_ids
  // through the stable (id -> type) map recorded once at IC load (setGadgetTypeMap()). A no-op
  // when the map is empty (uniform-default-type runs, e.g. --plummer with no --param IC).
  // T35: the device path skips this host rebuild of bodies_type entirely -- the kernel below reads
  // bodies_ids straight from the device and writes both outputs there. bodies_type (host) is then
  // NOT refreshed, which is safe because nothing else reads it on the device path; the two
  // consumers, bodies_forceSoftening and bodies_typeDevice, are written by the kernel.
  // GADGET_HIP_T35_NO_DEVICE_SOFT=1 forces the original host+map path. It exists so the two can be
  // compared on the same binary: the device path feeds every force through bodies_forceSoftening,
  // so "the energy line is identical" is a real equivalence test, not a plausibility argument.
  static const bool t35NoDeviceSoft = (getenv("GADGET_HIP_T35_NO_DEVICE_SOFT") != NULL);
  const bool softeningOnDevice = gadgetTypeByIdOk && !gadgetIdToType.empty() && !t35NoDeviceSoft;

  // Host bodies_type must follow the sort for EVERY multi-type IC, not only when the softening
  // itself is computed on the host: the zoom region (recomputeZoomRegion / pm_zoom_region_out_of_
  // range) and the snapshot writer (data.type[i] = bodies_type[i]) read it against the CURRENT
  // (sorted) particle order. With T35's device softening this rebuild was skipped, so on a
  // multi-species IC the zoom region came out as the whole box ("crosses the periodic box
  // boundary") and snapshots would have carried the wrong type per particle. Uniform ICs leave
  // gadgetIdToType empty and are unaffected. Use the flat id->type table when it exists (a
  // 1e8-particle zoom IC makes the unordered_map path far too slow per rebuild).
  if (!gadgetIdToType.empty())
  {
    tree.bodies_ids.d2h();
    const int n = (int) tree.bodies_type.size();
    if (gadgetTypeByIdOk && !gadgetTypeById.empty())
    {
      for (int i = 0; i < n; ++i)
        tree.bodies_type[i] = gadgetTypeById[tree.bodies_ids[i]];
    }
    else
    {
      for (int i = 0; i < n; ++i)
        tree.bodies_type[i] = gadgetIdToType.at(tree.bodies_ids[i]);
    }
  }

  GadgetSofteningState state;
  if (haveGadgetParams)
  {
    GadgetSofteningParams p;
    p.softening[0] = gadgetParams.SofteningGas;
    p.softening[1] = gadgetParams.SofteningHalo;
    p.softening[2] = gadgetParams.SofteningDisk;
    p.softening[3] = gadgetParams.SofteningBulge;
    p.softening[4] = gadgetParams.SofteningStars;
    p.softening[5] = gadgetParams.SofteningBndry;
    p.softeningMaxPhys[0] = gadgetParams.SofteningGasMaxPhys;
    p.softeningMaxPhys[1] = gadgetParams.SofteningHaloMaxPhys;
    p.softeningMaxPhys[2] = gadgetParams.SofteningDiskMaxPhys;
    p.softeningMaxPhys[3] = gadgetParams.SofteningBulgeMaxPhys;
    p.softeningMaxPhys[4] = gadgetParams.SofteningStarsMaxPhys;
    p.softeningMaxPhys[5] = gadgetParams.SofteningBndryMaxPhys;
    gadget_set_softenings(p, gadgetParams.ComovingIntegrationOn != 0, (double) get_t_current(), state);
  }
  else
  {
    // No --param config: flat length for every type, driven by the existing eps CLI value
    // (this->eps2 is already squared, eps2=eps*eps -- see the constructor).
    const double eps = std::sqrt((double) this->eps2);
    for (int t = 0; t < 6; ++t)
    {
      state.softeningTable[t] = eps;
      state.forceSoftening[t] = 2.8 * eps;
    }
  }

  const int n = (int) tree.bodies_type.size();

  if (softeningOnDevice)
  {
    // Upload the flat (id -> type) table once; it never changes after IC load.
    if (gadgetTypeByIdDev.get_size() == 0)
    {
      gadgetTypeByIdDev.cmalloc((int) gadgetTypeById.size(), false);
      for (size_t i = 0; i < gadgetTypeById.size(); ++i) gadgetTypeByIdDev[i] = gadgetTypeById[i];
      gadgetTypeByIdDev.h2d();
    }

    // The softening TABLE is what changes with time (the MaxPhys clamp under comoving
    // integration); it is six numbers, so it is passed by value rather than uploaded.
    // Non-const: set_args() takes void* and rejects const-qualified arguments (same constraint the
    // ts_timeMax plumbing hit).
    float4 fsLo = make_float4((float) state.forceSoftening[0], (float) state.forceSoftening[1],
                              (float) state.forceSoftening[2], (float) state.forceSoftening[3]);
    float2 fsHi = make_float2((float) state.forceSoftening[4], (float) state.forceSoftening[5]);
    ullong maxId = gadgetTypeByIdMax;
    int nn = n;
    gadgetRefreshSoftening.set_args(0, &nn, tree.bodies_ids.p(), gadgetTypeByIdDev.p(), &maxId,
                                    &fsLo, &fsHi,
                                    tree.bodies_forceSoftening.p(), tree.bodies_typeDevice.p());
    gadgetRefreshSoftening.setWork(n, 128);
    gadgetRefreshSoftening.execute2(execStream->s());

    // C-C-21 on the TABLE, not on 134M copies of it. Every per-particle value is one of these six
    // entries by construction, so checking the table is the same check -- and it is the check that
    // actually names the mis-configured parameter, which scanning the expanded array never did.
    {
      static bool s_softWarnedDev = false;
      double sMin = 1e300, sMax = -1e300;
      for (int t = 0; t < 6; ++t)
      {
        if (gadgetTypeUsed[t] == 0) continue;      // a type with no particles cannot break anything
        sMin = std::min(sMin, state.forceSoftening[t]);
        sMax = std::max(sMax, state.forceSoftening[t]);
      }
      if (sMin <= 0.0 && !s_softWarnedDev)
      {
        fprintf(stderr,
          "\n[C-C-21] WARNING: force softening is <= 0 for at least one POPULATED particle type "
          "(min=%.6g, max=%.6g over the types present in this IC). Gadget-2 would produce NaN "
          "accelerations for this configuration; this port instead computes UNSOFTENED forces and "
          "keeps running, so the error is silent. Check SofteningXxx / SofteningXxxMaxPhys.\n\n",
          sMin, sMax);
        s_softWarnedDev = true;
      }
      static const bool invOnDev = (getenv("GADGET_HIP_INVARIANTS") != NULL);
      if (invOnDev)
        fprintf(stderr, "[INVARIANT] C-C-21 softening (table): %s (min=%.6g max=%.6g)\n",
                (sMin > 0.0) ? "OK" : "VIOLATED", sMin, sMax);
    }
    return;   // bodies_forceSoftening and bodies_typeDevice are already current on the device
  }

  for (int i = 0; i < n; ++i)
  {
    const int t = tree.bodies_type[i];
    tree.bodies_forceSoftening[i] = (real) state.forceSoftening[t];
  }
  // C-C-21 invariant: a zero or negative softening must not pass silently. Gadget-2 divides by
  // the softening unconditionally, so a mis-configured SofteningXxxMaxPhys=0 produces NaN
  // accelerations there and the run visibly dies. This port's spline guard instead completes
  // normally with UNSOFTENED forces -- defensible in isolation, but it converts a configuration
  // error into a silent physics change. Report it instead: the run continues (so this cannot
  // break a legitimate setup), but the configuration error is no longer invisible.
  {
    static bool s_softWarned = false;
    double sMin = 1e300, sMax = -1e300;
    for (int i = 0; i < n; ++i)
    {
      const double s = (double) tree.bodies_forceSoftening[i];
      sMin = std::min(sMin, s); sMax = std::max(sMax, s);
    }
    if (n > 0 && sMin <= 0.0 && !s_softWarned)
    {
      fprintf(stderr,
        "\n[C-C-21] WARNING: force softening is <= 0 for at least one particle "
        "(min=%.6g, max=%.6g). Gadget-2 would produce NaN accelerations for this configuration; "
        "this port instead computes UNSOFTENED forces and keeps running, so the error is silent. "
        "Check SofteningXxx / SofteningXxxMaxPhys in the parameter file.\n\n", sMin, sMax);
      s_softWarned = true;
    }
    static const bool invOn = (getenv("GADGET_HIP_INVARIANTS") != NULL);
    if (invOn)
      fprintf(stderr, "[INVARIANT] C-C-21 softening: %s (min=%.6g max=%.6g over %d particles)\n",
              (sMin > 0.0) ? "OK" : "VIOLATED", sMin, sMax, n);
  }

  tree.bodies_forceSoftening.h2d();

  // Kept in lockstep with bodies_forceSoftening above -- same per-rebuild refresh, same reason
  // (the tree build's SFC sort reorders bodies_type on the host between rebuilds). Unconditional
  // as of Ticket 07 (PLAN.md): the tree-walk gravity kernel now reads bodies_typeDevice in every
  // build (see octree.h's own updated doc comment on the field).
  for (int i = 0; i < n; ++i)
    tree.bodies_typeDevice[i] = tree.bodies_type[i];
  tree.bodies_typeDevice.h2d();
}

void octree::recomputeTimestepGlobals(tree_structure &tree)
{
  // T41: the device reduction is the default again. It is decided ONCE, here, because the
  // previous shape had two conditions that had to agree -- a copy guard and dtOnDevice --
  // and when one was flipped without the other the host path silently ran on every
  // uniform-box run, costing 1.05 s/step (28% of wall clock at 512^3) for two weeks.
  // GADGET_HIP_T35_NO_DEVICE_DT=1 forces the host path for comparison or if a kernel
  // stall reappears; GADGET_HIP_T35_DT_CROSSCHECK=1 validates one against the other.
  static const bool t35NoDeviceDt = (getenv("GADGET_HIP_T35_NO_DEVICE_DT") != NULL);
  if (!haveGadgetParams || gadgetParams.ComovingIntegrationOn == 0)
  {
    // Matches find_dt_displacement_constraint()'s own early-out (timestep.c:574): no comoving
    // integration means no displacement constraint at all.
    gadgetDtDisplacement = haveGadgetParams ? gadgetParams.MaxSizeTimestep : 1.0e30;
    return;
  }

  // A real device-to-host sync -- deliberately done at tree-rebuild granularity only (this
  // method's own call site), not every timestep.
  // T35: these three copies (4.8 GB at 512^3) exist only to feed the host reduction below, which
  // the device path replaces -- so on that path they are skipped entirely.
  const bool dtOnDevice = !t35NoDeviceDt && gadgetTypeByIdOk && !gadgetIdToType.empty();

  // These three copies (4.8 GB at 512^3) exist ONLY to feed the host reduction below, so
  // they are skipped whenever the device path runs. Both later consumers of the host
  // arrays -- the snapshot/energy gather and the statisticsIter block -- issue their own
  // d2h(), so nothing downstream depends on this one having happened.
  if (!dtOnDevice)
  {
    tree.bodies_Ppos.d2h();
    tree.bodies_vel.d2h();
    tree.oriParticleOrder.d2h();   // C-D-08: needed for the index translation below
  }

  // C-D-08 INDEX SPACE. These three vectors are consumed as PARALLEL arrays by
  // gadget_find_dt_displacement_constraint, which bins them PER TYPE exactly as Gadget-2 does
  // (timestep.c:566-644: count[type]++, v2sum[type] += v^2, min_mass[type] = min(...)). They must
  // therefore all be in one index space.
  //   tree.bodies_type  -- CURRENT order (recomputeSoftening rebuilds it from the resorted ids)
  //   tree.bodies_Ppos  -- CURRENT order (sort_bodies permutes it every iteration)
  //   tree.bodies_vel   -- ORIGINAL order: sort_bodies' default per-iteration reorder permutes
  //                        only bodies_Ppos/bodies_ids, NOT the velocity arrays
  // The loop previously read `tree.bodies_vel[i]` directly, pairing particle i's mass and TYPE
  // with a different particle's velocity. That was inert only because every IC in this project so
  // far is single-species -- with one type the per-type bins receive the same multiset of values
  // whatever the permutation, and `v2sum`/`min_mass` are permutation-invariant. It becomes a real
  // defect the moment a second populated type exists, i.e. for the zoom runs T28 targets.
  // Same defect class as C-B-13 and C-C-18; `correct_particles` already uses this exact
  // translation (`pVel[unsortedIdx]`, timestep.cu).
  const int n = (int) tree.bodies_type.size();

  // T35: reduce on the device when we can. GADGET_HIP_T35_NO_DEVICE_DT=1 forces the host path, so
  // the two can be compared on one binary -- this feeds gadgetDtDisplacement, hence every
  // subsequent timestep, so "the step sequence is identical" is the equivalence test.
  // History (T41): this kernel was switched off by default after it stalled the GPU on the
  // 1e8-particle multi-species zoom ICs (once at step 5, and a driver "GPU Hang" at step 2).
  // The comment that replaced it claimed "only multi-type ICs reach this branch at all, so
  // the uniform-box production runs never exercised it" -- that was WRONG. The copy guard
  // keyed off the same flag, so disabling the kernel put EVERY run on the host path.
  // Measured cost at 512^3: 1.05 s/step, 28%% of wall clock (tickets/T41).
  // One cause of the stall is now understood and fixed: the bounds guard added later to
  // diagnose it did `return` from inside the block, leaving that thread short of the
  // __syncthreads() the rest of the block was waiting at -- a divergent barrier, undefined
  // and a hang on AMD. That guard postdates the original stalls, so it cannot be their
  // cause; it is fixed in timestep.cu regardless. Whether the zoom stall itself is gone is
  // an empirical question -- benchmarks B and C are the test.
  long long aggCount[6]  = {0,0,0,0,0,0};
  double    aggVSum[6]   = {0,0,0,0,0,0};
  double    aggMinMass[6] = {1e30,1e30,1e30,1e30,1e30,1e30};

  if (dtOnDevice)
  {
    if (dtAggVSum.get_size() == 0)
    {
      dtAggVSum.cmalloc(6, false); dtAggMinMass.cmalloc(6, false); dtAggCount.cmalloc(6, false);
    }
    for (int t = 0; t < 6; ++t) { dtAggVSum[t] = 0.0; dtAggMinMass[t] = 1.0e30; dtAggCount[t] = 0ull; }
    dtAggVSum.h2d(); dtAggMinMass.h2d(); dtAggCount.h2d();

    int nn = n;
    gadgetTimestepGlobals.set_args(0, &nn, tree.bodies_Ppos.p(), tree.bodies_vel.p(),
                                   tree.oriParticleOrder.p(), tree.bodies_typeDevice.p(),
                                   dtAggVSum.p(), dtAggMinMass.p(), dtAggCount.p());
    gadgetTimestepGlobals.setWork(n, 128);
    gadgetTimestepGlobals.execute2(execStream->s());
    // Phase 0b: the kernel indexes vel[] through oriParticleOrder. A stale or out-of-range entry
    // there is an out-of-bounds device read, which is the leading suspect for the indefinite
    // stalls seen on the 1e8-particle multi-species zoom ICs. Report it rather than let a wrong
    // answer (or a hang) pass silently; the check costs one symbol read per step.
    {
      const unsigned int bad = dtglobals_read_badidx();
      if (bad != 0)
      {
        fprintf(stderr, "[P0B] gadget_timestep_globals: %u out-of-range oriParticleOrder entries "
                        "at iter=%d (n=%d) -- index-space defect, NOT contention\n",
                bad, iter, localTree.n);
        dtglobals_reset_badidx();
      }
    }
    execStream->sync();
    dtAggVSum.d2h(); dtAggMinMass.d2h(); dtAggCount.d2h();

    for (int t = 0; t < 6; ++t)
    { aggCount[t] = (long long) dtAggCount[t]; aggVSum[t] = dtAggVSum[t]; aggMinMass[t] = dtAggMinMass[t]; }

    // GADGET_HIP_T35_DT_CROSSCHECK=1: recompute the same three aggregates on the host, from the
    // same buffers, and report any disagreement. This is the direct test of the device reduction --
    // the trajectory-level comparison can only say "something differs", while this says which of
    // count / vSum / minMass is wrong and by how much. Expensive (it does the very copies the
    // device path exists to avoid), so it is off by default and meant for a handful of steps.
    static const bool dtCrossCheck = (getenv("GADGET_HIP_T35_DT_CROSSCHECK") != NULL);
    if (dtCrossCheck)
    {
      tree.bodies_Ppos.d2h(); tree.bodies_vel.d2h(); tree.oriParticleOrder.d2h();
      tree.bodies_typeDevice.d2h();
      double hVSum[6] = {0,0,0,0,0,0}, hMinMass[6] = {1e30,1e30,1e30,1e30,1e30,1e30};
      long long hCount[6] = {0,0,0,0,0,0};
      for (int i = 0; i < n; ++i)
      {
        const int t = tree.bodies_typeDevice[i];
        if (t < 0 || t >= 6) continue;
        const real4 v = tree.bodies_vel[tree.oriParticleOrder[i]];
        hVSum[t] += (double) v.x*v.x + (double) v.y*v.y + (double) v.z*v.z;
        const double m = tree.bodies_Ppos[i].w;
        if (m < hMinMass[t]) hMinMass[t] = m;
        hCount[t]++;
      }
      for (int t = 0; t < 6; ++t)
      {
        if (hCount[t] == 0 && aggCount[t] == 0) continue;
        const double dv = (hVSum[t] != 0.0) ? (aggVSum[t] - hVSum[t]) / hVSum[t] : 0.0;
        fprintf(stderr, "[T35-DTCHECK] iter=%d type=%d  count dev=%lld host=%lld %s | "
                        "vSum dev=%.10g host=%.10g rel=%+.3e | minMass dev=%.10g host=%.10g %s\n",
                iter, t, aggCount[t], hCount[t],
                (aggCount[t] == hCount[t]) ? "OK" : "*** MISMATCH ***",
                aggVSum[t], hVSum[t], dv,
                aggMinMass[t], hMinMass[t],
                (aggMinMass[t] == hMinMass[t]) ? "OK" : "*** MISMATCH ***");
      }
    }
  }

  std::vector<float> mass, velSqr;
  if (!dtOnDevice)
  {
    mass.resize(n); velSqr.resize(n);
    for (int i = 0; i < n; ++i)
    {
      mass[i]   = tree.bodies_Ppos[i].w;
      const real4 v = tree.bodies_vel[tree.oriParticleOrder[i]];
      velSqr[i] = v.x*v.x + v.y*v.y + v.z*v.z;
    }
  }

  // C-D-08 probe (GADGET_HIP_CD08_STATS=1): how many particles WOULD have had their velocity
  // attributed to the wrong per-type bin by the old direct `bodies_vel[i]` read. Zero for a
  // single-species run by construction (one bin), non-zero as soon as a second type exists --
  // which is the whole claim, so it is measured rather than asserted.
  if (getenv("GADGET_HIP_CD08_STATS"))
  {
    long mismatched = 0, permuted = 0;
    for (int i = 0; i < n; ++i)
    {
      const uint o = tree.oriParticleOrder[i];
      if ((int) o != i) permuted++;
      if (tree.bodies_type[i] != tree.bodies_type[o]) mismatched++;
    }
    fprintf(stderr, "[CD08] n=%d  permuted=%ld (%.1f%%)  type-mismatched under the OLD read=%ld "
                    "(%.1f%%)\n", n, permuted, 100.0*permuted/n, mismatched, 100.0*mismatched/n);
  }

  const double time    = (double) get_t_current();
  const double hubble_a = gadget_hubble_a(gadgetParams.Omega0, gadgetParams.OmegaLambda,
                                           gadgetParams.Hubble, time);
  const double hfac = hubble_a * time * time; // hfac = a^2*H(a), timestep.c:66

  // Phase 5 ticket 07 (PLAN.md): PMGRID/PLACEHIGHRESREGION Asmth[0]/[1] refinement
  // (timestep.c:623-628), previously deferred (gadget_cosmology.h's own doc comment). `asmth0`
  // mirrors the exact same formula this file's own zoom block above (and main.cpp's PM setup) use
  // for the coarse grid; `asmth1`/`zoomMask` come straight from the already-current zoom region
  // (valid by construction here -- the zoom block above runs, and keeps gadgetZoomRegion fresh,
  // BEFORE this function is ever called each rebuild).
  bool         havePMGRID = false;
  double       asmth0 = 0.0, asmth1 = 0.0;
  unsigned int zoomMaskForTimestep = 0;
#ifdef PMGRID
  {
    float pmBoxSize = 0.0f; int pmGridSize = 0;
    if (pm_get_dump_context(pmBoxSize, pmGridSize))
    {
      havePMGRID = true;
#ifdef PERIODIC
      asmth0 = 1.25 * pmBoxSize / pmGridSize;
#else
      asmth0 = 1.25 * pmBoxSize / (2.0 * pmGridSize); // pmBoxSize holds meshSize here
#endif
#ifdef GADGET_HIP_HIGHRES
      if (haveGadgetZoom)
      {
        asmth1              = (double) gadgetZoomRegion.asmth1;
        zoomMaskForTimestep = gadgetZoomMask;
      }
#endif
    }
  }
#endif

  // Same constraint either way; only where the eighteen aggregates came from differs.
  gadgetDtDisplacement = dtOnDevice
    ? gadget_find_dt_displacement_constraint_agg(
        true, gadgetParams.Omega0, gadgetParams.OmegaBaryon, gadgetParams.G, gadgetParams.Hubble,
        gadgetParams.MaxSizeTimestep, gadgetParams.MaxRMSDisplacementFac, hfac,
        aggCount, aggVSum, aggMinMass, havePMGRID, asmth0, asmth1, zoomMaskForTimestep)
    : gadget_find_dt_displacement_constraint(
        true, gadgetParams.Omega0, gadgetParams.OmegaBaryon, gadgetParams.G, gadgetParams.Hubble,
        gadgetParams.MaxSizeTimestep, gadgetParams.MaxRMSDisplacementFac, hfac, tree.bodies_type,
        mass, velSqr, havePMGRID, asmth0, asmth1, zoomMaskForTimestep);
}

void octree::initComovingTables()
{
  // Always allocated (zero-filled), regardless of whether comoving integration is actually on --
  // so predict()/correct() can unconditionally pass a valid device pointer to
  // predictParticles/correctParticles without a separate null-pointer case; the kernels never
  // dereference these unless comovingIntegrationOn!=0 is also passed, so an all-zero table is
  // inert whenever comoving is off.
  gadgetDriftTable.cmalloc(GADGET_DRIFT_TABLE_LENGTH, false);
  gadgetGravKickTable.cmalloc(GADGET_DRIFT_TABLE_LENGTH, false);
  for (int i = 0; i < GADGET_DRIFT_TABLE_LENGTH; ++i)
    gadgetDriftTable[i] = gadgetGravKickTable[i] = 0.0f;
  gadgetDriftTable.h2d();
  gadgetGravKickTable.h2d();

  if (!haveGadgetParams || gadgetParams.ComovingIntegrationOn == 0)
    return;

  GadgetDriftTables tables;
  gadget_init_drift_tables(gadgetParams.TimeBegin, gadgetParams.TimeMax, gadgetParams.Omega0,
                            gadgetParams.OmegaLambda, gadgetParams.Hubble, tables);

  for (int i = 0; i < GADGET_DRIFT_TABLE_LENGTH; ++i)
  {
    gadgetDriftTable[i]    = tables.driftTable[i];
    gadgetGravKickTable[i] = tables.gravKickTable[i];
  }
  gadgetDriftTable.h2d();
  gadgetGravKickTable.h2d();

  gadgetLogTimeBegin = tables.logTimeBegin;   // T29: keep double
  gadgetLogTimeMax   = tables.logTimeMax;
  haveGadgetComovingTables = true;
  // Phase 3: the long-range kick factor is evaluated on the HOST, once per PM interval, so the
  // table has to outlive this function.
  s_gadgetTables   = tables;
  s_gadgetTablesOk = true;

  // Phase 2: the integer timeline, initialised at the first point where TimeBegin, TimeMax and the
  // comoving flag are all known. From here ticks are the clock and t_current is derived from them.
  initTimeline(gadgetParams.TimeBegin, gadgetParams.TimeMax, gadgetParams.ComovingIntegrationOn);
  fprintf(stderr, "[TIMELINE] integer timeline: TIMEBASE=%d dPerTick=%.17g Ti_current=%d "
                  "(TimeBegin=%.17g TimeMax=%.17g comoving=%d)\n",
          GADGET_TIMEBASE, gadgetDPerTick, Ti_current,
          gadgetParams.TimeBegin, gadgetParams.TimeMax, gadgetParams.ComovingIntegrationOn);
}

#ifdef GADGET_HIP_HIGHRES
void octree::setZoomConfig(unsigned int mask, float enlargeRegion)
{
  gadgetZoomMask    = mask;
  gadgetZoomEnlarge = enlargeRegion;
  haveGadgetZoom    = (mask != 0);
}

void octree::recomputeZoomRegion(tree_structure &tree, double asmth0, double rcut0, int gridSize,
                                  float periodicBoxSize)
{
  if (!haveGadgetZoom) return;

  tree.bodies_Ppos.d2h();
  const int n = (int) tree.bodies_type.size();
  std::vector<Vec3f> h_pos(n);
  for (int i = 0; i < n; ++i)
  {
    const real4 p = tree.bodies_Ppos[i];
    h_pos[i] = Vec3f{ p.x, p.y, p.z };
  }

  ZoomRegion newRegion;
  if (!pm_zoom_compute_region(h_pos, tree.bodies_type, gadgetZoomMask, (double) gadgetZoomEnlarge,
                               gridSize, asmth0, rcut0, newRegion))
  {
    fprintf(stderr, "[ZOOM] ERROR: --zoom-mask 0x%x matches no particle -- nothing to build a "
                     "high-res region around.\n", gadgetZoomMask);
    ::exit(1);
  }

  // Explicit "may not cross a periodic box boundary" check (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.2/
  // 1.6 -- Gadget-2 itself has no runtime assertion for this; fail loud rather than silently
  // produce wrong physics from a fine grid that wraps around the periodic box).
  if (!pm_zoom_region_fits_in_box(newRegion, periodicBoxSize))
  {
    fprintf(stderr, "[ZOOM] ERROR: high-res region [%.6g,%.6g,%.6g] + %.6g crosses the periodic "
                     "box boundary (boxSize=%.6g) -- not supported (Gadget-2's own documented "
                     "modeling requirement).\n", newRegion.corner[0], newRegion.corner[1],
                     newRegion.corner[2], newRegion.totalMeshSize, periodicBoxSize);
    ::exit(1);
  }

  if (gadgetZoomSolverReady)
    pm_solver_destroy_isolated(gadgetZoomSolver);

  const float asmthRatio = newRegion.asmth1 / (float) asmth0;
  gadgetZoomSolver = pm_solver_create_finegrid(gridSize, newRegion.meshSize, asmthRatio);
  gadgetZoomSolverReady = true;
  gadgetZoomRegion = newRegion;

  // Phase 5 ticket 07 (PLAN.md): keeps the tree-walk gravity kernel's own (Rcut[1],Asmth[1]) pair
  // (dev_approximate_gravity_warp_new.cu's g_pm_rcut2_1/g_pm_asmthfac_1) in lockstep with the
  // region's own PM-side asmth1/rcut1 -- both change together, only here, whenever the region is
  // (re)computed.
  pm_zoom_upload_rcut_asmth(newRegion.rcut1, newRegion.asmth1, gadgetZoomMask);

  fprintf(stderr, "[ZOOM] region corner=(%.6g,%.6g,%.6g) totalMeshSize=%.6g asmth1=%.6g "
                   "rcut1=%.6g asmthRatio=%.6g\n", newRegion.corner[0], newRegion.corner[1],
                   newRegion.corner[2], newRegion.totalMeshSize, newRegion.asmth1,
                   newRegion.rcut1, asmthRatio);
}
#endif

void octree::approximate_gravity(tree_structure &tree)
{

  uint2 node_begend;
  int level_start = tree.startLevelMin;
  node_begend.x   = tree.level_list[level_start].x;
  node_begend.y   = tree.level_list[level_start].y;

  tree.activePartlist.zeroMemGPUAsync(gravStream->s());
  LOG("node begend: %d %d iter-> %d\n", node_begend.x, node_begend.y, iter);

#ifdef _MAC_SPRINGEL_
  // T31 bisect: GADGET_HIP_T31_SKIP_GMA=1 skips this call. groupMaxAccInfo was zero-allocated
  // (build.cpp cmalloc(..., true)) and on the bootstrap iteration the loop only ever writes zeros
  // anyway, so skipping it leaves the device state the walk sees unchanged -- it isolates this
  // call's MEMORY TRAFFIC (2.1 GB d2h of bodies_acc0, plus oriParticleOrder and two h2d's) from
  // its arithmetic.
  {
    static const bool t31_skipGma = (getenv("GADGET_HIP_T31_SKIP_GMA") != NULL);
    if (!t31_skipGma) computeGroupMaxAccel(tree);
    else if (iter == 0) fprintf(stderr, "[T31] computeGroupMaxAccel SKIPPED (bisect)\n");
  }

  // T24: test the "smeared node near the periodic boundary" hypothesis for the iter~235 blowup --
  // bodies_acc1's own top-N (see compute_energies' T24-ACC1-TOPN diagnostic) shows the blown-up
  // particles sit right at/near the periodic edges (x~31956 paired with x~45-103, a genuine close
  // encounter across the x=0/32000 wrap), spread across MULTIPLE different groups (ruling out a
  // single corrupted GROUP) -- consistent with a single corrupted internal tree NODE whose
  // multipole center-of-mass was computed with a naive (non-minimum-image-aware) average across
  // that same boundary, giving it a wildly wrong apparent COM position and/or an anomalously huge
  // AABB (spanning most of the box) while still carrying a large, real mass. Dump the largest-AABB
  // internal nodes each iteration in the target window to check directly. Env-gated, host-side only.
  {
    static bool t24n_init = false;
    static int t24n_lo = 220, t24n_hi = 245;
    if (!t24n_init) {
      t24n_init = true;
      const char *e = getenv("GADGET_HIP_T24_NODE_WINDOW");
      if (e) sscanf(e, "%d,%d", &t24n_lo, &t24n_hi);
    }
    if (getenv("GADGET_HIP_T24_NODE_SCAN") && iter >= t24n_lo && iter <= t24n_hi) {
      tree.boxSizeInfo.d2h();
      tree.boxCenterInfo.d2h();
      tree.multipole.d2h();
      tree.cellSizeInfo.d2h(); //T26: dump alongside the old AABB-based bJ for direct comparison
      const int nNodes = tree.n_nodes;
      std::vector<int> nidx(nNodes);
      for (int i = 0; i < nNodes; i++) nidx[i] = i;
      const int topN = std::min(8, nNodes);
      // T26 debug: also find the SMALLEST-bJ (old, AABB-based) internal node with real mass -- T24's
      // bug was a pathologically SMALL AABB, not a large one, so a bottom-N scan is the relevant one
      // for checking whether T26's fixed-cell replacement is landing correctly for that same node.
      std::vector<int> nidxSmall = nidx;
      std::partial_sort(nidxSmall.begin(), nidxSmall.begin() + topN, nidxSmall.end(),
          [&](int a, int b) {
            const float4 ca = tree.boxCenterInfo[a];
            const float4 cb = tree.boxCenterInfo[b];
            // internal nodes only (cellOp > 0, per compute_scaling's own leaf-vs-node sign convention)
            const bool aIsNode = ca.w > 0.0f;
            const bool bIsNode = cb.w > 0.0f;
            if (aIsNode != bIsNode) return aIsNode; // nodes sort before leaves
            const float4 sa = tree.boxSizeInfo[a];
            const float4 sb = tree.boxSizeInfo[b];
            const float ba = 2.0f*std::max(sa.x, std::max(sa.y, sa.z));
            const float bb = 2.0f*std::max(sb.x, std::max(sb.y, sb.z));
            return ba < bb;
          });
      std::partial_sort(nidx.begin(), nidx.begin() + topN, nidx.end(),
          [&](int a, int b) {
            const float4 sa = tree.boxSizeInfo[a];
            const float4 sb = tree.boxSizeInfo[b];
            const float ba = 2.0f*std::max(sa.x, std::max(sa.y, sa.z));
            const float bb = 2.0f*std::max(sb.x, std::max(sb.y, sb.z));
            return ba > bb;
          });
      for (int k = 0; k < topN; k++) {
        const int i = nidx[k];
        const float4 sz  = tree.boxSizeInfo[i];
        const float4 ctr = tree.boxCenterInfo[i];
        const float4 com = tree.multipole[3*i];
        const float bJ = 2.0f*std::max(sz.x, std::max(sz.y, sz.z));
        fprintf(stderr, "[T24-NODE-SCAN] iter=%d rank=%d node=%d bJ=%.6g cellSizeT26=%.6g "
                "boxCenter=(%.6g,%.6g,%.6g) boxHalfSize=(%.6g,%.6g,%.6g) mass=%.6g "
                "COM=(%.6g,%.6g,%.6g)\n",
                iter, k, i, bJ, tree.cellSizeInfo[i], ctr.x, ctr.y, ctr.z, sz.x, sz.y, sz.z, ctr.w,
                com.x, com.y, com.z);
      }
      for (int k = 0; k < topN; k++) {
        const int i = nidxSmall[k];
        const float4 sz  = tree.boxSizeInfo[i];
        const float4 ctr = tree.boxCenterInfo[i];
        const float4 com = tree.multipole[3*i];
        const float bJ = 2.0f*std::max(sz.x, std::max(sz.y, sz.z));
        fprintf(stderr, "[T26-NODE-SCAN-SMALL] iter=%d rank=%d node=%d bJ=%.6g cellSizeT26=%.6g "
                "isNode=%d boxCenter=(%.6g,%.6g,%.6g) boxHalfSize=(%.6g,%.6g,%.6g) mass=%.6g "
                "COM=(%.6g,%.6g,%.6g)\n",
                iter, k, i, bJ, tree.cellSizeInfo[i], (ctr.w > 0.0f), ctr.x, ctr.y, ctr.z,
                sz.x, sz.y, sz.z, ctr.w, com.x, com.y, com.z);
      }
    }
  }
#endif

  // ---------------------------------------------------------------------------------------------
  // C-A-01 NODE AUDIT (GADGET_HIP_CA01_AUDIT=<iter>, host-side, zero cost when unset).
  //
  // C-A-01's stated first question: is `cellSizeInfo` -- the fixed octree cell side decoded from
  // node_bodies[].x's level field -- actually a valid Gadget-2 `len` for EVERY node, or does the
  // level field lose its fixed-cell meaning for nodes that `cl_link_tree` forms by merging children
  // created at different levels? T26 tried `bJ = cellSizeInfo` and regressed the T24 repro with the
  // mechanism explicitly not root-caused; that hypothesis is the leading suspect and it is testable
  // from a tree dump alone, with no force evaluation involved.
  //
  // Two tests per node, both necessary:
  //   (1) CONTAINMENT: the node's particle AABB must fit inside a cube of side cellSize.
  //       2*max(boxHalfSize) <= cellSize.  A violation means cellSize is too small to be `len`.
  //   (2) ALIGNMENT: the AABB must lie inside the SPECIFIC grid-aligned cell of that side, i.e.
  //       floor((r_min-corner)/cellSize) == floor((r_max-corner)/cellSize) per axis. Containment
  //       alone can pass while the node straddles a cell boundary, which would mean the level is
  //       right about SIZE but the node is not a real octree cell -- a different defect with the
  //       same symptom.
  // Plus the distribution C-A-01 actually asks for: the ratio bJ_AABB/cellSize, whose LOWER tail
  // (not its median) is the claim to settle, and the count of exactly-zero bJ.
  {
    static int ca01_iter = -2;
    static bool ca01_init = false;
    if (!ca01_init) { ca01_init = true; const char *e = getenv("GADGET_HIP_CA01_AUDIT"); if (e) ca01_iter = atoi(e); }
    if (ca01_iter >= 0 && iter == ca01_iter)
    {
      tree.boxSizeInfo.d2h(); tree.boxCenterInfo.d2h(); tree.cellSizeInfo.d2h();
      tree.node_bodies.d2h(); tree.multipole.d2h();
#ifdef _MAC_SPRINGEL_
      // Only populated in a Springel build -- elsewhere it is a size-1 stub (build.cpp), so the
      // cell-centre section below must not index it.
      tree.cellCenterInfo.d2h();
      const bool haveCellCentre = true;
#else
      const bool haveCellCentre = false;
#endif
      const int nNodes = tree.n_nodes;
      const double cx = tree.corner.x, cy = tree.corner.y, cz = tree.corner.z;
      const double dfac = tree.domain_fac;

      long nInternal = 0, nLeaf = 0, nContainViol = 0, nAlignViol = 0, nZeroBJ = 0, nLevelBad = 0;
      long nSingle = 0, nLeafChild = 0, nZeroBJLeaf = 0, nZeroBJInternal = 0, nSingleNotForced = 0;
      long nZeroUnprotected = 0, nAlignDeep = 0, nAlignUnexplained = 0;
      double worstHalfCellRatio = 0.0, worstUlpRatio = 0.0, worstCcUlp = 0.0, worstCcFrac = 0.0;
      double worstStraddle = 0.0, straddleSum = 0.0; int worstStraddleNode = -1;
      long nCellCentreViol = 0, nOffNonzero = 0, offN = 0, nCentreSelfTest = 0;
      double offSum = 0.0, offMax = 0.0;
      double worstContain = 0.0; int worstContainNode = -1;
      double minRatio = 1e30; int minRatioNode = -1;
      // log2-spaced histogram of bJ_AABB / cellSize in [0,1]
      const int NB = 12; long hist[NB+1]; for (int b = 0; b <= NB; b++) hist[b] = 0;

      for (int i = 0; i < nNodes; i++)
      {
        const float4 sz  = tree.boxSizeInfo[i];
        const float4 ctr = tree.boxCenterInfo[i];
        const bool isNode = (ctr.w > 0.0f);      // compute_scaling's leaf/node sign convention
        if (isNode) nInternal++; else { nLeaf++; }

        const uint bijx  = tree.node_bodies[i].x;
        const uint level = (bijx & LEVELMASK) >> BITLEVELS;
        const uint shiftLevelForNode = (level <= (uint)MAXLEVELS) ? (uint)MAXLEVELS - level : 0u;
        const uint pfirst = bijx & ILEVELMASK;
        const int  nchild = (int)(tree.node_bodies[i].y - pfirst);
        if (!isNode) { if (nchild == 1) nSingle++; nLeafChild += nchild; }
        if (level > (uint)MAXLEVELS) { nLevelBad++; continue; }

        const double cell = tree.cellSizeInfo[i];
        const double bJ   = 2.0 * std::max(sz.x, std::max(sz.y, sz.z));
        if (bJ == 0.0) { nZeroBJ++; if (isNode) nZeroBJInternal++; else nZeroBJLeaf++; }
        // compute_scaling force-opens nchild==1 by setting cellOp = 10e10 (Gadget-2's own rule),
        // then NEGATES cellOp for leaves as the leaf marker -- so the test is on |cellOp|, not on
        // cellOp. Verify the rule is actually in force: without it a zero-extent node is
        // approximable at any distance, which is C-A-01's UNBOUNDED claim.
        const bool forceOpened = (fabsf(ctr.w) > 1e9f);
        if (nchild == 1 && !forceOpened) nSingleNotForced++;
        // The dangerous set: bJ == 0 (so both MAC tests are structurally defeated) AND not
        // force-opened. This, and only this, is C-A-01's unbounded case actually being reachable.
        if (bJ == 0.0 && !forceOpened)
        {
          nZeroUnprotected++;
          if (nZeroUnprotected <= 5)
            fprintf(stderr, "[CA01-AUDIT]   unprotected zero-extent node %d: isNode=%d nchild=%d "
                            "cellOp=%.6g center=(%.9g,%.9g,%.9g) cell=%.6g\n",
                    i, (int)isNode, nchild, ctr.w, ctr.x, ctr.y, ctr.z, cell);
        }

        // (1) containment
        if (cell > 0.0 && bJ > cell * (1.0 + 1e-5))
        {
          nContainViol++;
          const double excess = bJ / cell;
          if (excess > worstContain) { worstContain = excess; worstContainNode = i; }
        }
        // (2) alignment: the AABB's two corners must land in the same cell on every axis
        if (cell > 0.0)
        {
          const double lo[3] = { ctr.x - sz.x, ctr.y - sz.y, ctr.z - sz.z };
          const double hi[3] = { ctr.x + sz.x, ctr.y + sz.y, ctr.z + sz.z };
          const double org[3] = { cx, cy, cz };
          bool straddles = false;
          double worstHere = 0.0;
          int    worstAxis = 0;
          for (int a = 0; a < 3; a++)
          {
            const double fl = floor((lo[a] - org[a]) / cell);
            const double fh = floor((hi[a] - org[a]) / cell);
            if (fl != fh)
            {
              straddles = true;
              // How FAR past the boundary, as a fraction of the cell. A node that merely touches a
              // cell face -- or whose float32 coordinate rounds across it -- straddles by ~1 ULP of
              // the coordinate, which at 512^3 is ~1e-6 of a deep cell. A structurally wrong level
              // would straddle by an O(1) fraction. The two are not confusable if measured.
              const double bnd = org[a] + fh * cell;   // the boundary the AABB crosses
              const double over = (hi[a] - bnd) / cell;
              const double under = (bnd - lo[a]) / cell;
              const double d = std::min(over, under);  // the shallower side is the overhang
              if (d > worstHere) { worstHere = d; worstAxis = a; }
            }
          }
          if (straddles)
          {
            nAlignViol++;
            if (worstHere > worstStraddle) { worstStraddle = worstHere; worstStraddleNode = i; }
            straddleSum += worstHere;
            if (worstHere > 1e-3) nAlignDeep++;
            // What scale are these straddles? Two candidate explanations, and normalising by the
            // CELL cannot tell them apart -- it flatters deep nodes and damns shallow ones.
            //   (a) the roundf half-fine-cell grid offset (C-A-16): absolute overhang <= 0.5*df
            //   (b) float32 rounding of the coordinate: absolute overhang ~ 1 ULP of |coord|
            // Measure the overhang in ABSOLUTE units and report it against both, rather than
            // asserting either. (Both explanations have been asserted about these numbers in this
            // audit, at different times; only this comparison distinguishes them.)
            const double absOver = worstHere * cell;
            // The ULP must be taken on the axis that actually straddled and at that axis's own
            // coordinate. An earlier version of this line used axis 0 unconditionally and reported
            // 177 ULP for a node whose straddle was on y at |coord| ~ 89 while x sat near the box
            // corner -- an artefact of the instrument, not of the tree.
            const double coordMag = std::max(std::fabs(lo[worstAxis]),
                                             std::max(std::fabs(hi[worstAxis]), 1e-30));
            const double ulp = std::ldexp(1.0, std::ilogb(coordMag) - 23);  // float32 ULP there
            const double rUlp = absOver / ulp;
            const double rHalfCell = absOver / (0.5 * (double) tree.domain_fac);
            if (rUlp > worstUlpRatio) worstUlpRatio = rUlp;
            if (rHalfCell > worstHalfCellRatio) worstHalfCellRatio = rHalfCell;
            if (rUlp > 4.0) nAlignUnexplained++;   // beyond a few ULP is not rounding
          }
        }
        // C-A-01 part (b): the fixed octree-cell CENTRE. Two things to establish -- that it is
        // right, and that it is not simply the AABB midpoint under another name (if the two
        // coincided the change would be a no-op and every downstream measurement meaningless).
        //   (i) the AABB must fit inside the cell CENTRED ON cellCenterInfo, on every axis. This
        //       is a direct test of the centre, stronger than the grid-alignment test above: a
        //       centre placed on the wrong cell fails it by half a cell or more.
        //  (ii) |AABB midpoint - cell centre| / cell, whose distribution says how much the fix
        //       actually moves the proximity box.
        if (cell > 0.0 && haveCellCentre)
        {
          const float4 cc = tree.cellCenterInfo[i];
          const double half = 0.5 * cell;
          double worstOff = 0.0;
          const double ctrArr[3] = { (double)ctr.x, (double)ctr.y, (double)ctr.z };
          const double offs[3] = { fabs((double)ctr.x - (double)cc.x),
                                   fabs((double)ctr.y - (double)cc.y),
                                   fabs((double)ctr.z - (double)cc.z) };
          const double hx[3] = { sz.x, sz.y, sz.z };
          bool ccBad = false;
          for (int a = 0; a < 3; a++)
          {
            if (offs[a] + hx[a] > half * (1.0 + 1e-5)) ccBad = true;
            if (offs[a] / cell > worstOff) worstOff = offs[a] / cell;
          }
          if (ccBad)
          {
            nCellCentreViol++;
            // By how much does it fail, and is that a misplaced cell or float noise? A wrong cell
            // misses by ~half a cell; rounding misses by ~1 ULP of the coordinate.
            for (int a = 0; a < 3; a++)
            {
              const double excess = (offs[a] + hx[a]) - half;
              if (excess <= 0.0) continue;
              const double cmag = std::max(std::fabs(ctrArr[a]), 1e-30);
              const double u    = std::ldexp(1.0, std::ilogb(cmag) - 23);
              const double r    = excess / u;
              if (r > worstCcUlp) worstCcUlp = r;
              if (excess / cell > worstCcFrac) worstCcFrac = excess / cell;
            }
            if (nCellCentreViol <= 3)
              fprintf(stderr, "[CA01-AUDIT]   cell-centre violation node %d: cell=%.6g "
                              "mid=(%.9g,%.9g,%.9g) cc=(%.9g,%.9g,%.9g) half=(%.6g,%.6g,%.6g)\n",
                      i, cell, ctr.x, ctr.y, ctr.z, cc.x, cc.y, cc.z, sz.x, sz.y, sz.z);
          }
          offSum += worstOff; offN++;
          if (worstOff > offMax) offMax = worstOff;
          if (worstOff > 0.0) nOffNonzero++;

          // SELF-TEST for the check above. "0 violations" is worth nothing unless the test can
          // report a violation, so run the identical test against a centre deliberately displaced
          // by one whole cell in x. Every node whose AABB is not degenerate must fail it. If this
          // counter comes back at 0 too, the check is inert and the real result means nothing.
          {
            bool badInjected = false;
            for (int a = 0; a < 3; a++)
            {
              const double off = (a == 0) ? fabs(((double)ctr.x - ((double)cc.x + cell))) : offs[a];
              if (off + hx[a] > half * (1.0 + 1e-5)) badInjected = true;
            }
            if (badInjected) nCentreSelfTest++;
          }
        }
        if (cell > 0.0)
        {
          const double r = bJ / cell;
          if (r < minRatio) { minRatio = r; minRatioNode = i; }
          int b = 0;
          if (r > 0.0) { b = (int)(NB + floor(log2(r))); if (b < 0) b = 0; if (b > NB) b = NB; }
          hist[b]++;
        }
      }

      fprintf(stderr, "[CA01-AUDIT] iter=%d nodes=%d internal=%ld leaves=%ld corner=(%.9g,%.9g,%.9g) "
                      "domain_fac=%.9g rootSide=%.9g\n",
              iter, nNodes, nInternal, nLeaf, cx, cy, cz, dfac, dfac * (double)(1u << MAXLEVELS));
      fprintf(stderr, "[CA01-AUDIT] containment violations (AABB wider than the claimed cell): %ld"
                      "   worst bJ/cell = %.6g at node %d\n", nContainViol, worstContain, worstContainNode);
      fprintf(stderr, "[CA01-AUDIT] alignment violations (AABB straddles the claimed cell): %ld"
                      "   deepest overhang = %.3g cell at node %d ; mean = %.3g cell ; "
                      "deeper than 1e-3 cell: %ld\n",
              nAlignViol, worstStraddle, worstStraddleNode,
              nAlignViol ? straddleSum/(double)nAlignViol : 0.0, nAlignDeep);
      if (nAlignViol)
        fprintf(stderr, "[CA01-AUDIT] straddle scale: worst overhang = %.3g float32 ULP of the "
                        "coordinate, and %.3g x the roundf half-fine-cell offset; beyond 4 ULP: %ld\n",
                worstUlpRatio, worstHalfCellRatio, nAlignUnexplained);
      fprintf(stderr, "[CA01-AUDIT] level field out of range: %ld\n", nLevelBad);
      fprintf(stderr, "[CA01-AUDIT] leaves: %ld holding %ld particles (mean %.2f); single-particle leaves: %ld\n",
              nLeaf, nLeafChild, nLeaf ? (double)nLeafChild/(double)nLeaf : 0.0, nSingle);
      fprintf(stderr, "[CA01-AUDIT] bJ_AABB == 0: %ld total (%ld leaves, %ld internal); "
                      "nchild==1 nodes NOT force-opened: %ld\n",
              nZeroBJ, nZeroBJLeaf, nZeroBJInternal, nSingleNotForced);
      fprintf(stderr, "[CA01-AUDIT] UNPROTECTED zero-extent nodes (bJ==0 and not force-opened): %ld\n",
              nZeroUnprotected);
      if (haveCellCentre)
        fprintf(stderr, "[CA01-AUDIT] cell-CENTRE violations (AABB outside the cell centred there): %ld"
                        "   [self-test: the same check on a centre displaced by one cell flags %ld]\n",
                nCellCentreViol, nCentreSelfTest);
      if (nCellCentreViol)
        fprintf(stderr, "[CA01-AUDIT] cell-centre failures miss by at most %.3g of a cell "
                        "(= %.3g float32 ULP of the coordinate): %s\n",
                worstCcFrac, worstCcUlp,
                worstCcFrac > 0.01 ? "MISPLACED CELL" : "float32 rounding, not a wrong cell");
      else
        fprintf(stderr, "[CA01-AUDIT] cell-CENTRE checks skipped: geometric-MAC build, nothing reads "
                        "cellCenterInfo and it is not allocated\n");
      if (haveCellCentre)
        fprintf(stderr, "[CA01-AUDIT] |AABB midpoint - cell centre| / cell: mean %.4g, max %.4g, "
                        "nonzero for %ld/%ld nodes\n",
              offN ? offSum/(double)offN : 0.0, offMax, nOffNonzero, offN);
      fprintf(stderr, "[CA01-AUDIT] min bJ/cell = %.6g at node %d\n", minRatio, minRatioNode);
      for (int b = 0; b <= NB; b++)
      {
        if (!hist[b]) continue;
        if (b == 0) fprintf(stderr, "[CA01-AUDIT] hist bJ/cell < 2^-12 : %ld\n", hist[b]);
        else        fprintf(stderr, "[CA01-AUDIT] hist bJ/cell in [2^%d,2^%d) : %ld\n", b-NB, b-NB+1, hist[b]);
      }
    }
  }

  // T26: direct floor-vs-fixed-cell bJ distribution comparison (Method step 3 of T26 -- does the old
  // T24 floor really inflate most nodes' effective size throughout the run, and does the new
  // fixed-cell bJ avoid that while still preventing the pathological shrink?). Host-side only,
  // zero-cost when unset. Compares three quantities per node: the raw AABB-doubled proxy (the
  // pre-T24 bJ), the T24-floored value (0.35*periodic box size, PERIODIC builds only -- computed
  // here even for non-PERIODIC builds using the passed-in box size env var, purely for this
  // comparison, not a runtime behavior change), and the T26 fixed-cell value actually used now.
  // Independent of whether _MAC_SPRINGEL_ is compiled in -- boxSizeInfo/cellSizeInfo are always
  // computed by compute_properties() regardless.
  if (getenv("GADGET_HIP_T26_BJ_DIST")) {
    static int t26_every = -1;
    if (t26_every == -1) {
      const char *e = getenv("GADGET_HIP_T26_BJ_DIST_EVERY");
      t26_every = e ? atoi(e) : 50;
      if (t26_every <= 0) t26_every = 1;
    }
    if (iter % t26_every == 0) {
      tree.boxSizeInfo.d2h();
      tree.cellSizeInfo.d2h();
      const int nNodes = tree.n_nodes;
      const char *boxEnv = getenv("GADGET_HIP_T26_BJ_DIST_BOXSIZE");
      const float t24FloorAbs = boxEnv ? 0.35f * atof(boxEnv) : 0.0f;
      // T27: softening-based floor candidate, same coefficient env var production uses
      // (GADGET_HIP_T27_SOFT_COEF), applied to the global max per-particle ForceSoftening (this
      // map's Zel'dovich test is single-species, so per-group max == global max in practice; using
      // the global max here is a deliberately conservative, cheap stand-in that avoids a
      // per-group oriParticleOrder walk inside this diagnostic).
      const char *t27CoefEnv = getenv("GADGET_HIP_T27_SOFT_COEF");
      const float t27Coef = t27CoefEnv ? atof(t27CoefEnv) : 0.0f;
      float t27MaxSoftening = 0.0f;
      if (t27Coef > 0.0f) {
        tree.bodies_forceSoftening.d2h();
        for (int i = 0; i < tree.n; i++)
          t27MaxSoftening = std::max(t27MaxSoftening, (float)tree.bodies_forceSoftening[i]);
      }
      const float t27FloorAbs = t27Coef * t27MaxSoftening;
      double sumAabb = 0.0, sumFloor = 0.0, sumFixed = 0.0, sumT27 = 0.0;
      float  maxAabb = 0.0f, maxFloor = 0.0f, maxFixed = 0.0f, maxT27 = 0.0f;
      float  minAabb = 1e30f, minFloor = 1e30f, minFixed = 1e30f, minT27 = 1e30f;
      long   nFloorInflated = 0, nFixedInflated = 0, nT27Inflated = 0; // count of nodes where the guard > 2x the raw AABB proxy
      for (int i = 0; i < nNodes; i++) {
        const float4 sz = tree.boxSizeInfo[i];
        const float aabb  = 2.0f*std::max(sz.x, std::max(sz.y, sz.z));
        const float floorv = std::max(aabb, t24FloorAbs);
        const float fixed  = tree.cellSizeInfo[i];
        const float t27v   = std::max(aabb, t27FloorAbs);
        sumAabb += aabb; sumFloor += floorv; sumFixed += fixed; sumT27 += t27v;
        maxAabb = std::max(maxAabb, aabb); maxFloor = std::max(maxFloor, floorv); maxFixed = std::max(maxFixed, fixed); maxT27 = std::max(maxT27, t27v);
        minAabb = std::min(minAabb, aabb); minFloor = std::min(minFloor, floorv); minFixed = std::min(minFixed, fixed); minT27 = std::min(minT27, t27v);
        if (floorv > 2.0f*aabb) nFloorInflated++;
        if (fixed  > 2.0f*aabb) nFixedInflated++;
        if (t27v   > 2.0f*aabb) nT27Inflated++;
      }
      fprintf(stderr, "[T26-BJ-DIST] iter=%d nNodes=%d "
              "meanAABB=%.6g meanFLOOR=%.6g meanFIXED=%.6g meanT27=%.6g "
              "minAABB=%.6g minFLOOR=%.6g minFIXED=%.6g minT27=%.6g "
              "maxAABB=%.6g maxFLOOR=%.6g maxFIXED=%.6g maxT27=%.6g "
              "nFloorInflated(>2x)=%ld(%.1f%%) nFixedInflated(>2x)=%ld(%.1f%%) nT27Inflated(>2x)=%ld(%.1f%%) "
              "t27Coef=%.6g t27MaxSoftening=%.6g\n",
              iter, nNodes,
              sumAabb/nNodes, sumFloor/nNodes, sumFixed/nNodes, sumT27/nNodes,
              minAabb, minFloor, minFixed, minT27,
              maxAabb, maxFloor, maxFixed, maxT27,
              nFloorInflated, 100.0*nFloorInflated/nNodes,
              nFixedInflated, 100.0*nFixedInflated/nNodes,
              nT27Inflated, 100.0*nT27Inflated/nNodes,
              t27Coef, t27MaxSoftening);
    }
  }

  //Set the kernel parameters, many!
  // T40: the kernel's stop condition is `bid >= n_active_groups`, so passing a reduced value
  // bounds a dispatch to a chunk. set_args stores the POINTER (my_cuda_rt.h), so this variable is
  // re-read at every launch and can be advanced between chunks without re-issuing the arguments.
  int t40GroupEnd = tree.n_active_groups;
  approxGrav.set_args(0, &t40GroupEnd,
                         &tree.n,
                         tree.bodies_forceSoftening.p(), // Phase 5 ticket 03: per-particle Gadget-2
                                                          // ForceSoftening, replacing flat eps2
                         tree.bodies_typeDevice.p(), // Phase 5 ticket 07: per-target type, selects
                                                      // the Rcut/Asmth pair add_acc() uses
                         &node_begend,
                         tree.active_group_list.p(),
                         tree.bodies_Ppos.p(),
                         tree.multipole.p(),
                         tree.bodies_acc1.p(),
                         tree.bodies_Ppos.p(),
                         tree.ngb.p(),
                         tree.activePartlist.p(),
                         tree.interactions.p(),
                         tree.boxSizeInfo.p(),
                         tree.groupSizeInfo.p(),
                         tree.boxCenterInfo.p(),
                         tree.groupCenterInfo.p(),
                         tree.bodies_Pvel.p(),
                         tree.generalBuffer1.p(),  //The buffer to store the tree walks
                         tree.bodies_h.p(),        //Per particle search radius
                         tree.bodies_dens.p(),     //Per particle density (x) and nnb (y)
                         tree.groupMaxAccInfo.p(), //Phase 3: Springel MAC per-group |a_old|
                         &(this->errTolForceAcc),
                         tree.cellSizeInfo.p(),
                         tree.nodeSoftInfo.p(),   // T28/C-A-04    //T26: per-node fixed octree-cell size
                         tree.groupMaxSofteningInfo.p(),  //T27: per-group max ForceSoftening, softening-based bJ floor
                         tree.cellCenterInfo.p());        //C-A-01(b): per-node fixed octree-cell centre

  // Ticket T10: env-var-gated override of the tree-walk kernel's actual runtime launch
  // dimensions, independent of the NTHREAD compile-time template parameter (which still governs
  // shared-memory sizing via BLOCKDIM2=NTHREAD2, see dev_approximate_gravity_warp_new.cu). Used to
  // test whether the kernel can run with fewer than a full 32-lane warp -- the __shfl_sync/
  // __ballot_sync calls throughout the kernel assume a full warp and are not templated on lane
  // count. Zero-cost when unset (falls back to the original NTHREAD/nBlocksForTreeWalk values).
  {
    static int t10_launchThreads = -1;
    static int t10_launchBlocks  = -1;
    if (t10_launchThreads == -1) {
      const char *e = getenv("GADGET_HIP_LAUNCH_THREADS");
      t10_launchThreads = e ? atoi(e) : 0;
    }
    if (t10_launchBlocks == -1) {
      const char *e = getenv("GADGET_HIP_LAUNCH_BLOCKS");
      t10_launchBlocks = e ? atoi(e) : 0;
    }
    const int nThreadsUse = t10_launchThreads > 0 ? t10_launchThreads : NTHREAD;
    const int nBlocksUse  = t10_launchBlocks  > 0 ? t10_launchBlocks  : nBlocksForTreeWalk;
    if (t10_launchThreads > 0 || t10_launchBlocks > 0) {
      fprintf(stderr, "[T10] approxGrav launch override: threads=%d blocks=%d (default NTHREAD=%d nBlocksForTreeWalk=%d)\n",
              nThreadsUse, nBlocksUse, NTHREAD, nBlocksForTreeWalk);
    }
    approxGrav.setWork(-1, nThreadsUse, nBlocksUse);
  }

  hipEventRecord(startLocalGrav, gravStream->s());
  // T31 diagnostic: time the walk kernel itself, so a stall is attributed to the kernel rather
  // than inferred from where the surrounding log lines stop.
  {
    static bool t31g = false;
    if (!t31g) { t31g = true;
      const char *e = getenv("GADGET_HIP_T31_FORCE_GEO");
      t31_set_force_geo(e ? atoi(e) : 0); }
  }
  walk_reset_bailouts();
  walk_reset_guard_window();   // T49 item A: did the old guard's blind window fire this step?
  walk_reset_high_water();     // Phase 5: per-step frontier margin
  walk_reset_retries();        // Phase 5 / T49 item B: big-stack retries, and whether any failed
  walk_reset_zoom_groups();    // T50 section 2: per-group Rcut[1] pruning, pure vs mixed
  {
    static bool t50set = false;
    if (!t50set) { t50set = true;
      const char *e = getenv("GADGET_HIP_T50_NO_PURE_PRUNE");
      t50_set_no_pure_prune(e ? atoi(e) : 0); }
  }
  {
    // T55: Rcut pruning geometry. Default 1 (per-axis, Gadget's), not 0 -- the Euclidean sphere
    // demonstrably loses short-range force Gadget keeps. 0 is retained only so the change can be
    // A/B'd in one binary.
    static bool t55set = false;
    if (!t55set) { t55set = true;
      const char *e = getenv("GADGET_HIP_T55_RCUT_MODE");
      const int m = e ? atoi(e) : 1;
      t55_set_rcut_mode(m);
      if (e) fprintf(stderr, "[T55] Rcut prune geometry = %d (%s)\n", m,
                     m == 0 ? "Euclidean sphere at Rcut (pre-T55)"
                            : (m == 2 ? "per-axis box at 6*Asmth" : "per-axis box at Rcut (Gadget)"));
    }
  }
  cb15_reset_counters();       // per-step forced-descent count, not cumulative
#ifdef UNEQUALSOFTENINGS
  {
    static const int t28off = (getenv("GADGET_HIP_T28_DISABLE") != NULL)
                                ? atoi(getenv("GADGET_HIP_T28_DISABLE")) : 0;
    static bool t28set = false;
    if (!t28set) { t28_set_disable(t28off); t28set = true; }
  }
  cb15_reset_counters();
#endif
  const bool t31_timeWalk = (getenv("GADGET_HIP_T31_TIME_WALK") != NULL);
  const double t31_walk0 = t31_timeWalk ? get_time() : 0.0;
  {
    // GADGET_HIP_WALK_CHUNKS: >0 forces a fixed chunk count, 0 disables chunking (one dispatch,
    // the pre-T40 behaviour and the A/B control), unset/-1 sizes adaptively.
    static const int t40Fixed = getenv("GADGET_HIP_WALK_CHUNKS")
                                  ? atoi(getenv("GADGET_HIP_WALK_CHUNKS")) : -1;
    static const double t40TargetMs = getenv("GADGET_HIP_WALK_CHUNK_MS")
                                  ? atof(getenv("GADGET_HIP_WALK_CHUNK_MS")) : 300.0;
    const int nAG = tree.n_active_groups;

    int nChunks = 1;
    if (t40Fixed > 0)      nChunks = t40Fixed;
    else if (t40Fixed < 0)
    {
      // No measurement yet on the first walk of a run. Falling back to a single dispatch there
      // leaves exactly the dispatch this ticket exists to bound -- measured at 3624 ms against a
      // 293 ms median once sizing kicks in -- so start pessimistic and let the measurement correct
      // it from the next step onward.
      nChunks = (g_t40LastWalkMs > 0.0f)
                  ? (int) ceil((double) g_t40LastWalkMs / t40TargetMs)
                  : 16;
    }
    if (nChunks < 1)  nChunks = 1;
    if (nChunks > 64) nChunks = 64;             // launch overhead is not free either
    if (nAG > 0 && nChunks > nAG) nChunks = nAG;

    if (nChunks <= 1 || nAG <= 0)
    {
      t40GroupEnd = nAG;
      approxGrav.execute2(gravStream->s());     //First half
    }
    else
    {
      // The counter lives at activePartlist[n] and the retry lock at [n+1] (build.cpp sizes the
      // buffer n+2). activePartlist is ALSO the walk's output mask, so only the counter slot may
      // be touched between chunks -- re-zeroing the array would erase the previous chunks' output.
      // Presetting is required rather than letting the counter carry over: blocks that race past
      // the end still increment it, so it overshoots and would skip groups.
      static uint t40Lo[65];
      // .p() yields the address OF the device pointer (that is what set_args consumes); the device
      // pointer itself comes from get_devMem(). Doing pointer arithmetic on .p() targets host
      // memory and hipMemcpyAsync rejects it with "invalid argument".
      uint *counter = ((uint *) tree.activePartlist.get_devMem()) + tree.n;
      for (int c = 0; c < nChunks; ++c)
      {
        t40Lo[c]    = (uint) ((long long) nAG * c / nChunks);
        t40GroupEnd = (int)  ((long long) nAG * (c + 1) / nChunks);
        CU_SAFE_CALL(hipMemcpyAsync(counter, &t40Lo[c], sizeof(uint),
                                    hipMemcpyHostToDevice, gravStream->s()));
        approxGrav.execute2(gravStream->s());
      }
    }
  }
  if (t31_timeWalk)
  {
    gravStream->sync();
    fprintf(stderr, "[T31] approxGrav kernel iter=%d: %.3f s (n_active_groups=%d)\n",
            iter, get_time() - t31_walk0, tree.n_active_groups);
  }
  // A group whose walk overflowed the cell-list stack got NO force this step, and the flag saying
  // so was previously discarded (see g_walk_bailouts). Silently integrating a partial force is the
  // worst failure mode available here, so refuse -- same argument as C-D-14's non-advancing step.
  {
    gravStream->sync();
#ifdef UNEQUALSOFTENINGS
    // T28/C-B-15: report how often the mixed-softening forced descent fired. Zero on a
    // single-species run is correct; zero on a multi-species run means the rule is inert and must
    // be investigated, not assumed fine.
    if (getenv("GADGET_HIP_T28_STATS"))
    {
      unsigned int forced = 0, undef = 0;
      cb15_read_counters(&forced, &undef);
      fprintf(stderr, "[T28] iter=%d forced_descents=%u undefined_nodes=%u\n", iter, forced, undef);
    }
#endif
    // T49 item A, reported UNCONDITIONALLY: a nonzero count is not a diagnostic, it is the
    // statement that a pre-fix build would have wrapped the cell-list ring buffer on this step and
    // walked cell indices belonging to the wrong level. The corrected guard turns that into an
    // honest overflow, which the bailout handling just below then deals with. First few
    // occurrences and then every 50th, so a run that trips it constantly stays readable.
    {
      static unsigned long long guardWindowSteps = 0, guardWindowTotal = 0;
      const unsigned int gw = walk_read_guard_window();
      if (gw > 0)
      {
        guardWindowSteps++;
        guardWindowTotal += gw;
        if (guardWindowSteps <= 5 || (guardWindowSteps % 50) == 0)
          fprintf(stderr, "[T49-A] iter=%d: %u group-level(s) exceeded the cell-list ring buffer by "
                          "an amount only the corrected guard sees (the pre-fix guard omitted "
                          "nextLevelCellCounter and would have wrapped it silently). Steps affected "
                          "so far: %llu, total occurrences: %llu\n",
                  iter, gw, guardWindowSteps, guardWindowTotal);
      }
    }

    // Phase 5 (GADGET_HIP_CELL_WATER=1 for every step; otherwise only when something is abnormal).
    // A big-stack retry is NEVER normal: it serialises that group through a GPU-wide spin lock, so
    // one retry can cost more wall time than the whole rest of the step. Report it unconditionally.
    {
      static const bool cellWater = (getenv("GADGET_HIP_CELL_WATER") != NULL);
      unsigned int hw = 0, hwOld = 0, forced = 0, undef = 0, retries = 0, retryFail = 0;
      walk_read_high_water(&hw, &hwOld);
      cb15_read_counters(&forced, &undef);
      walk_read_retries(&retries, &retryFail);
      unsigned int zPure = 0, zMixed = 0, zCoarse = 0;
      walk_read_zoom_groups(&zPure, &zMixed, &zCoarse);
      const unsigned int cap = walk_cell_list_capacity();
      // The percentage is over the groups that CONTAIN a high-res target, because those are the only
      // ones the Rcut[1] prune could ever apply to. `coarse` groups hold no high-res particle at all
      // and are already using the right Rcut -- counting them as "not pure" (which the first
      // two-bucket version did) made the genuinely-mixed boundary shell look ~3x larger than it is.
      if (cellWater || retries > 0 || retryFail > 0)
        fprintf(stderr, "[CELL-WATER] iter=%d frontier=%u/%u = %.1f%% (old_expr=%u)  "
                        "forced_descents=%u undefined_soft=%u  big_stack_retries=%u retry_failed=%u"
                        "  zoom_groups pure=%u mixed=%u coarse=%u (%.1f%% of high-res-bearing "
                        "groups pure)%s\n",
                iter, hw, cap, cap ? 100.0 * hw / cap : 0.0, hwOld,
                forced, undef, retries, retryFail,
                zPure, zMixed, zCoarse,
                (zPure + zMixed) ? 100.0 * zPure / (zPure + zMixed) : 0.0,
                retries > 0 ? "   <== SERIALISED through the big-stack spin lock" : "");
    }

    const unsigned int nBail = walk_read_bailouts();
    if (nBail > 0)
    {
      fprintf(stderr,
        "\nFATAL: %u group walk(s) ran out of cell-list stack this step and produced NO force for\n"
        "their particles. CELL_LIST_MEM_PER_WARP (include/node_specs.h) is too small for this tree.\n"
        "The forces for this step are incomplete; refusing to integrate them.\n"
        "  iter=%d  n_active_groups=%d / %d\n\n",
        nBail, iter, tree.n_active_groups, tree.n_groups);
      ::exit(1);
    }
  }
  hipEventRecord(endLocalGrav, gravStream->s());


#if 0
	//Print density information
	tree.bodies_dens.d2h();
	tree.bodies_Ppos.d2h();
	tree.bodies_h.d2h();

	int nnbMin = 10e7;
	int nnbMax = -10e7;
	int nnbSum = 0;

	static bool firstIter0 = true;
	for(int i=0; i < tree.n; i++)
	{
		float r = sqrt(pow(tree.bodies_Ppos[i].x,2) + pow(tree.bodies_Ppos[i].y, 2) + pow(tree.bodies_Ppos[i].z,2));

		nnbMin =  std::min(nnbMin, (int)tree.bodies_dens[i].y);
		nnbMax =  std::max(nnbMax, (int)tree.bodies_dens[i].y);
		nnbSum += (int)tree.bodies_dens[i].y;
if(firstIter0 == true || iter == 40){
		fprintf(stderr, "DENS Iter: %d\t%d\t%f\t%f\t%f\tr: %f\th: %f\td: %f\tnnb: %f\t logs: %f %f  \n",
			iter,
			i, tree.bodies_Ppos[i].x, tree.bodies_Ppos[i].y, tree.bodies_Ppos[i].z,
			r,
			tree.bodies_h[i],
			tree.bodies_dens[i].x, tree.bodies_dens[i].y,
			log10(tree.bodies_dens[i].x), log2(tree.bodies_dens[i].x)
			);
}

	}
		firstIter0 = false;
		fprintf(stderr,"STATD Iter: %d\tMin: %d\tMax: %d\tAvg: %f\n", iter, nnbMin, nnbMax, nnbSum / (float)tree.n);
//	exit(0);
#endif



  //Print interaction statistics
  #if 0
  tree.body2group_list.d2h();
  tree.interactions.d2h();
    long long directSum = 0;
    long long apprSum = 0;
    long long directSum2 = 0;
    long long apprSum2 = 0;


    int maxDir = -1;
    int maxAppr = -1;

    for(int i=0; i < tree.n; i++)
    {
      apprSum     += tree.interactions[i].x;
      directSum   += tree.interactions[i].y;

      maxAppr = max(maxAppr,tree.interactions[i].x);
      maxDir  = max(maxDir,tree.interactions[i].y);

      apprSum2     += tree.interactions[i].x*tree.interactions[i].x;
      directSum2   += tree.interactions[i].y*tree.interactions[i].y;

//      if(i < 35)
//      fprintf(stderr, "%d\t Direct: %d\tApprox: %d\t Group: %d \n",
//              i, tree.interactions[i].y, tree.interactions[i].x,
//              tree.body2group_list[i]);
    }
    cout << "Interaction at (rank= " << mpiGetRank() << " ) iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    cout << "avg dir: " << directSum / tree.n << "\tavg appr: " << apprSum / tree.n << "\tMaxdir: " << maxDir << "\tmaxAppr: " << maxAppr <<  endl;
    cout << "sigma dir: " << sqrt((directSum2  - directSum)/ tree.n) << "\tsigma appr: " << std::sqrt((apprSum2 - apprSum) / tree.n)  <<  endl;

  #endif


  if(mpiGetNProcs() == 1) //Only do it here if there is only one process
  {
   //#ifdef DO_BLOCK_TIMESTEP
  #if 0 //Demo mode
      //Reduce the number of valid particles
      getNActive.set_arg<int>(0,    &tree.n);
      getNActive.set_arg<cl_mem>(1, tree.activePartlist.p());
      getNActive.set_arg<cl_mem>(2, this->nactive.p());
      getNActive.set_arg<int>(3,    NULL, 128); //Dynamic shared memory , equal to number of threads
      getNActive.setWork(-1, 128,   NBLOCK_REDUCE);

      //JB Need a sync here This is required otherwise the gravity overlaps the reduction
      //and we get incorrect numbers.
      //Note Disabled this whole function for demo!
      gravStream->sync();
      getNActive.execute(execStream->s());



      //Reduce the last parts on the host
      this->nactive.d2h();
      tree.n_active_particles = this->nactive[0];
      for (int i = 1; i < NBLOCK_REDUCE ; i++)
          tree.n_active_particles += this->nactive[i];

      LOG("Active particles: %d \n", tree.n_active_particles);
    #else
      tree.n_active_particles = tree.n;
      LOG("Active particles: %d \n", tree.n_active_particles);
    #endif
  }
}
//end approximate


void octree::approximate_gravity_let(tree_structure &tree, tree_structure &remoteTree, int bufferSize, bool doActiveParticles)
{
  //Start and end node of the remote tree structure
  uint2 node_begend;
  node_begend.x =  0;
  node_begend.y =  remoteTree.remoteTreeStruct.w;

  //The texture offset used:
  int nodeTexOffset     = remoteTree.remoteTreeStruct.z ;

  //The start and end of the top nodes:
  node_begend.x = (remoteTree.remoteTreeStruct.w >> 16);
  node_begend.y = (remoteTree.remoteTreeStruct.w & 0xFFFF);

  //Number of particles and number of nodes in the remote tree
  int remoteP = remoteTree.remoteTreeStruct.x;
  int remoteN = remoteTree.remoteTreeStruct.y;

  LOG("LET node begend [%d]: %d %d iter-> %d\n", procId, node_begend.x, node_begend.y, iter);

  void *multiLoc = remoteTree.fullRemoteTree.a(1*(remoteP) + 2*(remoteN+nodeTexOffset));
  void *boxSILoc = remoteTree.fullRemoteTree.a(1*(remoteP));
  void *boxCILoc = remoteTree.fullRemoteTree.a(1*(remoteP) + remoteN + nodeTexOffset);

  approxGravLET.set_args(0,
                         &tree.n_active_groups,
                         &tree.n,
                         tree.bodies_forceSoftening.p(), // Phase 5 ticket 03: see approxGrav above
                         tree.bodies_typeDevice.p(),     // Phase 5 ticket 07: see approxGrav above
                         &node_begend,
                         tree.active_group_list.p(),
                         remoteTree.fullRemoteTree.p(),
                         &multiLoc,
                         tree.bodies_acc1.p(),
                         tree.bodies_Ppos.p(),
                         tree.ngb.p(),
                         tree.activePartlist.p(),
                         tree.interactions.p(),
                         &boxSILoc,
                         tree.groupSizeInfo.p(),
                         &boxCILoc,
                         tree.groupCenterInfo.p(),
                         tree.bodies_Pvel.p(),      //<- Predicted local body velocity
                         tree.generalBuffer1.p(),  //The buffer to store the tree walks
                         tree.bodies_h.p(),        //Per particle search radius
                         tree.bodies_dens.p());    //Per particle density (x) and nnb (y)

  approxGravLET.setWork(-1, NTHREAD, nBlocksForTreeWalk);

  if(letRunning)
  {
    //don't want to overwrite the data of previous LET tree
    gravStream->sync();

    //Add the time to the time sum for the let
    float msLET;
    CU_SAFE_CALL(hipEventElapsedTime(&msLET,startRemoteGrav, endRemoteGrav));
    runningLETTimeSum += msLET;
  }

  remoteTree.fullRemoteTree.h2d(bufferSize); //Only copy required data
  tree.activePartlist.zeroMemGPUAsync(gravStream->s()); //Resets atomics

  CU_SAFE_CALL(hipEventRecord(startRemoteGrav, gravStream->s()));
  approxGravLET.execute2(gravStream->s());
  CU_SAFE_CALL(hipEventRecord(endRemoteGrav, gravStream->s()));
  letRunning = true;


 //Print interaction statistics
  #if 0
    tree.interactions.d2h();
//     tree.body2group_list.d2h();

    long long directSum = 0;
    long long apprSum = 0;

    int maxDir = -1;
    int maxAppr = -1;

    long long directSum2 = 0;
    long long apprSum2 = 0;


    for(int i=0; i < tree.n; i++)
    {
      apprSum     += tree.interactions[i].x;
      directSum   += tree.interactions[i].y;

      maxAppr = max(maxAppr,tree.interactions[i].x);
      maxDir  = max(maxDir, tree.interactions[i].y);

      apprSum2     += (tree.interactions[i].x*tree.interactions[i].x);
      directSum2   += (tree.interactions[i].y*tree.interactions[i].y);
    }

    cout << "Interaction (LET) at (rank= " << mpiGetRank() << " ) iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    cout << "avg dir: " << directSum / tree.n << "\tavg appr: " << apprSum / tree.n  << "\tMaxdir: " << maxDir << "\tmaxAppr: " << maxAppr <<  endl;
    cout << "sigma dir: " << sqrt((directSum2  - directSum)/ tree.n) << "\tsigma appr: " << std::sqrt((apprSum2 - apprSum) / tree.n)  <<  endl;
  #endif

  if(doActiveParticles) //Only do it here if there is only one process
  {
   //#ifdef DO_BLOCK_TIMESTEP
  #if 0 //Demo mode
      //Reduce the number of valid particles
      getNActive.set_arg<int>(0,    &tree.n);
      getNActive.set_arg<cl_mem>(1, tree.activePartlist.p());
      getNActive.set_arg<cl_mem>(2, this->nactive.p());
      getNActive.set_arg<int>(3,    NULL, 128); //Dynamic shared memory , equal to number of threads
      getNActive.setWork(-1, 128,   NBLOCK_REDUCE);

      //JB Need a sync here This is required otherwise the gravity overlaps the reduction
      //and we get incorrect numbers.
      //Note Disabled this whole function for demo!
      gravStream->sync();
      getNActive.execute(execStream->s());



      //Reduce the last parts on the host
      this->nactive.d2h();
      tree.n_active_particles = this->nactive[0];
      for (int i = 1; i < NBLOCK_REDUCE ; i++)
          tree.n_active_particles += this->nactive[i];

      LOG("Active particles: %d \n", tree.n_active_particles);
    #else
      tree.n_active_particles = tree.n;
      LOG("Active particles: %d \n", tree.n_active_particles);
    #endif
  }
}
//end approximate



void octree::correct(tree_structure &tree)
{
  //TODO this might be moved to the gravity call where we have that info anyway?
  tree.n_active_particles = tree.n;
  #ifdef DO_BLOCK_TIMESTEP
    //Reduce the number of valid particles
    gravStream->sync(); //Sync to make sure that the gravity phase is finished
//    getNActive.set_arg<int>(0,    &tree.n);
//    getNActive.set_arg<cl_mem>(1, tree.activePartlist.p());
//    getNActive.set_arg<cl_mem>(2, this->nactive.p());
//    getNActive.set_arg<int>(3,    NULL, 128); //Dynamic shared memory , equal to number of threads
    getNActive.set_args(sizeof(int)*128, &tree.n, tree.activePartlist.p(), this->nactive.p());
    getNActive.setWork(-1, 128,   NBLOCK_REDUCE);
    getNActive.execute2(execStream->s());

    //Reduce the last parts on the host
    this->nactive.d2h();
    tree.n_active_particles = this->nactive[0];
    for (int i = 1; i < NBLOCK_REDUCE ; i++)
        tree.n_active_particles += this->nactive[i];
  #endif
  LOG("Active particles: %d \n", tree.n_active_particles);


  my_dev::dev_mem<int2>    timeBuffer;    // Phase 2: bodies_time is ticks
  my_dev::dev_mem<real4>   real4Buffer1;
  my_dev::dev_mem<int>     newEndBuffer;   // Phase 2: the new Ti_endstep, in ticks

  int memOffset = timeBuffer.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
      memOffset = real4Buffer1.cmalloc_copy(tree.generalBuffer1, tree.n, memOffset);
      memOffset = newEndBuffer.cmalloc_copy(tree.generalBuffer1, tree.n, memOffset);

  // Gadget-2 time integration rework: compute_dt() must run BEFORE correctParticles() now,
  // because correctParticles()'s kick (timestep.c:254-282's midpoint-to-midpoint formula) needs
  // to know the size of the NEW step (to find the new step's midpoint) while the OLD
  // Ti_begstep/Ti_endstep pair in tree.bodies_time is still intact -- previously compute_dt ran
  // second and overwrote tree.bodies_time in place, which is what made the old (wrong) kick
  // formula only able to see the just-completed step's own length. compute_dt now writes the new
  // end-of-step time into newEndBuffer instead of touching tree.bodies_time at all; correctParticles
  // reads both the untouched old time[] and newEndBuffer and commits the final (begin,end) pair
  // itself. It also now reads the freshly-computed force straight from tree.bodies_acc1 (this
  // step's raw gravity output, valid for active particles) rather than tree.bodies_acc0, since
  // acc0 hasn't been updated with this step's force yet at this point in the pipeline.
  #ifdef DO_BLOCK_TIMESTEP
    // Phase 5 ticket 04 (PLAN.md): timestep.c:47-60's fac1/atime/hubble_a -- cheap host scalar
    // math, recomputed every step (unlike the displacement ceiling, which is expensive enough to
    // only recompute per tree-rebuild, see recomputeTimestepGlobals()).
    float ts_errTolIntAccuracy, ts_atime, ts_fac1, ts_hubble_a, ts_maxSizeTimestep, ts_minSizeTimestep;
    if (haveGadgetParams)
    {
      const double time = (double) t_current;
      ts_errTolIntAccuracy = (float) gadgetParams.ErrTolIntAccuracy;
      ts_maxSizeTimestep   = (float) gadgetParams.MaxSizeTimestep;
      ts_minSizeTimestep   = (float) gadgetParams.MinSizeTimestep;
      if (gadgetParams.ComovingIntegrationOn)
      {
        ts_atime    = (float) time;
        ts_fac1     = (float) (1.0 / (time * time));
        ts_hubble_a = (float) gadget_hubble_a(gadgetParams.Omega0, gadgetParams.OmegaLambda,
                                               gadgetParams.Hubble, time);
      }
      else
      {
        ts_atime = ts_fac1 = ts_hubble_a = 1.0f;
      }
    }
    else
    {
      // No --param config: no Gadget-2 accuracy/size parameters exist to drive the real
      // criterion -- fall back to values that make it degrade to a fixed step of `timeStep`
      // (this port's own pre-ticket CLI default), keeping every existing dev/diagnostic run's
      // behavior unchanged. ErrTolIntAccuracy=0 -> dt=0 -> immediately clamped up to
      // MinSizeTimestep=timeStep; MaxSizeTimestep=timeStep too so the two clamps agree.
      ts_errTolIntAccuracy = 0.0f;
      ts_atime = ts_fac1 = ts_hubble_a = 1.0f;
      ts_maxSizeTimestep = ts_minSizeTimestep = timeStep;
    }
    float ts_dtDisplacement = (float) this->gadgetDtDisplacement;
    // Phase 5 ticket 08 (PLAN.md): same comoving-vs-not idiom already used by correctParticles's
    // own `correctComovingFlag` right below -- see compute_dt's own comment (timestep.cu) for why
    // this now has to know whether `dt` is delta-LOG(a) (needs exp()) or linear physical time.
    int ts_comovingFlag = haveGadgetComovingTables ? 1 : 0;

    // C-D-04: the ladder is a power-of-2 subdivision of the run's own time axis, log-spaced under
    // comoving integration and linear otherwise -- exactly Gadget-2's Timebase_interval (init.c:51/56).
    float ts_timelineSpan = 0.0f;
    if (haveGadgetParams)
      ts_timelineSpan = ts_comovingFlag
          ? (float)(gadgetLogTimeMax - gadgetLogTimeBegin)
          : (float)(gadgetParams.TimeMax - gadgetParams.TimeBegin);
    // T25: the snap needs the axis ORIGIN as well as its length -- same convention as
    // ts_timelineSpan (log(a) under comoving, linear t otherwise).
    // C-D-10: end of the simulated timespan, so compute_dt can truncate the last step onto it
    // exactly as Gadget-2 does. 0 disables the clamp (no --param, hence no TimeMax).
    float ts_timeMax = haveGadgetParams ? (float) gadgetParams.TimeMax : 0.0f;
    // T47: bodies_acc1 reaching compute_dt is already G-scaled (pm_scale_acc_masked, :1784) while
    // GravPM is not, so the kernel must apply G to the long-range term. Not const: set_args stores a
    // non-const void* and dereferences it at launch.
    // GADGET_HIP_T47_NO_G=1 restores the old (wrong) behaviour, so the effect of this one fix can be
    // measured on its own -- same idiom as GADGET_HIP_T4_FIX_G and GADGET_HIP_T45_KEEP_ACC0_D2H.
    static const bool t47NoG = (getenv("GADGET_HIP_T47_NO_G") != NULL);
    static bool t47Reported = false;
    GCarrier ts_pmG = GCarrier::just_G(
        (!t47NoG && !useDirectGravity && haveGadgetParams && gadgetParams.G > 0.0)
          ? (float) gadgetParams.G : 1.0f);
    if (!t47Reported && s_pmCadenceActive)
    {
      t47Reported = true;
      fprintf(stderr, "[T47] timestep criterion: long-range term scaled by G = %.6f%s\n",
              (double) ts_pmG.v, t47NoG ? "  (GADGET_HIP_T47_NO_G=1: the pre-fix behaviour)" : "");
    }
    float ts_timelineOrigin = 0.0f;
    if (haveGadgetParams)
      ts_timelineOrigin = ts_comovingFlag ? (float) gadgetLogTimeBegin
                                          : (float) gadgetParams.TimeBegin;
    computeDt.set_args(0, &tree.n, &t_current, &ts_errTolIntAccuracy, &ts_atime, &ts_fac1,
                          &ts_hubble_a, &ts_maxSizeTimestep, &ts_minSizeTimestep, &ts_dtDisplacement,
                          &ts_comovingFlag,
                          (void *) newEndBuffer.as<CurrentOrder>(),
                          (void *) tree.bodies_time.as<OriginalOrder>(),
                          (void *) tree.oriParticleOrder.as_const<CurrentOrder>(),
                          (void *) tree.bodies_acc1.as_const<CurrentOrder>(),
                          (void *) tree.bodies_forceSoftening.as_const<CurrentOrder>(),
                          (void *) tree.activePartlist.as_const<CurrentOrder>(),
                          &ts_timelineSpan, &ts_timelineOrigin,
                          &ts_timeMax,
                          &gadgetDPerTick, &Ti_current,   // C-D-10
                          // Phase 3: GADGET sizes the step from GravAccel + GravPM
                          // (timestep.c:443-445). With the cadence off acc1 already holds both and
                          // these are NULL, which is what the kernel tests.
                          s_pmCadenceActive ? (void *) tree.bodies_ids.as_const<CurrentOrder>()
                                            : (void *) &s_nullDevPtr,
                          s_pmCadenceActive ? (void *) &s_gravPMx : &s_nullDevPtr,
                          s_pmCadenceActive ? (void *) &s_gravPMy : &s_nullDevPtr,
                          s_pmCadenceActive ? (void *) &s_gravPMz : &s_nullDevPtr,
                          &ts_pmG);
    // C-D-14: Gadget-2 treats an unrepresentable (non-advancing) timestep as a hard error --
    // endrun(818), timestep.c:537-552. The port already counted the condition in compute_dt, but
    // t7_reset_nonadvancing()/t7_read_nonadvancing() were declared and defined and NEVER CALLED,
    // so the counter incremented on the GPU and nothing ever looked at it: the comment on the
    // counter claimed "the host turns it into a clear abort" while the run would in fact hang.
    {
      static const int injectStall = (getenv("GADGET_HIP_T7_INJECT") != NULL) ? 1 : 0;
      static bool      injectSet   = false;
      if (!injectSet) { t7_set_inject_stall(injectStall); injectSet = true; }
    }
    {
      // C-D-13 A/B: GADGET_HIP_CD13=0 turns the SYNCHRONIZATION rule off in the SAME binary, so a
      // before/after comparison cannot be confounded by a rebuild.
      static const int cd13On = (getenv("GADGET_HIP_CD13") != NULL)
                                  ? atoi(getenv("GADGET_HIP_CD13")) : 1;
      static bool cd13Set = false;
      if (!cd13Set) { cd13_set_enabled(cd13On); cd13Set = true; }
    }
    t7_reset_nonadvancing();
    cd13_reset_blocked();
    computeDt.setWork(tree.n, 128);
    computeDt.execute2(execStream->s());
    execStream->sync();
    {
      // C-D-13 instrument (GADGET_HIP_CD13_STATS=1): report how often the SYNCHRONIZATION rule
      // actually blocked a growth. Zero every step means either the ladder is already commensurate
      // or the rule is inert -- and those must not be confused (PLAN.md rule 8).
      static const bool cd13Stats = (getenv("GADGET_HIP_CD13_STATS") != NULL);
      if (cd13Stats)
      {
        unsigned int reached = 0, grow = 0, blocked = 0;
        cd13_read_detail(&reached, &grow, &blocked);
        fprintf(stderr, "[CD13] iter=%d reached=%u grow_attempts=%u blocked=%u span=%.6g\n",
                iter, reached, grow, blocked, ts_timelineSpan);
      }

      const unsigned int nStuck = t7_read_nonadvancing();
      if (nStuck > 0)
      {
        int   oIdx = -1;
        float oTc = 0.0f, oEnd = 0.0f, oDt = 0.0f, oAc = 0.0f;
        t7_read_first_offender(&oIdx, &oTc, &oEnd, &oDt, &oAc);
        fprintf(stderr,
                "\nFATAL (C-D-14): a timestep that cannot advance the clock was assigned to %u "
                "particle(s).\nWe better stop -- this is Gadget-2's endrun(818) condition "
                "(timestep.c:537-552).\n"
                "  first offender: idx=%d  t_current=%.9g  newEnd=%.9g  dt=%.9g  |a|=%.9g\n"
                "  iter=%d  n=%d  active_groups=%d/%d\n\n",
                nStuck, oIdx, oTc, oEnd, oDt, oAc, iter, tree.n,
                tree.n_active_groups, tree.n_groups);
        ::exit(1);
      }
    }
  #endif

  // Phase 5 ticket 05 (PLAN.md): see predict()'s own comment on why comovingIntegrationOn is a
  // non-const local passed by address.
  // ------------------------------------------------------------------------------------------
  // Phase 3: GADGET-2's long-range kick (timestep.c:345-384).
  //
  // Runs BEFORE correct_particles, where the index spaces are still the documented ones
  // (bodies_vel in ORIGINAL order, reached through oriParticleOrder; ids in CURRENT order).
  // correct_particles resets oriParticleOrder to identity at the end of its own kernel, so after
  // it the translation this kernel needs no longer exists. Order does not matter physically --
  // the two kicks are independent additive impulses on the same velocity.
  // Phase 3 bisect (GADGET_HIP_PM_NO_KICK=1): run everything except the impulse. With the kick
  // off and PM excluded from acc1 the run has NO long-range force at all, which is the reference
  // for "how bad is missing PM" -- if the cadence matches that, the kick is contributing nothing.
  static const bool pmNoKick = (getenv("GADGET_HIP_PM_NO_KICK") != NULL);
  if (s_pmCadenceActive && !pmNoKick && Ti_current >= s_PM_Ti_endstep)   // ">=": see pmDue
  {
    // The next PM interval: the largest power-of-two tick count not exceeding dt_displacement,
    // with GADGET's own growth rule -- lengthen only if an integer number of the new steps still
    // reaches the end of the timeline, otherwise stay put (timestep.c:348-357).
    gadget_tick_t ti_step = (gadgetDtDisplacement > 0.0 && gadgetDPerTick > 0.0)
        ? gadget_tick_pow2_floor(gadgetDtDisplacement / gadgetDPerTick)
        : (gadget_tick_t) 1;
    // GADGET_HIP_PM_INTERVAL_SHIFT=k shortens the PM interval by 2^k. This is the convergence
    // knob, and it is what actually decides whether this operator is right.
    //
    // Comparing the cadence against PM-every-step cannot decide it: GADGET-2 uses the cadence
    // too, and with the long-range force drifting ~15% per interval (rung 2) ANY correct cadence
    // differs from every-step by a large factor. What distinguishes a correct midpoint kick is
    // that the difference is SECOND ORDER in the interval -- halve the interval, the deviation
    // falls ~4x. The retracted T36 operator (stale force folded into each particle's own step,
    // active particles only) does not converge that way, because it is not a discretisation of
    // the same operator at all.
    static const int pmShift = getenv("GADGET_HIP_PM_INTERVAL_SHIFT")
                             ? atoi(getenv("GADGET_HIP_PM_INTERVAL_SHIFT")) : 0;
    if (pmShift > 0) ti_step >>= pmShift;
    if (ti_step < 1) ti_step = 1;
    const gadget_tick_t oldSpan = s_PM_Ti_endstep - s_PM_Ti_begstep;
    if (oldSpan > 0 && ti_step > oldSpan &&
        ((GADGET_TIMEBASE - s_PM_Ti_endstep) % ti_step) != 0)
      ti_step = oldSpan;
    if (Ti_current == GADGET_TIMEBASE) ti_step = 0;   // the final step closes the interval

    // Midpoint to midpoint. This is what makes the long-range force a leapfrog kick rather than
    // an acceleration: on the first step PM_Ti_beg == PM_Ti_end == 0, so tstart == 0 and the
    // particle receives the half-kick that starts the sequence; on the last, ti_step == 0 closes
    // it with the matching half. Getting this wrong is invisible over a few steps and shows up as
    // a percent-level energy drift by z=0.
    // Anchor a mid-timeline start. Without this the first interval would be measured from tick 0
    // and its midpoint-to-midpoint factor would span everything since the beginning of the run.
    if (s_PM_Ti_begstep == 0 && s_PM_Ti_endstep == 0 && Ti_current > 0)
      s_PM_Ti_begstep = s_PM_Ti_endstep = Ti_current;
    const gadget_tick_t tiStart = (s_PM_Ti_begstep + s_PM_Ti_endstep) / 2;
    const gadget_tick_t tiEnd   = s_PM_Ti_endstep + ti_step / 2;
    const double aStart = gadgetTimeline.toTime(tiStart);
    const double aEnd   = gadgetTimeline.toTime(tiEnd);
    // Not const: set_args stores a non-const void* to whatever it is handed and dereferences
    // it at launch, so a const object cannot be passed through it.
    float dtGravKick = (float) gadget_get_gravkick_factor(s_gadgetTables, aStart, aEnd);

    // THE FACTOR G. The tree kernel and every PM call produce G-FREE accelerations -- PM is called
    // with /*gravityConstant=*/1.0f, hardcoded, mirroring GADGET-2's forcetree.c convention -- and
    // G is applied ONCE afterwards, to the summed bodies_acc1, by pm_scale_acc_masked() below.
    //
    // GravPM is stored BEFORE that scaling. With the cadence off, the long-range force sits inside
    // acc1 and is scaled along with everything else. With the cadence ON it never enters acc1, so
    // it never meets pm_scale_acc_masked at all, and this kick must apply G itself.
    //
    // Measured: acc1 read 3.262 in the PM block and 140.3 at correct_particles -- ratio 43.0,
    // exactly G = 43.007106. Without this the long-range impulse was 43x too small, which is why
    // the cadence reproduced "no PM at all" (-94.9% in Ekin) while still passing every check on
    // the impulse itself: right particle, right direction, right count, right time factor, and
    // the one scale factor nobody multiplied by.
    //
    // Same class as T4-FIX-G, where the whole per-step dynamics silently ran at G=1: invisible in
    // any test at G=1, because there "forgot to multiply by G" and "multiplied by 1" agree.
    if (!useDirectGravity && haveGadgetParams && gadgetParams.G > 0.0)
      dtGravKick = (float) (dtGravKick * gadgetParams.G);
    // Phase 3 probe: scale the long-range impulse. If the dynamics barely respond to a 10x kick,
    // the impulse is not reaching the trajectory and the size of the factor is beside the point;
    // if they respond proportionally, the plumbing is fine and the factor is what is wrong.
    static const double kickScale = getenv("GADGET_HIP_PM_KICK_SCALE")
                                  ? atof(getenv("GADGET_HIP_PM_KICK_SCALE")) : 1.0;
    dtGravKick = (float) (dtGravKick * kickScale);
    // The factor G is already inside dtGravKick, which is precisely what GCarrier asserts. Wrapping
    // it here rather than multiplying inside the kernel keeps the arithmetic v * (dt*G) -- the same
    // float it has always been -- while making `gpm[id] * someFloat` fail to compile.
    GCarrier dtGravKickG = GCarrier::gravkick_times_G(dtGravKick);

    s_PM_Ti_begstep = s_PM_Ti_endstep;
    s_PM_Ti_endstep = s_PM_Ti_begstep + ti_step;

    // Validation rung 1: one byte per particle, counting the long-range kicks it received this
    // interval. Every entry must be exactly 1. This is the assertion that would have caught the
    // retracted T36 outright -- there, inactive particles scored 0.
    static const bool kickAudit = (getenv("GADGET_HIP_PM_KICK_AUDIT") != NULL);
    s_pmKickNoPerm = (getenv("GADGET_HIP_PM_KICK_NOPERM") != NULL) ? 1 : 0;
    if (kickAudit && !s_pmKickAudit)
      CU_SAFE_CALL(hipMalloc((void **) &s_pmKickAudit, (size_t) tree.n * sizeof(unsigned char)));

    // bodies_Pvel, NOT bodies_vel. bodies_vel is a pure OUTPUT of correct_particles, which
    // reconstructs it wholesale from pVel every step:
    //     vel[idx] = pVel[unsortedIdx];            (inactive branch)
    //     float4 v = pVel[unsortedIdx]; ... kick    (active branch)
    // so an impulse written into bodies_vel here is overwritten a few microseconds later and has
    // no effect whatsoever. Measured exactly that way: quartering the PM interval changed the
    // final state only in the 7th significant digit, because the long-range kick was being
    // applied and then discarded. The index expression is unchanged -- pVel uses the same
    // unsorted[] space bodies_vel did -- only the array is wrong.
    static const bool kickBoth = (getenv("GADGET_HIP_PM_KICK_BOTH") != NULL);
    // Each buffer now NAMES the space it is being used in, at the point where the author decides
    // it. bodies_Pvel is not permuted by the per-step sort, so it is OriginalOrder here; bodies_ids
    // is permuted, so CurrentOrder; the GravPM store is id-indexed. See buffer_registry.h.
    gadgetPMKick.set_args(0, &tree.n,
                          (void *) tree.bodies_Pvel.as<OriginalOrder>(),
                          kickBoth ? (void *) tree.bodies_vel.as<OriginalOrder>()
                                   : (void *) &s_nullDevPtr,
                          (void *) tree.oriParticleOrder.as_const<CurrentOrder>(),
                          (void *) tree.bodies_ids.as_const<CurrentOrder>(),
                          (void *) &s_gravPMx, (void *) &s_gravPMy,
                          (void *) &s_gravPMz, &dtGravKickG,
                          kickAudit ? (void *) &s_pmKickAudit : &s_nullDevPtr,
                          &s_pmKickNoPerm);
    // Phase 3 probe (GADGET_HIP_PM_DV_PROBE=1): observe the velocity change the kick ACTUALLY
    // produces, instead of inferring it. Three probes were spent reasoning about an impulse that
    // every indirect check said was correct; this reads bodies_Pvel immediately before and after
    // the kernel and reports the rms difference. Expected |dv| = rms|GravPM| * dtGravKick.
    static const bool dvProbe = (getenv("GADGET_HIP_PM_DV_PROBE") != NULL);
    std::vector<real4> vBefore;
    if (dvProbe && s_pmKickCount < 3)
    {
      execStream->sync();
      tree.bodies_Pvel.d2h();
      vBefore.assign(&tree.bodies_Pvel[0], &tree.bodies_Pvel[0] + tree.n);
    }
    gadgetPMKick.setWork(tree.n, 128);
    gadgetPMKick.execute2(execStream->s());
    if (dvProbe && s_pmKickCount < 3)
    {
      execStream->sync();
      tree.bodies_Pvel.d2h();
      double sdv = 0.0, sv = 0.0; long changed = 0;
      for (int i = 0; i < tree.n; ++i)
      {
        const real4 a = vBefore[i], b = tree.bodies_Pvel[i];
        const double dx = (double)b.x-a.x, dy = (double)b.y-a.y, dz = (double)b.z-a.z;
        if (dx || dy || dz) changed++;
        sdv += dx*dx + dy*dy + dz*dz;
        sv  += (double)a.x*a.x + (double)a.y*a.y + (double)a.z*a.z;
      }
      fprintf(stderr, "[PM-DV-PROBE] interval %ld  dtGravKick=%.6e  rms|dv|=%.6e  rms|v_before|=%.6e"
                      "  changed=%ld/%d  |dv|/|v|=%.6e\n",
              s_pmKickCount + 1, dtGravKick, sqrt(sdv/tree.n), sqrt(sv/tree.n),
              changed, tree.n, sqrt(sdv/std::max(sv,1e-300)));
    }
    s_pmKickCount++;

    if (kickAudit)
    {
      execStream->sync();
      std::vector<unsigned char> h(tree.n);
      CU_SAFE_CALL(hipMemcpy(&h[0], s_pmKickAudit, (size_t) tree.n * sizeof(unsigned char),
                             hipMemcpyDeviceToHost));
      long zero = 0, one = 0, many = 0;
      for (int i = 0; i < tree.n; ++i)
      { if (h[i] == 0) zero++; else if (h[i] == 1) one++; else many++; }
      fprintf(stderr, "[PM-KICK-AUDIT] interval %ld  ti=[%d,%d)  dtGravKick=%.9g  "
                      "kicked-once=%ld  never=%ld  more-than-once=%ld  %s\n",
              s_pmKickCount, s_PM_Ti_begstep, s_PM_Ti_endstep, dtGravKick, one, zero, many,
              (zero == 0 && many == 0) ? "OK" : "*** IMPULSE ACCOUNTING VIOLATED ***");
      CU_SAFE_CALL(hipMemset(s_pmKickAudit, 0, (size_t) tree.n * sizeof(unsigned char)));
    }

    // Phase 3 probe (GADGET_HIP_PM_ACC_PROBE=1): how big is the long-range force compared with the
    // short-range one it was split out of? If |GravPM| is comparable to |acc1| then the kick, with
    // the analytically-correct factor, must matter as much as folding PM into acc1 did -- and it
    // demonstrably does not. Reading both back is cheap at this box size.
    static const bool accProbe = (getenv("GADGET_HIP_PM_ACC_PROBE") != NULL);
    if (accProbe && s_pmKickCount <= 3)
    {
      execStream->sync();
      tree.bodies_acc1.d2h();
      std::vector<float> gx(tree.n), gy(tree.n), gz(tree.n);
      // Diagnostic only; layout-identical, see the note at the other memcpy of these arrays.
      CU_SAFE_CALL(hipMemcpy(&gx[0], s_gravPMx, (size_t)tree.n*sizeof(GFreeAcc), hipMemcpyDeviceToHost));
      CU_SAFE_CALL(hipMemcpy(&gy[0], s_gravPMy, (size_t)tree.n*sizeof(GFreeAcc), hipMemcpyDeviceToHost));
      CU_SAFE_CALL(hipMemcpy(&gz[0], s_gravPMz, (size_t)tree.n*sizeof(GFreeAcc), hipMemcpyDeviceToHost));
      double sa = 0.0, sg = 0.0;
      for (int i = 0; i < tree.n; ++i)
      {
        const real4 a = tree.bodies_acc1[i];
        sa += (double)a.x*a.x + (double)a.y*a.y + (double)a.z*a.z;
        sg += (double)gx[i]*gx[i] + (double)gy[i]*gy[i] + (double)gz[i]*gz[i];
      }
      fprintf(stderr, "[PM-ACC-PROBE] interval %ld  rms|acc1(short)|=%.6e  rms|GravPM|=%.6e  "
                      "ratio PM/short=%.4f  dtGravKick=%.6e  rms dv=%.6e\n",
              s_pmKickCount + 1, sqrt(sa/tree.n), sqrt(sg/tree.n),
              sqrt(sg/std::max(sa,1e-300)), dtGravKick, sqrt(sg/tree.n)*dtGravKick);
    }

    static bool s_pmPhase3Logged = false;
    if (!s_pmPhase3Logged)
    {
      fprintf(stderr, "[PM] Phase 3 long-range kick ACTIVE: PM interval %d ticks (%.6g dloga), "
                      "one kick per particle per interval.\n",
              (int) ti_step, ti_step * gadgetDPerTick);
      s_pmPhase3Logged = true;
    }
  }

  int correctComovingFlag = haveGadgetComovingTables ? 1 : 0;
  // Spaces per include/buffer_registry.h. This is the Phase 3 kernel, so the buffers it WRITES are
  // named in the current space -- that write is what re-unifies the two -- while everything it reads
  // from the previous step's layout (bodies_acc0, bodies_Pvel, bodies_time) is original-space.
  // `pos` and `pPos` are deliberately the same buffer: bodies_pos is no longer a distinct array, so
  // `pos[idx] = pPos[idx]` inside the kernel is a self-copy.
  correctParticles.set_args(0, &tree.n, &t_current,
                            (void *) tree.bodies_time.as<OriginalOrder>(),
                            (void *) tree.activePartlist.as_const<CurrentOrder>(),
                            (void *) tree.bodies_vel.as<CurrentOrder>(),
                            (void *) tree.bodies_acc0.as_const<OriginalOrder>(),
                            (void *) tree.bodies_acc1.as_const<CurrentOrder>(),
                            (void *) tree.bodies_h.as<CurrentOrder>(),
                            (void *) tree.bodies_dens.as_const<CurrentOrder>(),
                            (void *) tree.bodies_Ppos.as<CurrentOrder>(),
                            (void *) tree.bodies_Ppos.as_const<CurrentOrder>(),
                            (void *) tree.bodies_Pvel.as_const<OriginalOrder>(),
                            (void *) tree.oriParticleOrder.as<CurrentOrder>(),
                            (void *) real4Buffer1.as<CurrentOrder>(),
                            (void *) timeBuffer.as<CurrentOrder>(),
                            &correctComovingFlag, gadgetGravKickTable.p(),
                            &gadgetLogTimeBegin, &gadgetLogTimeMax,
                            (void *) newEndBuffer.as_const<CurrentOrder>(),
                            &gadgetDPerTick, &Ti_current);
  // Probe (GADGET_HIP_PM_ACC1_PROBE=1): rms|acc1| as correct_particles is about to consume it.
  // Runs in EVERY mode too, unlike PM-ACC-PROBE which sits inside the cadence-only block. The
  // point: `every` and `cadence` diverge 5.4x in per-step velocity growth at identical timesteps,
  // which is 41x more than the measured GravPM impulse can explain. If acc1 differs by that
  // factor between the modes, then what pm_add_by_id_masked contributes is not what the kick
  // multiplies -- even though both are supposed to read the same array.
  {
    static const bool acc1Probe = (getenv("GADGET_HIP_PM_ACC1_PROBE") != NULL);
    static int acc1Count = 0;
    if (acc1Probe && acc1Count < 6)
    {
      execStream->sync();
      tree.bodies_acc1.d2h();
      double sa = 0.0; double mx = 0.0;
      for (int i = 0; i < tree.n; ++i)
      { const real4 a = tree.bodies_acc1[i];
        const double m = (double)a.x*a.x + (double)a.y*a.y + (double)a.z*a.z;
        sa += m; if (m > mx) mx = m; }
      fprintf(stderr, "[PM-ACC1-PROBE] iter=%d  rms|acc1 into correct| = %.10e  max=%.6e\n",
              iter, sqrt(sa/tree.n), sqrt(mx));
      acc1Count++;
    }
  }
  correctParticles.setWork(tree.n, 128);
  correctParticles.execute2(execStream->s());
  // Phase 3 probe (GADGET_HIP_PM_VEL_PROBE=1): the LAST unverified link. The kick is measured
  // landing in bodies_Pvel, but bodies_vel is what the drift reads, and correct_particles
  // rebuilds it from pVel. Report rms|bodies_vel| right after that rebuild; comparing this
  // between GADGET_HIP_PM_KICK_SCALE=0 and =1 shows directly whether the impulse survives.
  {
    static const bool velProbe = (getenv("GADGET_HIP_PM_VEL_PROBE") != NULL);
    static int velProbeCount = 0;
    if (velProbe && velProbeCount < 8)
    {
      execStream->sync();
      tree.bodies_vel.d2h();
      double sv = 0.0;
      for (int i = 0; i < tree.n; ++i)
      { const real4 v = tree.bodies_vel[i]; sv += (double)v.x*v.x + (double)v.y*v.y + (double)v.z*v.z; }
      fprintf(stderr, "[PM-VEL-PROBE] iter=%d  rms|bodies_vel after correct| = %.10e\n",
              iter, sqrt(sv/tree.n));
      velProbeCount++;
    }
  }

  //Copy the shuffled items back to their original buffers
  if (gadget_hip_debug_log)
  fprintf(stderr, "[BUG4-COPYBACK] tree.n=%d real4Buffer1.get_size()=%d timeBuffer.get_size()=%d "
                   "bodies_acc0.get_size()=%d bodies_time.get_size()=%d\n",
          tree.n, real4Buffer1.get_size(), timeBuffer.get_size(),
          tree.bodies_acc0.get_size(), tree.bodies_time.get_size());
  tree.bodies_acc0.copy_devonly(real4Buffer1, tree.n);
  tree.bodies_time.copy_devonly(timeBuffer, timeBuffer.get_size());
  {
    static int bug4CorrectIdx = -1;
    bug4CorrectIdx++;
    if (gadget_hip_debug_log && bug4CorrectIdx >= 68 && bug4CorrectIdx <= 74) {
      tree.bodies_time.d2h();
      tree.activePartlist.d2h();
      int sIdx[6] = {0, 1, 1000, 16384, 32000, tree.n - 1};
      for (int k = 0; k < 6; k++) {
        int idx = sIdx[k];
        int2 bt = tree.bodies_time[idx];
        uint32_t xb, yb; memcpy(&xb,&bt.x,sizeof(xb)); memcpy(&yb,&bt.y,sizeof(yb));
        fprintf(stderr, "[BUG4-BRACKET] correctIdx=%d AFTER_COPYBACK bodies_time[%d]=(0x%08x(%.9g), 0x%08x(%.9g)) active_list[%d]=%u\n",
                bug4CorrectIdx, idx, xb, bt.x, yb, bt.y, idx, tree.activePartlist[idx]);
      }
      int nActive = 0;
      for (int i = 0; i < tree.n; i++) if (tree.activePartlist[i] == 1) nActive++;
      fprintf(stderr, "[BUG4-BRACKET] correctIdx=%d nActive=%d / %d\n", bug4CorrectIdx, nActive, tree.n);
    }
  }

  // T13: genuine per-iteration active-particle count (real GPU-side read of tree.activePartlist,
  // NOT the fake host-side n_active_particles counter -- see gpu_iterate.cpp DO_BLOCK_TIMESTEP
  // #if 0 "Demo mode" block, which unconditionally sets tree.n_active_particles = tree.n every
  // iteration regardless of actual timestep-bin desync). Gated by GADGET_HIP_NACTIVE_DUMP so it
  // costs nothing when unset.
  if (getenv("GADGET_HIP_NACTIVE_DUMP")) {
    static int t13Iter = -1;
    t13Iter++;
    tree.activePartlist.d2h();
    int t13NActive = 0;
    for (int i = 0; i < tree.n; i++) if (tree.activePartlist[i] == 1) t13NActive++;
    fprintf(stderr, "[T13-NACTIVE] iter=%d nActive=%d / %d\n", t13Iter, t13NActive, tree.n);
  }
}



 //Double precision
// T19 (2026-09-02): `usePreKickState` reproduces Gadget-2's own energy-sampling timing exactly
// (Gadget2/run.c:47-59: energy_statistics() runs BEFORE advance_and_find_timesteps()'s kick, using
// the PRE-kick velocity -- valid at the OLD step's midpoint -- paired with the freshly-drifted
// position and freshly-computed potential). The default (false) preserves the port's original
// behavior (tree.bodies_pos/bodies_vel/bodies_acc0, i.e. AFTER correct()'s kick has already been
// applied -- the POST-kick velocity, valid at the NEW step's midpoint) for any other caller.
double octree::compute_energies(tree_structure &tree, bool usePreKickState)
{
  Ekin = 0.0; Epot = 0.0;

  #if 0
    double hEkin = 0.0;
    double hEpot = 0.0;

    tree.bodies_Ppos.d2h();
    tree.bodies_vel.d2h();
    tree.bodies_acc0.d2h();
    for (int i = 0; i < tree.n; i++) {
      float4 vel = tree.bodies_vel[i];
      hEkin += tree.bodies_Ppos[i].w*0.5*(vel.x*vel.x +
                                 vel.y*vel.y +
                                 vel.z*vel.z);
      hEpot += tree.bodies_Ppos[i].w*0.5*tree.bodies_acc0[i].w;
      //if(i < 128)
      if(i < 0)
      {
    	  LOGF(stderr,"%d\tAcc: %f %f %f %f\tPx: %f\tVx: %f\tkin: %f\tpot: %f\n", i,
    			  tree.bodies_acc0[i].x, tree.bodies_acc0[i].y, tree.bodies_acc0[i].z,
    			  tree.bodies_acc0[i].w, tree.bodies_Ppos[i].x, tree.bodies_vel[i].x,
    			  hEkin, hEpot);
      }
    }
    MPI_Barrier(mpiCommWorld);
    double hEtot = hEpot + hEkin;
    LOG("Energy (on host): Etot = %.10lg Ekin = %.10lg Epot = %.10lg \n", hEtot, hEkin, hEpot);
  #endif

  //float2 energy: x is kinetic energy, y is potential energy
  int blockSize = NBLOCK_REDUCE ;
  my_dev::dev_mem<double2>  energy;
  energy.cmalloc_copy(tree.generalBuffer1, blockSize, 0);

  // T30: coefficients for the self-potential removal and Gadget's comoving+periodic
  // normalisation (potential.c:246-254, scaled by All.G at :261). Computed once here rather than
  // in the kernel so the cosmology lookups stay on the host.
  //   selfCoef = G * 2.8      (forceSoftening is 2.8*SofteningTable; the removal uses the latter)
  //   cc10Coef = G * 2.8372975 * rho_mean^(1/3),  rho_mean = Omega0*3*H^2/(8*pi*G)
  // cc10Coef is zero unless the run is BOTH comoving and periodic, matching Gadget's own guard.
  double t30_selfCoef = 0.0, t30_cc10Coef = 0.0;
  if (haveGadgetParams)
  {
    // gadgetParams.G is All.G, already derived from the unit system (or from
    // GravityConstantInternal when the parameter file sets it non-zero) -- the same value
    // pm_scale_acc_masked applies to acc.w, so the two are consistent by construction.
    const double G = gadgetParams.G;
    t30_selfCoef = G * 2.8;
#ifdef PERIODIC
    if (gadgetParams.ComovingIntegrationOn && gadgetParams.PeriodicBoundariesOn)
    {
      const double H   = gadgetParams.Hubble;   // All.Hubble
      const double rho = gadgetParams.Omega0 * 3.0 * H * H / (8.0 * M_PI * G);
      t30_cc10Coef     = G * 2.8372975 * std::cbrt(rho);
    }
#endif
  }
  if (usePreKickState)
  {
    // pos/acc come from THIS step's fresh, already-sorted-into-tree-order buffers (pPos/acc1);
    // vel must go through the same oriParticleOrder ("unsorted[]") indirection correct_particles()
    // itself uses, since bodies_Pvel (unlike bodies_Ppos) is never resorted -- see
    // compute_energy_double_prekick's own comment (timestep.cu) for the full mechanism.
    int energyComovingFlag = haveGadgetComovingTables ? 1 : 0;
    // The long-range term of Gadget's energy synchronisation (global.c:79-87). Gadget calls
    // energy_statistics() BEFORE advance_and_find_timesteps() (run.c), so PM_Ti_begstep/endstep
    // still describe the interval we are INSIDE -- and so does this port: compute_energies() runs
    // at the top of the step, the long-range kick in correct(). The factor is the same for every
    // particle, so it is a scalar here and only the per-particle GravPM goes to the device.
    //
    // Not const: set_args stores a non-const void* and dereferences it at launch.
    float dtGravKickPM = 0.0f;
    const bool pmEnergyTerm = s_pmCadenceActive && s_gravPMx != nullptr &&
                              haveGadgetComovingTables && s_gadgetTablesOk &&
                              s_PM_Ti_endstep > s_PM_Ti_begstep;
    if (pmEnergyTerm)
    {
      const gadget_tick_t tiMid = (s_PM_Ti_begstep + s_PM_Ti_endstep) / 2;
      const double aBeg = gadgetTimeline.toTime(s_PM_Ti_begstep);
      const double aMid = gadgetTimeline.toTime(tiMid);
      const double aCur = gadgetTimeline.toTime(Ti_current);
      // Gadget's own two-term spelling, kept verbatim rather than collapsed to a single
      // midpoint->now integral, so the sign convention is inherited instead of re-derived.
      double d = gadget_get_gravkick_factor(s_gadgetTables, aBeg, aCur)
               - gadget_get_gravkick_factor(s_gadgetTables, aBeg, aMid);
      // Same factor G as the kick itself: GravPM is stored G-FREE here (the cadence applies G in
      // the kick because the stored force never meets pm_scale_acc_masked), while Gadget stores it
      // G-scaled -- hence Gadget's GravPM[j]/All.G in gravtree.c and no G here in global.c.
      if (!useDirectGravity && haveGadgetParams && gadgetParams.G > 0.0) d *= gadgetParams.G;
      dtGravKickPM = (float) d;
    }
    // Same contract as the kick: G is already folded in above, so the wrapper is an assertion about
    // what this number is, not a change to it.
    GCarrier dtGravKickPMG = GCarrier::gravkick_times_G(dtGravKickPM);
    computeEnergyPreKick.set_args(sizeof(double)*128*2, &tree.n,
                                   (void *) tree.bodies_Ppos.as_const<CurrentOrder>(),
                                   (void *) tree.bodies_Pvel.as_const<OriginalOrder>(),
                                   (void *) tree.bodies_acc1.as_const<CurrentOrder>(),
                                   (void *) tree.oriParticleOrder.as_const<CurrentOrder>(),
                                   (void *) tree.bodies_time.as_const<OriginalOrder>(),
                                   &t_current, &energyComovingFlag, gadgetGravKickTable.p(),
                                   &gadgetLogTimeBegin, &gadgetDPerTick, &gadgetLogTimeMax,
                                   (void *) tree.bodies_forceSoftening.as_const<CurrentOrder>(),
                                   &t30_selfCoef, &t30_cc10Coef,
                                   pmEnergyTerm ? (void *) tree.bodies_ids.as_const<CurrentOrder>()
                                                : (void *) &s_nullDevPtr,
                                   pmEnergyTerm ? (void *) &s_gravPMx : &s_nullDevPtr,
                                   pmEnergyTerm ? (void *) &s_gravPMy : &s_nullDevPtr,
                                   pmEnergyTerm ? (void *) &s_gravPMz : &s_nullDevPtr,
                                   &dtGravKickPMG,
                                   energy.p());
    // "A feature that declines to engage must say so" (T44 item 3): the cadence being on while
    // this term is off would silently restore the sawtooth this fix removes.
    // The FIRST energy sample is iter 0, where PM_Ti_begstep == PM_Ti_endstep == 0 because the
    // first long-range kick has not run yet -- so the term is legitimately off there and reporting
    // on that call alone says "OFF" about something that engages one step later. Report when it
    // first engages; complain only if it still has not by the third sample with the cadence on.
    static int  s_pmEnergyCalls = 0;
    static bool s_pmEnergyEverOn = false;
    if (s_pmCadenceActive)
    {
      s_pmEnergyCalls++;
      if (pmEnergyTerm && !s_pmEnergyEverOn)
      {
        s_pmEnergyEverOn = true;
        fprintf(stderr, "[PM-ENERGY] long-range term in the energy diagnostic: ACTIVE "
                        "(first engaged on energy sample %d)\n", s_pmEnergyCalls);
      }
      else if (!pmEnergyTerm && !s_pmEnergyEverOn && s_pmEnergyCalls == 3)
        fprintf(stderr, "[PM-ENERGY] long-range term OFF after 3 energy samples with the cadence "
                        "active -- reported Ekin WILL sawtooth with the PM interval\n");
    }
    computeEnergyPreKick.setWork(-1, 128, blockSize);
    computeEnergyPreKick.execute2(execStream->s());
  }
  else
  {
    computeEnergy.set_args(sizeof(double)*128*2, &tree.n, tree.bodies_Ppos.p(), tree.bodies_vel.p(),
                            tree.bodies_acc0.p(), tree.bodies_forceSoftening.p(),
                            &t30_selfCoef, &t30_cc10Coef, energy.p());
    computeEnergy.setWork(-1, 128, blockSize);
    computeEnergy.execute2(execStream->s());
  }

  //Reduce the last parts on the host
  energy.d2h();
  Ekin = energy[0].x;
  Epot = energy[0].y;
  for (int i = 1; i < blockSize ; i++)
  {
      Ekin += energy[i].x;
      Epot += energy[i].y;
  }

  // T24: the T14 GADGET_HIP_EPOT_TOPN diagnostic reads tree.bodies_acc0[i].w, but the PRODUCTION
  // energy call (gpu_iterate.cpp:1149, usePreKickState=true, per T19) actually reduces
  // tree.bodies_acc1[i].w via computeEnergyPreKick/compute_energy_double_prekickD (this exact
  // function, pos=bodies_Ppos, acc=bodies_acc1) -- a DIFFERENT buffer than acc0. Add a matching
  // top-N dump against the buffer that's actually summed here, plus a per-block partial-sum dump
  // of the `energy[]` reduction array itself, to distinguish "one/few real particles have an
  // extreme acc1.w" from "the block-level reduction itself is corrupted/reading garbage for some
  // block" (e.g. an uninitialized or out-of-range block for this n_bodies/blockSize/gridSize
  // combination). Gated by env var, zero-cost when unset.
  if (usePreKickState && getenv("GADGET_HIP_T24_ACC1_TOPN"))
  {
    // Per-block partial sums first -- if ONE block's own energy[i].y is already wildly extreme
    // (vs. the rest, which should all be small/comparable), that points at the reduction kernel
    // itself (garbage/uninitialized shared-mem carryover, or a block indexing/gridSize mismatch)
    // rather than at any real particle.
    double blockMin = energy[0].y, blockMax = energy[0].y;
    int blockMinIdx = 0, blockMaxIdx = 0;
    for (int i = 1; i < blockSize; i++) {
      if (energy[i].y < blockMin) { blockMin = energy[i].y; blockMinIdx = i; }
      if (energy[i].y > blockMax) { blockMax = energy[i].y; blockMaxIdx = i; }
    }
    fprintf(stderr, "[T24-ACC1-BLOCKS] iter=%d blockSize=%d n=%d Epot_total=%.10lg "
                     "blockMin=%.10lg(@%d) blockMax=%.10lg(@%d)\n",
            iter, blockSize, tree.n, Epot, blockMin, blockMinIdx, blockMax, blockMaxIdx);

    tree.bodies_Ppos.d2h();
    tree.bodies_acc1.d2h();
    const int topN = std::min(5, tree.n);
    std::vector<int> idxs(tree.n);
    for (int i = 0; i < tree.n; i++) idxs[i] = i;
    std::partial_sort(idxs.begin(), idxs.begin() + topN, idxs.end(),
                       [&](int a, int b) { return tree.bodies_acc1[a].w < tree.bodies_acc1[b].w; });
    double topSum = 0.0;
    for (int k = 0; k < topN; k++)
    {
      int i = idxs[k];
      double partE = (double)tree.bodies_Ppos[i].w * 0.5 * (double)tree.bodies_acc1[i].w;
      topSum += partE;
      fprintf(stderr, "[T24-ACC1-TOPN] iter=%d rank=%d idx=%d pos=(%.6g,%.6g,%.6g) mass=%.6g "
              "acc1=(%.6g,%.6g,%.6g) potW=%.6g partE=%.6g\n",
              iter, k, i,
              tree.bodies_Ppos[i].x, tree.bodies_Ppos[i].y, tree.bodies_Ppos[i].z, tree.bodies_Ppos[i].w,
              tree.bodies_acc1[i].x, tree.bodies_acc1[i].y, tree.bodies_acc1[i].z,
              tree.bodies_acc1[i].w, partE);
    }
    fprintf(stderr, "[T24-ACC1-TOPN] iter=%d topN_energy_sum=%.10lg total_Epot=%.10lg frac=%.6g\n",
            iter, topSum, Epot, (Epot != 0.0) ? topSum / Epot : 0.0);

    // T28 (Reconsider Ticket 24's root cause): Ticket 26 explicitly flagged that the pathological
    // node's actual level/AABB/mass/member-spread was never directly measured -- get it here, for
    // the exact particles T24-ACC1-TOPN just identified, reusing its own idxs/topN. For each of the
    // topN particles, find its own LEAF node (a leaf's [pfirst,pfirst+nchild) range in node_bodies
    // partitions the SAME tree.bodies_Ppos index space compute_leaf/propsLeafD itself uses --
    // confirmed via compute_properties.cpp's propsLeafD.set_args call, which passes
    // tree.bodies_Ppos as the leaf kernel's own body_pos array) and dump its real subdivision level
    // (decoded exactly from node_bodies[idx].x's LEVELMASK/BITLEVELS bits, node_specs.h -- the same
    // bits build_tree.cu itself writes once at creation and cellSizeInfo's own host-independent
    // compute_scaling decode already relies on, not inferred), raw AABB half-extents (boxSizeInfo,
    // NEVER modified by any runtime bJ floor -- the floor only clamps a kernel-local float variable,
    // so this is always the TRUE unfloored geometry regardless of which floor the running build
    // ships), the level-derived cell size (cellSizeInfo, Ticket 26), the REAL mass (multipole[3*i].w
    // -- NOT boxCenterInfo[i].w, which is `cellOp`, a MAC-angle-squared quantity per
    // compute_propertiesD.cu:483-512, negated for leaves as a leaf-vs-node sign flag only; T24/T26's
    // own T24-NODE-SCAN/T26-NODE-SCAN-SMALL diagnostics printed THIS field labeled "mass=", a real
    // but harmless diagnostic-only mislabeling bug, flagged here separately since it's not this
    // ticket's own scope to fix), and the TRUE member particle position spread (enumerated directly
    // from tree.bodies_Ppos over the leaf's own [pfirst,pfirst+nchild) range, independently of the
    // AABB/mass/COM the tree already reports for it -- this is the direct data-integrity
    // cross-check Method step 1 asks for: does the reported AABB/mass/COM actually correspond to
    // the SAME population a raw member enumeration finds, which would rule out an index-desync of
    // the class already found three times on this map, Tickets 12/19/23).
    if (getenv("GADGET_HIP_T28_TRACE"))
    {
      tree.node_bodies.d2h();
      tree.boxSizeInfo.d2h();
      tree.boxCenterInfo.d2h();
      tree.multipole.d2h();
      tree.cellSizeInfo.d2h();
      tree.body2group_list.d2h();
      tree.groupCenterInfo.d2h();
      tree.groupSizeInfo.d2h();
      tree.groupMaxAccInfo.d2h();
      const int nNodes = tree.n_nodes;
      // Real periodic box size: not host-accessible (only lives as a device __constant__,
      // g_periodic_boxSize, set once via pm_periodic_setup_gravity_kernel) -- same env-var
      // workaround Ticket 26's own GADGET_HIP_T26_BJ_DIST_BOXSIZE diagnostic already used for this
      // identical problem. Defaults to this map's own known N=32768/PMGRID64 Zel'dovich value.
      const char *boxEnv = getenv("GADGET_HIP_T28_BOXSIZE");
      const float t28BoxSize = boxEnv ? (float)atof(boxEnv) : 32000.0f;

      struct T28LeafInfo {
        bool found = false; int nodeIdx = -1; unsigned level = 0;
        float bJ = 0, cellSize = 0, mass = 0;
        float comx = 0, comy = 0, comz = 0;
        float cx = 0, cy = 0, cz = 0, hx = 0, hy = 0, hz = 0;
        unsigned pfirst = 0, nchild = 0;
        float minx = 1e30f, miny = 1e30f, minz = 1e30f, maxx = -1e30f, maxy = -1e30f, maxz = -1e30f;
        double sumMass = 0.0;
        int group = -1;
        float gcx = 0, gcy = 0, gcz = 0, gsx = 0, gsy = 0, gsz = 0, groupMaxAcc = 0;
      };

      std::vector<T28LeafInfo> infos(topN);

      for (int k = 0; k < topN; k++)
      {
        const int pidx = idxs[k];
        T28LeafInfo &L = infos[k];
        // Linear scan: leaves partition all n particles into contiguous, non-overlapping ranges of
        // the SAME bodies_Ppos index space (compute_leaf's own construction), so exactly one leaf's
        // range contains pidx.
        for (int n = 0; n < nNodes; n++)
        {
          if (tree.boxCenterInfo[n].w > 0.0f) continue; // internal node, not a leaf (cellOp sign convention)
          const uint2 bij = tree.node_bodies[n];
          const unsigned pfirst = bij.x & ILEVELMASK;
          if (bij.y <= pfirst) continue;
          const unsigned nchild = bij.y - pfirst;
          if ((unsigned)pidx < pfirst || (unsigned)pidx >= pfirst + nchild) continue;
          L.found = true; L.nodeIdx = n;
          L.level = (bij.x & LEVELMASK) >> BITLEVELS;
          const float4 sz  = tree.boxSizeInfo[n];
          const float4 ctr = tree.boxCenterInfo[n];
          const real4  com = tree.multipole[3*n];
          L.bJ = 2.0f*std::max(sz.x, std::max(sz.y, sz.z));
          L.cellSize = tree.cellSizeInfo[n];
          L.mass = com.w;
          L.comx = com.x; L.comy = com.y; L.comz = com.z;
          L.cx = ctr.x; L.cy = ctr.y; L.cz = ctr.z; L.hx = sz.x; L.hy = sz.y; L.hz = sz.z;
          L.pfirst = pfirst; L.nchild = nchild;
          for (unsigned m = pfirst; m < pfirst + nchild; m++)
          {
            const float4 p = tree.bodies_Ppos[m];
            L.minx = std::min(L.minx, p.x); L.maxx = std::max(L.maxx, p.x);
            L.miny = std::min(L.miny, p.y); L.maxy = std::max(L.maxy, p.y);
            L.minz = std::min(L.minz, p.z); L.maxz = std::max(L.maxz, p.z);
            L.sumMass += (double)p.w;
          }
          break;
        }
        L.group = (int)tree.body2group_list[pidx];
        const float4 gc = tree.groupCenterInfo[L.group];
        const float4 gs = tree.groupSizeInfo[L.group];
        L.gcx = gc.x; L.gcy = gc.y; L.gcz = gc.z; L.gsx = gs.x; L.gsy = gs.y; L.gsz = gs.z;
        L.groupMaxAcc = tree.groupMaxAccInfo[L.group];

        fprintf(stderr, "[T28-LEAF] iter=%d rank=%d idx=%d found=%d node=%d level=%u "
                "boxCenter=(%.6g,%.6g,%.6g) boxHalfSize=(%.6g,%.6g,%.6g) bJ_raw_AABB=%.6g "
                "cellSizeT26_levelDerived=%.6g realMass_multipoleW=%.10g nchild=%u "
                "memberSumMass=%.10lg memberSpreadX=[%.6g,%.6g] memberSpreadY=[%.6g,%.6g] "
                "memberSpreadZ=[%.6g,%.6g] COM=(%.6g,%.6g,%.6g) group=%d "
                "groupCenter=(%.6g,%.6g,%.6g) groupHalfSize=(%.6g,%.6g,%.6g) groupMaxAcc=%.6g\n",
                iter, k, pidx, (int)L.found, L.nodeIdx, L.level,
                L.cx, L.cy, L.cz, L.hx, L.hy, L.hz, L.bJ,
                L.cellSize, (double)L.mass, L.nchild,
                L.sumMass, L.minx, L.maxx, L.miny, L.maxy, L.minz, L.maxz,
                L.comx, L.comy, L.comz, L.group, L.gcx, L.gcy, L.gcz, L.gsx, L.gsy, L.gsz,
                L.groupMaxAcc);
      }

      // T28 step 1/3: cross-pair recompute of split_node_grav_springel's EXACT decision (same
      // formula, same periodic pm_nearest wrap, dev_approximate_gravity_warp_new.cu:1109-1200) for
      // every pair of the topN particles whose leaves differ -- tests directly whether particle b's
      // own LEAF, evaluated as a candidate source node against particle a's own GROUP, would be
      // split (correct) or wrongly approximated, using the raw (pre-floor) bJ recorded above, and
      // separately using the actually-shipped floored bJ (0.35*boxSize, Ticket 24) for comparison.
      auto pmNearest = [&](float x) {
        const float half = 0.5f*t28BoxSize;
        if (x > half) return x - t28BoxSize;
        if (x < -half) return x + t28BoxSize;
        return x;
      };
      for (int a = 0; a < topN; a++)
      {
        if (!infos[a].found) continue;
        const T28LeafInfo &G = infos[a]; // a's own GROUP is the walk's target
        for (int b = 0; b < topN; b++)
        {
          if (a == b || !infos[b].found || infos[b].nodeIdx == infos[a].nodeIdx) continue;
          const T28LeafInfo &N = infos[b]; // b's own LEAF is the candidate source node
          float drx = fabsf(pmNearest(G.gcx - N.comx)) - G.gsx;
          float dry = fabsf(pmNearest(G.gcy - N.comy)) - G.gsy;
          float drz = fabsf(pmNearest(G.gcz - N.comz)) - G.gsz;
          drx = std::max(drx, 0.0f); dry = std::max(dry, 0.0f); drz = std::max(drz, 0.0f);
          const float ds2 = drx*drx + dry*dry + drz*drz;
          const float mJ = fabsf(N.mass);
          const float aold = this->errTolForceAcc * G.groupMaxAcc;
          const float bJFloored = std::max(N.bJ, 0.35f*t28BoxSize);
          for (int variant = 0; variant < 2; variant++)
          {
            const float bJ = variant == 0 ? N.bJ : bJFloored;
            const float bJ2 = bJ*bJ;
            const bool primary = mJ*bJ2 > ds2*ds2*aold;
            const float proximity = 0.6f*bJ;
            const bool proxTest = (drx < proximity && dry < proximity && drz < proximity);
            const bool splitCell = primary || proxTest;
            fprintf(stderr, "[T28-PAIR] iter=%d targetIdx=%d(group=%d) sourceIdx=%d(leafNode=%d,level=%u) "
                    "variant=%s bJ=%.6g mJ=%.10g ds2=%.6g aold=%.6g primaryOpens=%d proxOpens=%d "
                    "splitCell=%d(%s)\n",
                    iter, idxs[a], G.group, idxs[b], N.nodeIdx, N.level,
                    variant == 0 ? "raw" : "floored",
                    bJ, (double)mJ, ds2, aold, (int)primary, (int)proxTest, (int)splitCell,
                    splitCell ? "SPLIT-correct" : "APPROX-wrongful-if-close");
          }
        }
      }

      // T28 step 1 (extended): the 5x5 cross-pair check above only tests the topN particles'
      // OWN leaves against each other's groups -- it does not rule out a SIXTH, unlisted node
      // elsewhere in the tree being the true wrongfully-approximated source (T24's own
      // T24-NODE-SCAN comment already flagged this possibility: "spread across MULTIPLE
      // different groups" is consistent with one bad shared source node, not necessarily one of
      // these five's own tiny leaves). Full scan: for each of the topN particles' own GROUP,
      // evaluate the EXACT split_node_grav_springel decision (both raw and floored bJ) against
      // EVERY node in the tree, and report the smallest-ds2 node that still gets APPROXIMATED
      // (splitCell=false) under the RAW bJ -- this is the genuinely-closest node this group's
      // real walk would wrongly approximate if no floor existed, independent of any assumption
      // about which particles are involved.
      for (int a = 0; a < topN; a++)
      {
        if (!infos[a].found) continue;
        const T28LeafInfo &G = infos[a];
        int bestNode = -1; float bestDs2 = 1e30f; float bestBJ = 0, bestMJ = 0;
        int bestNodeFlooredStillApprox = -1; float bestDs2Floored = 1e30f;
        long nZeroBJWrongApprox = 0;
        // T28 follow-up: same scan, but for nodes with bJ>0 (i.e. NOT covered by the bJ<=0
        // single-particle fix) that STILL get wrongly approximated -- keep the closest few, to see
        // whether the bJ<=0 fix alone is sufficient or whether a second, nonzero-bJ population of
        // wrongly-approximated close nodes also needs addressing.
        std::vector<std::tuple<float,int,float,float,unsigned>> nonzeroWrong; // ds2, node, bJ, mJ, nchild
        for (int n = 0; n < nNodes; n++)
        {
          if (n == infos[a].nodeIdx) continue;
          const float4 sz = tree.boxSizeInfo[n];
          const real4  com = tree.multipole[3*n];
          const float mJ = fabsf(com.w);
          if (mJ <= 0.0f) continue; // skip massless/empty
          const float bJraw = 2.0f*std::max(sz.x, std::max(sz.y, sz.z));
          float drx = fabsf(pmNearest(G.gcx - com.x)) - G.gsx;
          float dry = fabsf(pmNearest(G.gcy - com.y)) - G.gsy;
          float drz = fabsf(pmNearest(G.gcz - com.z)) - G.gsz;
          drx = std::max(drx, 0.0f); dry = std::max(dry, 0.0f); drz = std::max(drz, 0.0f);
          const float ds2 = drx*drx + dry*dry + drz*drz;
          const float aold = this->errTolForceAcc * G.groupMaxAcc;
          // raw
          {
            const float bJ2 = bJraw*bJraw;
            const bool primary = mJ*bJ2 > ds2*ds2*aold;
            const float proximity = 0.6f*bJraw;
            const bool proxTest = (drx < proximity && dry < proximity && drz < proximity);
            const bool splitCell = primary || proxTest;
            if (!splitCell && bJraw <= 0.0f) nZeroBJWrongApprox++;
            if (!splitCell && ds2 < bestDs2) { bestDs2 = ds2; bestNode = n; bestBJ = bJraw; bestMJ = mJ; }
            if (!splitCell && bJraw > 0.0f) {
              const uint2 bijn = tree.node_bodies[n];
              const unsigned pf = bijn.x & ILEVELMASK;
              const unsigned nc = (bijn.y > pf) ? (bijn.y - pf) : 0;
              nonzeroWrong.push_back(std::make_tuple(ds2, n, bJraw, mJ, nc));
            }
          }
          // floored (0.35*boxSize, Ticket 24's actually-shipped formula)
          {
            const float bJf = std::max(bJraw, 0.35f*t28BoxSize);
            const float bJ2 = bJf*bJf;
            const bool primary = mJ*bJ2 > ds2*ds2*aold;
            const float proximity = 0.6f*bJf;
            const bool proxTest = (drx < proximity && dry < proximity && drz < proximity);
            const bool splitCell = primary || proxTest;
            if (!splitCell && ds2 < bestDs2Floored) { bestDs2Floored = ds2; bestNodeFlooredStillApprox = n; }
          }
        }
        unsigned bestLevel = 0, bestNchild = 0, bestPfirst = 0;
        float bestPx = 0, bestPy = 0, bestPz = 0;
        if (bestNode >= 0) {
          const uint2 bij = tree.node_bodies[bestNode];
          bestPfirst = bij.x & ILEVELMASK;
          bestLevel = (bij.x & LEVELMASK) >> BITLEVELS;
          bestNchild = (bij.y > bestPfirst) ? (bij.y - bestPfirst) : 0;
          const bool bestIsLeaf = tree.boxCenterInfo[bestNode].w <= 0.0f;
          if (bestIsLeaf && bestNchild >= 1) {
            const float4 pp = tree.bodies_Ppos[bestPfirst];
            bestPx = pp.x; bestPy = pp.y; bestPz = pp.z;
          }
        }
        fprintf(stderr, "[T28-SCAN] iter=%d targetIdx=%d group=%d nNodesScanned=%d "
                "nZeroBJWrongApprox=%ld "
                "closestWrongApprox_RAW: node=%d level=%u nchild=%u pos0=(%.6g,%.6g,%.6g) "
                "ds2=%.6g dist=%.6g bJ=%.6g mJ=%.10g "
                "closestWrongApprox_FLOORED: node=%d ds2=%.6g dist=%.6g\n",
                iter, idxs[a], G.group, nNodes, nZeroBJWrongApprox,
                bestNode, bestLevel, bestNchild, bestPx, bestPy, bestPz,
                bestDs2, sqrt((double)std::max(bestDs2,0.0f)), bestBJ, (double)bestMJ,
                bestNodeFlooredStillApprox, bestDs2Floored,
                sqrt((double)std::max(bestDs2Floored,0.0f)));

        std::sort(nonzeroWrong.begin(), nonzeroWrong.end(),
                  [](const auto &x, const auto &y) { return std::get<0>(x) < std::get<0>(y); });
        const int nShow = std::min((size_t)6, nonzeroWrong.size());
        for (int s = 0; s < nShow; s++) {
          const auto &e = nonzeroWrong[s];
          fprintf(stderr, "[T28-SCAN-NONZERO] iter=%d targetIdx=%d group=%d rank=%d node=%d "
                  "ds2=%.6g dist=%.6g bJ=%.6g mJ=%.10g nchild=%u\n",
                  iter, idxs[a], G.group, s, std::get<1>(e), std::get<0>(e),
                  sqrt((double)std::max(std::get<0>(e),0.0f)), std::get<2>(e), (double)std::get<3>(e),
                  std::get<4>(e));
        }
      }
    }
  }

  //Sum the values / energies of the system using MPI
  AllSum(Epot); AllSum(Ekin);

  Etot = Epot + Ekin;

  // T14 (2026-09-02, ticket T14 Session 5): root-causing the confirmed energy anomaly (T14
  // Resolution Session 4 -- de grows monotonically to ~24-27x the CPU reference's worst-case
  // drift, never relaxing, while the 6 already-traced particle IDs show nothing extreme). This
  // dumps the top-N most-negative per-particle acc0.w values -- the EXACT SAME field
  // compute_energy_double (above) sums into Epot, no separate formula -- by original particle id,
  // every call, so we can tell directly whether the Epot blowup is concentrated in a handful of
  // particles (a real or spurious close-approach force/potential issue for specific bodies) or
  // spread across the whole population (more consistent with a systematic integration/softening
  // issue than a localized one). Gated by env var, zero-cost when unset.
  if (getenv("GADGET_HIP_EPOT_TOPN"))
  {
    tree.bodies_Ppos.d2h();
    tree.bodies_acc0.d2h();
    tree.bodies_ids.d2h();
    const int topN = std::min(5, tree.n);
    std::vector<int> idxs(tree.n);
    for (int i = 0; i < tree.n; i++) idxs[i] = i;
    std::partial_sort(idxs.begin(), idxs.begin() + topN, idxs.end(),
                       [&](int a, int b) { return tree.bodies_acc0[a].w < tree.bodies_acc0[b].w; });
    double topSum = 0.0;
    for (int k = 0; k < topN; k++)
    {
      int i = idxs[k];
      double partE = (double)tree.bodies_Ppos[i].w * 0.5 * (double)tree.bodies_acc0[i].w;
      topSum += partE;
      fprintf(stderr, "[T14-EPOT-TOPN] iter=%d rank=%d id=%llu pos=(%.6g,%.6g,%.6g) mass=%.6g "
              "acc0=(%.6g,%.6g,%.6g) potW=%.6g partE=%.6g\n",
              iter, k, (unsigned long long)tree.bodies_ids[i],
              tree.bodies_Ppos[i].x, tree.bodies_Ppos[i].y, tree.bodies_Ppos[i].z, tree.bodies_Ppos[i].w,
              tree.bodies_acc0[i].x, tree.bodies_acc0[i].y, tree.bodies_acc0[i].z,
              tree.bodies_acc0[i].w, partE);
    }
    fprintf(stderr, "[T14-EPOT-TOPN] iter=%d topN_energy_sum=%.10lg total_Epot=%.10lg frac=%.6g\n",
            iter, topSum, Epot, (Epot != 0.0) ? topSum / Epot : 0.0);
  }

  // T14 (2026-09-02, ticket T14 Session 5): test hypothesis that the confirmed energy anomaly is
  // (at least partly) a DIAGNOSTIC-ONLY bug -- this port's compute_energy_double sums raw
  // pos.w*0.5*vel^2 and pos.w*0.5*acc.w with NO comoving scale-factor normalization, whereas
  // Gadget-2 CPU's own compute_global_quantities_of_system() (global.c:56,88-89, confirmed read
  // directly from /home/sergey/work/gpu-code/gadget/Gadget-2.0.7/Gadget2/global.c) divides
  // EnergyPot by a1=All.Time and EnergyKin by a2=All.Time^2 before ever comparing energies across
  // time -- exactly matching this run's own ComovingIntegrationOn=1 (a1/a2 both == 1 only in the
  // non-comoving case, never here). This is a genuine, confirmed difference in FORMULA, not
  // inference. NOTE: Gadget-2 also kick-extrapolates each particle's velocity to the current
  // global time using its own GravAccel before squaring (global.c:69-80) to handle asynchronous
  // per-particle timesteps -- NOT reproduced here, so this corrected `de2` is expected to still
  // differ somewhat from a true Gadget-2-equivalent value, especially deep in a close encounter
  // where nActive << n (this port's own block-timestep hierarchy is real, per T13's nActive
  // finding elsewhere in this file) -- but isolating just the a/a^2 normalization piece first is
  // the right incremental test. Gated by env var, zero extra device work (host-side arithmetic
  // only on already-computed Ekin/Epot).
  if (getenv("GADGET_HIP_COMOVING_DE_CHECK"))
  {
    static double a0 = -1.0;
    static double Etot0_comoving = 0.0;
    const bool comoving = haveGadgetParams && gadgetParams.ComovingIntegrationOn != 0;
    const double a = comoving ? (double)this->t_current : 1.0;
    if (store_energy_flag || a0 < 0.0)
    {
      a0 = a;
      Etot0_comoving = Epot / a0 + Ekin / (a0 * a0);
    }
    const double Etot_comoving = Epot / a + Ekin / (a * a);
    const double de_comoving = (Etot0_comoving != 0.0) ? (Etot_comoving - Etot0_comoving) / Etot0_comoving : 0.0;
    fprintf(stderr, "[T14-COMOVING-DE] iter=%d time=%lg a=%lg a0=%lg Epot_c=%.10lg Ekin_c=%.10lg "
            "Etot_c=%.10lg Etot0_c=%.10lg de_comoving=%lg de_raw=%lg\n",
            iter, this->t_current, a, a0, Epot / a, Ekin / (a * a), Etot_comoving, Etot0_comoving,
            de_comoving, (Etot - Etot0) / Etot0);
  }

  // T14 (2026-09-02, ticket T14 Session 9): test the SECOND confirmed-but-unimplemented gap noted
  // by Session 5's comment above -- Gadget-2's kick-time velocity extrapolation
  // (global.c:60-84: kicks each particle's velocity, using its OWN GravAccel, from its last-kick
  // time up to the current global sync time before it's ever squared into EnergyKin, precisely
  // because under an individual/block-timestep scheme most particles are NOT freshly kicked at
  // every global step -- their stored `vel` is stale by up to a full bin-width). This port's
  // compute_energy_double sums the raw, un-extrapolated `bodies_vel` with no such correction.
  // Session 9 found (by reading the FULL [T14-COMOVING-DE] series, not just sparse samples) that
  // `de_comoving` jumps in discrete steps exactly at each particle-bin's own sync boundary (visible
  // from iter=1 onward, not something new at t~=0.1295) and that the jump SIZE grows over time in
  // lockstep with the encounter intensifying -- exactly the signature expected if the missing
  // kick-extrapolation (whose error scales with acc * time-since-last-kick, both of which grow near
  // pericenter) is the dominant driver of the whole anomaly, not a distinct bug localized to one
  // iteration. This block tests that directly: recompute Ekin using each particle's OWN
  // acceleration to linearly extrapolate its velocity from its last-kick time (bodies_time.x) to
  // the current global sync time, then apply the same /a, /a^2 comoving normalization as the
  // GADGET_HIP_COMOVING_DE_CHECK block above, and compare. NOTE: this uses a linear
  // v + acc*(t_current - tb) extrapolation, not Gadget-2's exact hubble-drift-factor-table kick
  // integral (get_gravkick_factor, global.c:69-80) -- a deliberate first-order approximation to
  // test the HYPOTHESIS cheaply (same spirit as the existing /a,/a^2 piece, which was also not
  // exact and still worked well) -- a residual gap vs true Gadget-2 fidelity is expected even if
  // this confirms the hypothesis. Independent env var so it can be run alone or alongside
  // GADGET_HIP_COMOVING_DE_CHECK. Not yet run-verified as of this commit (see T14 ticket Session 9
  // checkpoint) -- ready for a follow-up session to test, ideally via a fast restart from a
  // snapshot near the transition (t~=0.118, well before the t~=0.1295 zero-crossing) rather than a
  // full re-run from TimeBegin.
  if (getenv("GADGET_HIP_KICK_EXTRAP_CHECK"))
  {
    static double a0k = -1.0;
    static double Etot0_extrap = 0.0;
    const bool comoving = haveGadgetParams && gadgetParams.ComovingIntegrationOn != 0;
    const double a = comoving ? (double)this->t_current : 1.0;

    tree.bodies_Ppos.d2h();
    tree.bodies_vel.d2h();
    tree.bodies_acc0.d2h();
    tree.bodies_time.d2h();

    double ekinExtrap = 0.0;
    for (int i = 0; i < tree.n; i++)
    {
      const real4 p  = tree.bodies_Ppos[i];
      const real4 v  = tree.bodies_vel[i];
      const real4 ac = tree.bodies_acc0[i];
      const int2 bt = tree.bodies_time[i];
      // .x is Ti_begstep (ticks); convert before differencing against a scale factor.
      const double dtSinceKick = (double)this->t_current - gadgetTimeline.toTime(bt.x);
      const double vx = (double)v.x + (double)ac.x * dtSinceKick;
      const double vy = (double)v.y + (double)ac.y * dtSinceKick;
      const double vz = (double)v.z + (double)ac.z * dtSinceKick;
      ekinExtrap += (double)p.w * 0.5 * (vx * vx + vy * vy + vz * vz);
    }

    if (store_energy_flag || a0k < 0.0)
    {
      a0k = a;
      Etot0_extrap = Epot / a0k + ekinExtrap / (a0k * a0k);
    }
    const double EkinExtrap_c = ekinExtrap / (a * a);
    const double Etot_extrap_c = Epot / a + EkinExtrap_c;
    const double de_comoving_extrap = (Etot0_extrap != 0.0) ? (Etot_extrap_c - Etot0_extrap) / Etot0_extrap : 0.0;
    fprintf(stderr, "[T14-KICK-EXTRAP-DE] iter=%d time=%lg a=%lg a0=%lg Ekin_raw=%.10lg "
            "Ekin_extrap=%.10lg EkinExtrap_c=%.10lg Etot_extrap_c=%.10lg Etot0_extrap=%.10lg "
            "de_comoving_extrap=%lg\n",
            iter, this->t_current, a, a0k, Ekin, ekinExtrap, EkinExtrap_c, Etot_extrap_c,
            Etot0_extrap, de_comoving_extrap);
  }

  // T14 (2026-09-02, ticket T14 Session 13): test the THIRD confirmed-but-unimplemented gap,
  // flagged by Session 11's addendum and left untested by Session 12: compute_energy_double
  // (timestep.cu:556-562) sums tree.bodies_acc0[i].w (the potential field) for EVERY particle
  // every call, but that field is only ever WRITTEN for particles in the current iteration's
  // active_groups (confirmed by direct read, dev_approximate_gravity_warp_new.cu:1445-1456/
  // 1482-1492 -- acc_out[addr] is written only for body_i[] addresses drawn from
  // active_groups[bid], nothing else touches it) -- so an inactive particle's potential
  // contribution to Epot is stale from whenever it was LAST active, while it's being divided by
  // the CURRENT global comoving scale factor `a` just like every other (fresh) particle's
  // contribution. Session 13 confirmed via direct read of the bundled Gadget-2 CPU reference
  // (global.c:56, potential.c, run.c:55/220/415) that the CPU reference does NOT share this
  // problem: P[i].Potential is refreshed by a dedicated, full-population, all-particles
  // compute_potential() pass, called only when an energy-statistics sample is actually due
  // (TimeBetStatistics-spaced, ~20 times across the whole run) -- so the CPU reference's own
  // `EnergyPot` is always evaluated fresh, at the exact moment it's used, for every particle,
  // never reused across many intervening steps the way this port's bodies_acc0.w is. This is a
  // genuine, source-confirmed departure from Gadget-2's own approach, not an inherent limitation
  // Gadget-2 shares.
  //
  // A true fix would require a full extra tree-walk (recomputing acc0.w for ALL particles, not
  // just the active_groups subset) every time Epot is needed -- exactly what Gadget-2 CPU does,
  // but expensive to add safely as a one-off diagnostic here. Instead this block tests the
  // cheapest well-posed FIRST-ORDER correction obtainable from data already on hand, symmetric in
  // spirit to the existing kick-extrapolation piece above: extrapolate each particle's own
  // potential forward from its last-active time (bodies_time.x) to the current global sync time
  // using dPhi/dt = grad(Phi).v = -a.v (since acceleration a = -grad(Phi) is the standard
  // convention used elsewhere in this file, e.g. the kick-extrapolation block just above), i.e.
  // pot_extrap = acc0.w - (acc0.xyz . vel.xyz) * dtSinceKick. This captures ONLY the particle's
  // own motion through an otherwise-frozen potential field -- it does NOT capture the field
  // itself changing shape as OTHER particles move (which needs a real re-walk to capture) -- so
  // this is a deliberately partial test: if it closes a meaningful fraction of the residual
  // anomaly, that's evidence the particle's-own-motion term matters; if it does little or nothing,
  // that argues the missing piece is specifically the field-reshaping term, i.e. genuinely
  // requires a re-walk-based fix, not a local extrapolation. Logs three variants for direct
  // comparison against the existing [T14-COMOVING-DE] (raw) and [T14-KICK-EXTRAP-DE] (velocity-only
  // corrected) series: potential-extrapolation alone (kinetic still raw), and potential+kick
  // extrapolation combined (both corrected). Gated by its own env var so it can run alongside or
  // independently of the other two checks.
  if (getenv("GADGET_HIP_EPOT_EXTRAP_CHECK"))
  {
    static double a0p = -1.0;
    static double Etot0_potextrap = 0.0;
    static double Etot0_both = 0.0;
    const bool comoving = haveGadgetParams && gadgetParams.ComovingIntegrationOn != 0;
    const double a = comoving ? (double)this->t_current : 1.0;

    tree.bodies_Ppos.d2h();
    tree.bodies_vel.d2h();
    tree.bodies_acc0.d2h();
    tree.bodies_time.d2h();

    double epotExtrap = 0.0;
    double ekinExtrapBoth = 0.0;
    for (int i = 0; i < tree.n; i++)
    {
      const real4 p  = tree.bodies_Ppos[i];
      const real4 v  = tree.bodies_vel[i];
      const real4 ac = tree.bodies_acc0[i];
      const int2 bt = tree.bodies_time[i];
      // .x is Ti_begstep (ticks); convert before differencing against a scale factor.
      const double dtSinceKick = (double)this->t_current - gadgetTimeline.toTime(bt.x);

      const double potExtrapW = (double)ac.w
          - ((double)ac.x * (double)v.x + (double)ac.y * (double)v.y + (double)ac.z * (double)v.z)
            * dtSinceKick;
      epotExtrap += (double)p.w * 0.5 * potExtrapW;

      const double vx = (double)v.x + (double)ac.x * dtSinceKick;
      const double vy = (double)v.y + (double)ac.y * dtSinceKick;
      const double vz = (double)v.z + (double)ac.z * dtSinceKick;
      ekinExtrapBoth += (double)p.w * 0.5 * (vx * vx + vy * vy + vz * vz);
    }

    if (store_energy_flag || a0p < 0.0)
    {
      a0p = a;
      Etot0_potextrap = epotExtrap / a0p + Ekin / (a0p * a0p);
      Etot0_both = epotExtrap / a0p + ekinExtrapBoth / (a0p * a0p);
    }
    const double EpotExtrap_c = epotExtrap / a;
    const double Etot_potextrap_c = EpotExtrap_c + Ekin / (a * a);
    const double de_comoving_potextrap =
        (Etot0_potextrap != 0.0) ? (Etot_potextrap_c - Etot0_potextrap) / Etot0_potextrap : 0.0;

    const double Etot_both_c = EpotExtrap_c + ekinExtrapBoth / (a * a);
    const double de_comoving_both =
        (Etot0_both != 0.0) ? (Etot_both_c - Etot0_both) / Etot0_both : 0.0;

    fprintf(stderr, "[T14-EPOT-EXTRAP-DE] iter=%d time=%lg a=%lg a0=%lg Epot_raw=%.10lg "
            "Epot_extrap=%.10lg EpotExtrap_c=%.10lg Etot_potextrap_c=%.10lg Etot0_potextrap=%.10lg "
            "de_comoving_potextrap=%lg de_comoving_both=%lg\n",
            iter, this->t_current, a, a0p, Epot, epotExtrap, EpotExtrap_c, Etot_potextrap_c,
            Etot0_potextrap, de_comoving_potextrap, de_comoving_both);
  }

  // T16 (2026-09-02, ticket T16): promote the comoving (/a, /a^2) energy normalization from
  // opt-in diagnostic (GADGET_HIP_COMOVING_DE_CHECK above) to the PRODUCTION Etot/de/d(de)
  // fields logged every iteration by default -- this is the exact, already-tested formula from
  // that diagnostic (T14 ticket, Sessions 5/7/9: confirmed, via live comparison against the
  // Gadget-2 CPU reference, that the /a,/a^2-normalized energy tracks the reference's own
  // behavior closely), not a new one. Matches Gadget-2 CPU's own convention exactly
  // (compute_global_quantities_of_system, Gadget2/global.c:56,88-89): for
  // ComovingIntegrationOn=1, divide EnergyPot by the scale factor a=All.Time and EnergyKin by
  // a^2 before ever comparing energies across time. For a non-comoving run (or if the port
  // hasn't parsed gadget params at all) a_prod==1.0, so this reduces identically to the raw
  // values -- non-comoving behavior is unaffected by design, matching Gadget-2's own conditional.
  //
  // Deliberately computed into fresh local values (Epot_c/Ekin_c/Etot_c) and separate persisted
  // state (Etot0_prod/Etot1_prod, file-scope statics declared near de_max/dde_max above) rather
  // than mutating the existing Epot/Ekin/Etot/Etot0/Etot1 members in place: the three diagnostics
  // above (GADGET_HIP_COMOVING_DE_CHECK, GADGET_HIP_KICK_EXTRAP_CHECK,
  // GADGET_HIP_EPOT_EXTRAP_CHECK) read those raw members directly earlier in this same function
  // call -- including GADGET_HIP_COMOVING_DE_CHECK's own "de_raw" field, which reads the raw
  // Etot0 member directly -- and must keep computing against genuinely raw values, not this
  // call's own already-normalized ones. Those three diagnostics are untouched by this change.
  const bool comoving_prod = haveGadgetParams && gadgetParams.ComovingIntegrationOn != 0;
  const double a_prod = comoving_prod ? (double)this->t_current : 1.0;
  const double Epot_c = Epot / a_prod;
  const double Ekin_c = Ekin / (a_prod * a_prod);
  const double Etot_c = Epot_c + Ekin_c;

  if (store_energy_flag) {
    Ekin0 = Ekin;
    Epot0 = Epot;
    Etot0 = Etot;
    Ekin1 = Ekin;
    Epot1 = Epot;
    Etot1 = Etot;
    Etot0_prod = Etot_c;
    Etot1_prod = Etot_c;
    tinit = get_time();
    store_energy_flag = false;
  }


  double de  = (Etot_c - Etot0_prod)/Etot0_prod;
  double dde = (Etot_c - Etot1_prod)/Etot1_prod;

  // T26: this gate USED to test `tree.n_active_particles == tree.n`, which is a no-op — the
  // active-particle reduction that would set it is disabled (`approximate_gravity`, `#if 0
  // //Demo mode`, gpu_iterate.cpp:3013, inherited from Bonsai), so `n_active_particles` is
  // assigned `tree.n` unconditionally at :3036 and the condition ALWAYS passed. Measured at
  // 512^3: n_active=134217728 on every iteration including ones where only 3 of 4194802 groups
  // were active. de_max/dde_max have therefore always included stale sparse-step readings.
  // The live signal is the group count, which the walk does maintain.
  const bool activeSetFull = (tree.n_active_groups == tree.n_groups);
  if(activeSetFull)
  {
    de_max  = std::max( de_max, std::abs( de));
    dde_max = std::max(dde_max, std::abs(dde));
  }

  Ekin1 = Ekin;
  Epot1 = Epot;
  Etot1 = Etot;
  Etot1_prod = Etot_c;

  // T26 (option b): compute_energies sums acc[].w over ALL particles, but the tree walk refreshes
  // bodies_acc1 for ACTIVE particles only. On a step where the active set collapses, the reported
  // Epot is therefore the PREVIOUS potential merely rescaled by 1/a -- measured at 512^3 as an
  // exact 1/a prefactor (dEpot/Epot = 8.954e-3 on every sparse step, matching 1-a_prev/a_now to
  // ~4e-6, against a true ~6.5e-3 from the neighbouring full steps). The error is ~2.4e-3 relative,
  // transient and self-correcting on the next full step, but in an alternating sparse/full phase
  // half the printed values carry it. Dynamics are NOT affected: inactive particles are never
  // kicked (timestep.cu:52) and compute_dt is active-only (timestep.cu:25) -- this is a diagnostic
  // artefact only. Mark the reading rather than printing it silently, using the same
  // `activeSetFull` signal that now (correctly) gates de_max/dde_max above.
  // Gadget has no equivalent problem because it never reuses
  // the per-step force for energy: energy_statistics() is preceded by a dedicated all-particle
  // compute_potential() (potential.c:22, run.c:55). See tickets/T26 for the measurement and for
  // option (c), the faithful fix.
  char epotStaleTag[96] = "";
  if(!activeSetFull)
    snprintf(epotStaleTag, sizeof(epotStaleTag), "  EPOT_STALE(active_groups=%d/%d)",
             tree.n_active_groups, tree.n_groups);

  if(mpiGetRank() == 0)
  {
#if 0
  LOG("iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec\n",
		  iter, this->t_current, Etot_c, Ekin_c, Epot_c, de, de_max, dde, dde_max, get_time() - tinit);
  LOGF(stderr, "iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec\n",
		  iter, this->t_current, Etot_c, Ekin_c, Epot_c, de, de_max, dde, dde_max, get_time() - tinit);
#else
  printf("iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec%s\n",
		  iter, this->t_current, Etot_c, Ekin_c, Epot_c, de, de_max, dde, dde_max, get_time() - tinit, epotStaleTag);
  fprintf(stderr, "iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec%s\n",
		  iter, this->t_current, Etot_c, Ekin_c, Epot_c, de, de_max, dde, dde_max, get_time() - tinit, epotStaleTag);
#endif

  // Close-encounter audit diagnostic (user request 2026-08-30): dump the full position set
  // (overwriting each iteration, so only the last successful one before a crash survives) so
  // the actual minimum pairwise separation feeding the crashing tree build can be checked
  // offline for physical sanity (kpc-scale, matching softening) rather than a numerical-
  // underflow artifact of this port's own force/softening chain.
  tree.bodies_Ppos.d2h();
  FILE *fpos = fopen("/tmp/gadgethip_lastpos.bin", "wb");
  if (fpos)
  {
    fwrite(&tree.n, sizeof(int), 1, fpos);
    for (int i = 0; i < tree.n; i++)
    {
      real4 p = tree.bodies_Ppos[i];
      fwrite(&p, sizeof(real4), 1, fpos);
    }
    fclose(fpos);
  }
  // Same diagnostic, extended: bodies_ids (unconditionally reordered every sort_bodies() call,
  // per octree.h's own comment) and oriParticleOrder (the permutation array sort_bodies() computes
  // and every dataReorder() call gathers through) -- to check whether groups of particles found
  // sharing an exact-duplicate position also share the same underlying ID (real duplicate source
  // data) or the same oriParticleOrder entry (a broken/non-bijective permutation), rather than a
  // coincidence of independently-evolved trajectories.
  tree.bodies_ids.d2h();
  tree.oriParticleOrder.d2h();
  FILE *fids = fopen("/tmp/gadgethip_lastids.bin", "wb");
  if (fids)
  {
    fwrite(&tree.n, sizeof(int), 1, fids);
    for (int i = 0; i < tree.n; i++)
    {
      unsigned long long id = tree.bodies_ids[i];
      fwrite(&id, sizeof(unsigned long long), 1, fids);
    }
    fclose(fids);
  }
  FILE *ford = fopen("/tmp/gadgethip_lastorder.bin", "wb");
  if (ford)
  {
    fwrite(&tree.n, sizeof(int), 1, ford);
    for (int i = 0; i < tree.n; i++)
    {
      unsigned int o = tree.oriParticleOrder[i];
      fwrite(&o, sizeof(unsigned int), 1, ford);
    }
    fclose(ford);
  }
  }

  return de;
}


// ==============================================================================================
// Restart: stop/resume with the full integrator state (see octree.h for the rationale and the
// index-space argument). Format is this port's own -- 64-byte header, then arrays in ORIGINAL
// (load) order, so a resume is indistinguishable from a fresh IC load.
// ==============================================================================================
#define GADGET_HIP_RESTART_MAGIC   0x47484952u   /* "GHIR" */
// v2: bodies_time holds (Ti_begstep, Ti_endstep) as int2 ticks rather than float2 scale factors.
// The record is the same WIDTH, so a v1 file would deserialise without error and be interpreted as
// ticks -- every particle's schedule silently wrong. Refusing on version is the only guard.
#define GADGET_HIP_RESTART_VERSION 2

struct GadgetHipRestartHeader
{
  unsigned int magic;
  unsigned int version;
  int          n;
  int          iter;
  double       t_current;
  double       t_previous;
  double       nextGadgetSnapTime;
  double       nextEnergyStat;
  double       gadgetDtDisplacement;
  int          gadgetSnapshotCount;
  int          comovingIntegrationOn;   // sanity: refuse a resume into a different integration mode
  double       timeBegin;               // sanity: refuse a resume against a different run setup
  double       timeMax;
  // Energy-conservation REPORTING baselines. Not integrator state -- the run evolves identically
  // without them -- but `de` is measured against the energy at the FIRST call to compute_energies,
  // so a resume that re-baselined would print de=0 at the resume point and every subsequent de
  // would be relative to a different origin. That silently makes a resumed run's conservation
  // record incomparable to an uninterrupted one, which is exactly the comparison this facility
  // has to support. Etot0_prod/Etot1_prod are the a-normalised production baselines
  // (gpu_iterate.cpp:43-44); the raw Ekin0..Etot1 class members feed a debug diagnostic only.
  double       etot0_prod, etot1_prod, deMax, ddeMax;
  double       ekin0, epot0, etot0, ekin1, epot1, etot1;
  int          storeEnergyFlag;
};

bool octree::writeRestartFile(const char *path)
{
  tree_structure &tree = this->localTree;
  const int n = tree.n;

  tree.bodies_Ppos.d2h();  tree.bodies_ids.d2h();   tree.oriParticleOrder.d2h();
  tree.bodies_vel.d2h();   tree.bodies_Pvel.d2h();
  tree.bodies_time.d2h();  tree.bodies_acc0.d2h();

  // Un-permute the two arrays the per-iteration reorder touches, so the whole file is in one
  // (original) order. oriParticleOrder[i] is the ORIGINAL index of the particle now at i.
  std::vector<real4>      pos(n);
  std::vector<ullong>     ids(n);
  for (int i = 0; i < n; i++)
  {
    const uint o = tree.oriParticleOrder[i];
    if ((int) o >= n) { fprintf(stderr, "FATAL: restart write: oriParticleOrder[%d]=%u out of range\n", i, o); return false; }
    pos[o] = tree.bodies_Ppos[i];
    ids[o] = tree.bodies_ids[i];
  }

  const std::string tmp = std::string(path) + ".tmp";
  FILE *f = fopen(tmp.c_str(), "wb");
  if (!f) { fprintf(stderr, "FATAL: restart write: cannot open %s\n", tmp.c_str()); return false; }

  GadgetHipRestartHeader h;
  memset(&h, 0, sizeof(h));
  h.magic = GADGET_HIP_RESTART_MAGIC; h.version = GADGET_HIP_RESTART_VERSION;
  // iter+1, not iter: this write happens at the END of a completed iteration but BEFORE the
  // `iter++` below, so the iteration the resumed run must do NEXT is iter+1. Storing `iter` would
  // make a resume silently REDO the last completed step -- drifting, re-evaluating gravity and
  // re-kicking from an already-kicked state.
  h.n = n; h.iter = this->iter + 1;
  h.t_current = this->t_current; h.t_previous = this->t_previous;
  h.nextGadgetSnapTime = this->nextGadgetSnapTime;
  h.nextEnergyStat = this->restartNextEnergyStat;
  h.gadgetDtDisplacement = this->gadgetDtDisplacement;
  h.gadgetSnapshotCount = this->gadgetSnapshotCount;
  h.comovingIntegrationOn = haveGadgetParams ? gadgetParams.ComovingIntegrationOn : 0;
  h.timeBegin = haveGadgetParams ? gadgetParams.TimeBegin : 0.0;
  h.timeMax   = haveGadgetParams ? gadgetParams.TimeMax   : 0.0;
  h.etot0_prod = Etot0_prod; h.etot1_prod = Etot1_prod;
  h.deMax = de_max;          h.ddeMax = dde_max;
  h.ekin0 = this->Ekin0; h.epot0 = this->Epot0; h.etot0 = this->Etot0;
  h.ekin1 = this->Ekin1; h.epot1 = this->Epot1; h.etot1 = this->Etot1;
  h.storeEnergyFlag = this->store_energy_flag ? 1 : 0;

  // T51: the recentring shift, as a sidecar rather than a header field -- see the comment on
  // gadgetZoomShift. Absent sidecar == no shift, which is the right reading of any restart written
  // before this existed. Guarded: the octree members only exist in a GADGET_HIP_HIGHRES build.
#ifdef GADGET_HIP_HIGHRES
  if (gadgetZoomShifted)
  {
    const std::string zp = std::string(path) + ".zoomshift";
    FILE *zf = fopen(zp.c_str(), "w");
    if (zf)
    {
      fprintf(zf, "%.17g %.17g %.17g\n", gadgetZoomShift[0], gadgetZoomShift[1], gadgetZoomShift[2]);
      fclose(zf);
    }
    else
      fprintf(stderr, "[ZOOM] WARNING: could not write %s -- a resume from this restart will not "
                      "know the recentring shift, and its snapshots would come out offset.\n",
              zp.c_str());
  }
#endif

  bool ok = fwrite(&h, sizeof(h), 1, f) == 1;
  ok = ok && fwrite(pos.data(),               sizeof(real4),  n, f) == (size_t) n;
  ok = ok && fwrite(ids.data(),               sizeof(ullong), n, f) == (size_t) n;
  ok = ok && fwrite(&tree.bodies_vel[0],      sizeof(real4),  n, f) == (size_t) n;
  ok = ok && fwrite(&tree.bodies_Pvel[0],     sizeof(real4),  n, f) == (size_t) n;
  ok = ok && fwrite(&tree.bodies_time[0],     sizeof(int2), n, f) == (size_t) n;
  ok = ok && fwrite(&tree.bodies_acc0[0],     sizeof(real4),  n, f) == (size_t) n;
  if (fclose(f) != 0) ok = false;

  if (!ok) { fprintf(stderr, "FATAL: restart write to %s failed (disk full?)\n", tmp.c_str()); remove(tmp.c_str()); return false; }
  // Rename only after a complete, successful write: a crash mid-dump must not leave a truncated
  // file that a later resume would silently accept.
  if (rename(tmp.c_str(), path) != 0) { fprintf(stderr, "FATAL: restart: cannot rename %s -> %s\n", tmp.c_str(), path); return false; }
  fprintf(stderr, "[RESTART] wrote %s  (iter=%d t=%.9g n=%d)\n", path, this->iter, this->t_current, n);
  return true;
}

bool octree::readRestartFile(const char *path)
{
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "FATAL: restart: cannot open %s\n", path); return false; }

#ifdef GADGET_HIP_HIGHRES
  // T51: recover the recentring shift from the sidecar. Its ABSENCE is meaningful and correct --
  // a restart written before this existed holds unshifted coordinates -- so this is not an error.
  // A resume must not recompute the shift from the resumed positions: those are already shifted, so
  // recomputing would yield ~0 and the snapshot un-shift would be lost.
  {
    const std::string zp = std::string(path) + ".zoomshift";
    FILE *zf = fopen(zp.c_str(), "r");
    if (zf)
    {
      double sx = 0, sy = 0, sz = 0;
      if (fscanf(zf, "%lg %lg %lg", &sx, &sy, &sz) == 3)
      {
        const double sh[3] = { sx, sy, sz };
        setZoomShift(sh);
        fprintf(stderr, "[ZOOM] resumed recentring shift (%.6g, %.6g, %.6g) from %s\n",
                sx, sy, sz, zp.c_str());
      }
      else
        fprintf(stderr, "[ZOOM] WARNING: %s is unreadable; continuing with NO shift, which will "
                        "offset this run's snapshots if the restart was in fact recentred.\n",
                zp.c_str());
      fclose(zf);
    }
  }
#endif

  GadgetHipRestartHeader h;
  if (fread(&h, sizeof(h), 1, f) != 1) { fprintf(stderr, "FATAL: restart: %s is truncated\n", path); fclose(f); return false; }
  if (h.magic != GADGET_HIP_RESTART_MAGIC)
  { fprintf(stderr, "FATAL: restart: %s is not a restart file (bad magic)\n", path); fclose(f); return false; }
  if (h.version != GADGET_HIP_RESTART_VERSION)
  { fprintf(stderr, "FATAL: restart: %s has version %u, this build writes %u\n", path, h.version, GADGET_HIP_RESTART_VERSION); fclose(f); return false; }

  // Refuse a resume into a DIFFERENT run setup. Silently resuming a comoving run as
  // non-comoving, or against a different timespan, would corrupt the whole continuation the same
  // way an IC/TimeBegin mismatch does -- and just as invisibly.
  if (haveGadgetParams)
  {
    const int ci = gadgetParams.ComovingIntegrationOn;
    if (h.comovingIntegrationOn != ci)
    { fprintf(stderr, "FATAL: restart: file has ComovingIntegrationOn=%d, parameter file says %d\n", h.comovingIntegrationOn, ci); fclose(f); return false; }
    if (h.timeMax > 0.0 && fabs(h.timeMax - gadgetParams.TimeMax) / h.timeMax > 1e-6)
    { fprintf(stderr, "FATAL: restart: file was written for TimeMax=%.10g, parameter file says %.10g\n", h.timeMax, gadgetParams.TimeMax); fclose(f); return false; }
  }

  tree_structure &tree = this->localTree;
  if (h.n != tree.n)
  { fprintf(stderr, "FATAL: restart: file has %d particles, this run allocated %d\n", h.n, tree.n); fclose(f); return false; }
  const int n = h.n;

  bool ok = true;
  ok = ok && fread(&tree.bodies_Ppos[0], sizeof(real4),  n, f) == (size_t) n;
  ok = ok && fread(&tree.bodies_ids[0],  sizeof(ullong), n, f) == (size_t) n;
  ok = ok && fread(&tree.bodies_vel[0],  sizeof(real4),  n, f) == (size_t) n;
  ok = ok && fread(&tree.bodies_Pvel[0], sizeof(real4),  n, f) == (size_t) n;
  ok = ok && fread(&tree.bodies_time[0], sizeof(int2), n, f) == (size_t) n;
  ok = ok && fread(&tree.bodies_acc0[0], sizeof(real4),  n, f) == (size_t) n;
  fclose(f);
  if (!ok) { fprintf(stderr, "FATAL: restart: %s is truncated (short read)\n", path); return false; }

  // The file is in original order, so this IS the load order now; sort_bodies rebuilds
  // oriParticleOrder from it on the first iteration exactly as it does for a fresh IC.
  tree.bodies_Ppos.h2d(); tree.bodies_ids.h2d();
  tree.bodies_vel.h2d();  tree.bodies_Pvel.h2d();
  tree.bodies_time.h2d(); tree.bodies_acc0.h2d();
  // bodies_pos is bound to bodies_Ppos (C-A-22 memory fold), so it needs no separate restore.

  this->set_t_current((float) h.t_current);
  this->t_previous           = (float) h.t_previous;
  this->iter                 = h.iter;
  this->nextGadgetSnapTime   = h.nextGadgetSnapTime;
  this->restartNextEnergyStat= h.nextEnergyStat;
  this->gadgetDtDisplacement = h.gadgetDtDisplacement;
  this->gadgetSnapshotCount  = h.gadgetSnapshotCount;
  this->restartResumed       = true;
  Etot0_prod = h.etot0_prod; Etot1_prod = h.etot1_prod;
  de_max     = h.deMax;      dde_max    = h.ddeMax;
  this->Ekin0 = h.ekin0; this->Epot0 = h.epot0; this->Etot0 = h.etot0;
  this->Ekin1 = h.ekin1; this->Epot1 = h.epot1; this->Etot1 = h.etot1;
  this->store_energy_flag = (h.storeEnergyFlag != 0);

  fprintf(stderr, "[RESTART] resumed from %s : iter=%d t=%.9g n=%d snapCount=%d\n",
          path, h.iter, h.t_current, n, h.gadgetSnapshotCount);
  return true;
}
