#ifndef DEVFUNCTIONDEFINITIONS_H
#define DEVFUNCTIONDEFINITIONS_H

// Host-side declarations of every GPU kernel launched through my_dev::kernel.
//
// GENERATED from the definitions by tools/check_kernel_decls.py --write. Do not hand-edit: run the
// tool. Each kernel-defining .cu now INCLUDES this header, so the compiler compares declaration
// against definition and a divergence is a build error rather than a latent one.
//
// Why that matters: these declarations are used only for their ADDRESS (load_kernels.cpp passes
// &kernel to hipLaunchKernel), so for years a wrong signature here compiled, linked and ran
// silently. When the comparison was first switched on it found 25 divergences, including
// compute_dt declared with 12 parameters against 25 real ones, correct_particles 13 against 22,
// get_Tnext with float2* where the Phase 2 integer timeline made it int2*, and setActiveGroups
// with a float second parameter that is an int. None was an active bug precisely because nothing
// ever read these types -- which is also what made the header useless as documentation and unusable
// for type-checking set_args (T44 item 2).

// Self-contained: hip_runtime.h for float4/uint2/..., node_specs.h for uint/real/real4/bodyStruct
// and setupParams. setupParams used to live in octree.h, a host-only header, so this file carried a
// layout-compatible duplicate called setupParams2 and the three scanKernels entries were declared
// with a DIFFERENT TYPE than they are defined with. It now lives in node_specs.h, visible to both.
#include "hip/hip_runtime.h"
#include "node_specs.h"
// GFreeAcc / GCarrier: the factor G made executable (T44 item 1). Several kernel parameters are
// declared with these types, so this header cannot be parsed without them.
#include "gadget_units.h"
#include "gadget_index_spaces.h"

// Macro-generated in dev_approximate_gravity_warp_new.cu; not extracted by the tool.
extern "C" void  (dev_approximate_gravity)(const int n_active_groups, int n_bodies, float eps2, uint2 node_begend, int *active_groups, float4 *body_pos, float4 *multipole_data, float4 *acc_out, float4 *group_body_pos, int *ngb_out, int *active_inout, int2 *interactions, float4 *boxSizeInfo, float4 *groupSizeInfo, float4 *boxCenterInfo, float4 *groupCenterInfo, float4 *body_vel, int *MEM_BUF);
extern "C" void  (dev_approximate_gravity_let)(const int n_active_groups, int n_bodies, float eps2, uint2 node_begend, int *active_groups, float4 *body_pos, float4 *multipole_data, float4 *acc_out, float4 *group_body_pos, int *ngb_out, int *active_inout, int2 *interactions, float4 *boxSizeInfo, float4 *groupSizeInfo, float4 *boxCenterInfo, float4 *groupCenterInfo, float4 *body_vel, int *MEM_BUF);

extern "C" void  (assignColorsKernel)(float4 *colors, ulonglong1 *ids, int numParticles, float2 *density, float maxDensity, float4 color2, float4 color3, float4 color4, float4 starColor, float4 bulgeColor, float4 darkMatterColor, float4 dustColor, int m_brightFreq, float4 t_current);   // depthSort.cu
extern "C" void  (build_group_list2)(const int n_particles, uint *validList, const uint2 startLevelBeginEnd, uint2 *node_bodies, int *node_level_list, int treeDepth);   // build_tree.cu
extern "C" void  (calcDepthKernel)(float4 *pos, float *depth, int *indices, float4 modelViewZ, int numParticles);   // depthSort.cu
extern "C" void  (cl_build_key_list)(uint4 *body_key, float4 *body_pos, int n_bodies, float4 corner);   // build_tree.cu
extern "C" void  (cl_build_nodes)(uint level, uint *compact_list_len, uint *level_offset, uint *last_level, uint2 *level_list, uint *compact_list, uint4 *bodies_key, uint4 *node_key, uint *n_children, uint2 *node_bodies);   // build_tree.cu
extern "C" void  (cl_build_valid_list)(int n_bodies, int level, uint4 *body_key, uint *valid_list, const uint *workToDo);   // build_tree.cu
extern "C" void  (cl_link_tree)(int n_nodes, uint *n_children, uint2 *node_bodies, float4 *bodies_pos, float4 corner, uint2 *level_list, uint* valid_list, uint4 *node_keys, uint4 *bodies_key, uint levelMin);   // build_tree.cu
extern "C" void  (compact_count)(volatile uint2 *values, uint *counts, const int N, setupParams sParam, const uint *workToDo);   // scanKernels.cu
extern "C" void  (compact_move)(uint2 *values, uint *output, uint *counts, const int N, setupParams sParam, const uint *workToDo);   // scanKernels.cu
extern "C" void  (compute_dt)(const int n_bodies, float tc, float errTolIntAccuracy, float atime, float fac1, float hubble_a, float maxSizeTimestep, float minSizeTimestep, float dtDisplacement, int comovingFlag, CurrentOrder<int> newEnd, OriginalOrder<int2> time, CurrentOrder<const uint> unsorted, CurrentOrder<const float4> bodies_acc, CurrentOrder<const float> bodies_forceSoftening, CurrentOrder<const uint> active_list, float timelineSpan, float timelineOrigin, float timeMax, double dPerTick, int Ti_current, CurrentOrder<const unsigned long long> ids, IdSpace<const GFreeAcc> gpmx, IdSpace<const GFreeAcc> gpmy, IdSpace<const GFreeAcc> gpmz, const GCarrier G);   // timestep.cu
extern "C" void  (compute_energy_double)(const int n_bodies, float4 *pos, float4 *vel, float4 *acc, const float *forceSoftening, double selfCoef, double cc10Coef, double2 *energy);   // timestep.cu
extern "C" void  (compute_energy_double_prekick)(const int n_bodies, CurrentOrder<const float4> pos, OriginalOrder<const float4> vel, CurrentOrder<const float4> acc, CurrentOrder<const uint> unsorted, OriginalOrder<const int2> time, float tc, int comovingFlag, const float *gravKickTable, double logTimeBegin, double dPerTick, double logTimeMax, CurrentOrder<const float> forceSoftening, double selfCoef, double cc10Coef, CurrentOrder<const unsigned long long> ids, IdSpace<const GFreeAcc> gpmx, IdSpace<const GFreeAcc> gpmy, IdSpace<const GFreeAcc> gpmz, GCarrier dtGravKickPM, double2 *energy);   // timestep.cu
extern "C" void  (compute_leaf)(const int n_leafs, uint *leafsIdxs, uint2 *node_bodies, float4 *body_pos, double4 *multipole, float4 *nodeLowerBounds, float4 *nodeUpperBounds, float4 *body_vel, ulonglong1 *body_id, float *body_h, const float h_min, const float *body_forceSoftening, float *nodeSoftInfo);   // compute_propertiesD.cu
extern "C" void  (compute_non_leaf)(const int curLevel, uint *leafsIdxs, uint *node_level_list, uint *n_children, double4 *multipole, float4 *nodeLowerBounds, float4 *nodeUpperBounds, float *nodeSoftInfo);   // compute_propertiesD.cu
extern "C" void  (compute_scaling)(const int node_count, double4 *multipole, float4 *nodeLowerBounds, float4 *nodeUpperBounds, uint *n_children, float4 *multipoleF, float theta, float4 *boxSizeInfo, float4 *boxCenterInfo, uint2 *node_bodies, float domain_fac, float *cellSizeInfo, float4 corner, float4 *cellCenterInfo);   // compute_propertiesD.cu
extern "C" void  (correct_particles)(const int n_bodies, float tc, OriginalOrder<int2> time, CurrentOrder<const uint> active_list, CurrentOrder<float4> vel, OriginalOrder<const float4> acc0, CurrentOrder<const float4> acc1, CurrentOrder<float> body_h, CurrentOrder<const float2> body_dens, CurrentOrder<float4> pos, CurrentOrder<const float4> pPos, OriginalOrder<const float4> pVel, CurrentOrder<uint> unsorted, CurrentOrder<float4> acc0_new, CurrentOrder<int2> time_new, int comovingIntegrationOn, const float *gravKickTable, double logTimeBegin, double logTimeMax, CurrentOrder<const int> newEnd, double dPerTick, int Ti_current);   // timestep.cu
extern "C" void  (dev_direct_gravity)(float4 *accel, float4 *i_positions, float4 *j_positions, int numBodies_i, int numBodies_j, const float *i_soft, const float *j_soft, float boxSize);   // dev_direct_gravity.cu
extern "C" void  (exclusive_scan_block)(int *ptr, const int N, int *count);   // scanKernels.cu
extern "C" void  (gadget_aold_mag)(const int n_bodies, OriginalOrder<const float4> acc, CurrentOrder<const uint> unsorted, CurrentOrder<const unsigned long long> ids, IdSpace<const GFreeAcc> gpmx, IdSpace<const GFreeAcc> gpmy, IdSpace<const GFreeAcc> gpmz, const GCarrier G, OriginalOrder<float> magOri);   // timestep.cu
extern "C" void  (gadget_box_wrap)(const int n_bodies, const float boxSize, float4 *pPos);   // timestep.cu
extern "C" void  (gadget_pm_kick)(const int n_bodies, OriginalOrder<float4> vel, OriginalOrder<float4> vel2, CurrentOrder<const uint> unsorted, CurrentOrder<const unsigned long long> ids, IdSpace<const GFreeAcc> gpmx, IdSpace<const GFreeAcc> gpmy, IdSpace<const GFreeAcc> gpmz, const GCarrier dtGravKick, unsigned char *kickAudit, const int noPerm);   // timestep.cu
extern "C" void  (gadget_pm_staleness)(const int n_bodies, CurrentOrder<const unsigned long long> ids, CurrentOrder<const float> fx, CurrentOrder<const float> fy, CurrentOrder<const float> fz, IdSpace<const GFreeAcc> gpmx, IdSpace<const GFreeAcc> gpmy, IdSpace<const GFreeAcc> gpmz);   // timestep.cu
extern "C" void  (gadget_refresh_softening)(const int n, const ullong *ids, const unsigned char *typeById, const ullong maxId, const float4 forceSoftLo, const float2 forceSoftHi, float *forceSoftening, int *typeDevice);   // timestep.cu
extern "C" void  (gadget_timestep_globals)(const int n, const float4 *pos, const float4 *vel, const uint *unsorted, const int *typeDevice, double *outVSum, double *outMinMass, unsigned long long *outCount);   // timestep.cu
extern "C" void  (get_Tnext)(const int n_bodies, int2 *time, int *tnext);   // timestep.cu
extern "C" void  (get_nactive)(const int n_bodies, uint *valid, uint *tnact);   // timestep.cu
extern "C" void  (gpu_boundaryReduction)(const int n_particles, float4 *positions, float3 *output_min, float3 *output_max);   // build_tree.cu
extern "C" void  (gpu_boundaryReductionGroups)(const int n_groups, float4 *positions, float4 *sizes, float3 *output_min, float3 *output_max);   // build_tree.cu
extern "C" void  (gpu_build_level_list)(const int n_nodes, const int n_leafs, uint *leafsIdxs, uint2 *node_bodies, uint *valid_list);   // build_tree.cu
extern "C" void  (gpu_domainCheckSFC)(int n_bodies, uint4 lowBoundary, uint4 highBoundary, uint4 *body_key, int *validList);   // parallel.cu
extern "C" void  (gpu_domainCheckSFCAndAssign)(int n_bodies, int nProcs, uint4 lowBoundary, uint4 highBoundary, uint4 *boundaryList, uint4 *body_key, uint2 *validList, uint *idList, int procId);   // parallel.cu
extern "C" void  (gpu_extractOutOfDomainParticlesAdvancedSFC2)(int offset, int n_extract, uint2 *extractList, float4 *Ppos, float4 *Pvel, float4 *pos, float4 *vel, float4 *acc0, float4 *acc1, float2 *time, unsigned long long *body_id, uint4 *body_key, float *h, bodyStruct *destination);   // parallel.cu
extern "C" void  (gpu_extractSampleParticlesSFC)(int n_bodies, int nSamples, float sample_freq, uint4 *body_pos, uint4 *samplePosition);   // parallel.cu
extern "C" void  (gpu_insertNewParticlesSFC)(int n_extract, int n_insert, int n_oldbodies, int offset, float4 *Ppos, float4 *Pvel, float4 *pos, float4 *vel, float4 *acc0, float4 *acc1, float2 *time, unsigned long long *body_id, uint4 *body_key, float *h, bodyStruct *source);   // parallel.cu
extern "C" void  (gpu_internalMove)(int n_extract, int n_bodies, double4 xlow, double4 xhigh, int *extractList, int *indexList, float4 *Ppos, float4 *Pvel, float4 *pos, float4 *vel, float4 *acc0, float4 *acc1, float2 *time, int *body_id);   // parallel.cu
extern "C" void  (gpu_internalMoveSFC)(int n_extract, int n_bodies, uint4 lowBoundary, uint4 highBoundary, int *extractList, int *indexList, float4 *Ppos, float4 *Pvel, float4 *pos, float4 *vel, float4 *acc0, float4 *acc1, float2 *time, unsigned long long *body_id, uint4 *body_key);   // parallel.cu
extern "C" void  (gpu_internalMoveSFC2)(int n_extract, int n_bodies, uint4 lowBoundary, uint4 highBoundary, int2 *extractList, int *indexList, float4 *Ppos, float4 *Pvel, float4 *pos, float4 *vel, float4 *acc0, float4 *acc1, float2 *time, unsigned long long *body_id, uint4 *body_key, float *h);   // parallel.cu
extern "C" void  (gpu_setPHGroupData)(const int n_groups, const int n_particles, float4 *bodies_pos, int2 *group_list, float4 *groupCenterInfo, float4 *groupSizeInfo);   // compute_propertiesD.cu
extern "C" void  (gpu_setPHGroupDataGetKey)(const int n_groups, const int n_particles, float4 *bodies_pos, int2 *group_list, float4 *groupCenterInfo, float4 *groupSizeInfo, uint4 *body_key, float4 corner);   // compute_propertiesD.cu
extern "C" void  (gpu_setPHGroupDataGetKey2)(const int n_groups, float4 *bodies_pos, int2 *group_list, uint4 *body_key, float4 corner);   // compute_propertiesD.cu
extern "C" void  (predict_particles)(const int n_bodies, float tc, float tp, float4 *pos, float4 *vel, float4 *acc, int2 *time, float4 *pPos, float4 *pVel, int comovingIntegrationOn, const float *driftTable, const float *gravKickTable, double logTimeBegin, double dPerTick, int tp_ti, double logTimeMax);   // timestep.cu
extern "C" void  (setActiveGroups)(const int n_bodies, int tc, OriginalOrder<int2> time, CurrentOrder<const uint> body2grouplist, GroupOrder<uint> valid_list, const int n_groups, CurrentOrder<const uint> oriParticleOrder);   // timestep.cu
extern "C" void  (split_move)(uint2 *valid, uint *output, uint *counts, const int N, setupParams sParam);   // scanKernels.cu
extern "C" void  (store_group_list)(int n_particles, int n_groups, uint *validList, uint *body2group_list, uint2 *group_list);   // build_tree.cu

#endif
