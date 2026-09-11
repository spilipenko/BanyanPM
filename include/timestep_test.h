#pragma once
#include <hip/hip_runtime.h>

// Phase 5 ticket 04 (PLAN.md): direct-launch test wrapper for the real compute_dt() kernel
// (CUDAkernels/timestep.cu), bypassing the legacy my_dev::kernel .set_args()/.execute2() wrapper
// entirely -- same one-shot-diagnostic idiom as pm.h's softening_test_pair. `h_acc`/
// `h_forceSoftening` are per-particle inputs (n entries each); `h_dtOut[i]` receives the
// resulting `time[i].y - time[i].x` (the chosen timestep) after the kernel runs with every
// particle marked active. `comovingFlag` defaults to 0 (linear `y = tc + dt`, matching this
// test's own existing case set and reference formula, both of which validate the raw `dt`
// clamp/round sequence directly, not the separate tc->y advancement rule -- see compute_dt's own
// comment in timestep.cu, Phase 5 ticket 08 (PLAN.md), for why that rule is now conditional).
void timestep_test_compute_dt(int n, float tc, float errTolIntAccuracy, float atime, float fac1,
                               float hubble_a, float maxSizeTimestep, float minSizeTimestep,
                               float dtDisplacement, const float4 *h_acc,
                               const float *h_forceSoftening, float *h_dtOut, hipStream_t stream,
                               int comovingFlag = 0);

// Phase 5 ticket 05 (PLAN.md): direct-launch test for the device-side driftfac table lookup
// (driftFactorD() in timestep.cu), confirming it matches the host implementation
// (gadget_driftfac.cpp) it's a separate re-implementation of. `h_driftTable`/`h_gravKickTable`
// must each have GADGET_DRIFT_TABLE_LENGTH entries (gadget_driftfac.h). For test case i,
// `h_wantKick[i]` nonzero selects the grav-kick table, zero selects the drift table;
// `h_out[i] = driftFactorD(table, h_t0[i], h_t1[i])`.
void driftfac_test_lookup(const float *h_driftTable, const float *h_gravKickTable,
                           double logTimeBegin, double logTimeMax,   // T29: double
                           const float *h_t0,
                           const float *h_t1, const int *h_wantKick, int n, float *h_out);
