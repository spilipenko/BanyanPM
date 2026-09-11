#pragma once

// Phase 5 ticket 05 (PLAN.md): driftfac.c's GSL-integrated drift/grav-kick factor tables
// (PHASE5_ZOOM_COMOVING_SPEC.md Sec 2.1-2.2), reused for both the tree (short-range) and PM
// (long-range) kicks -- this port recomputes PM every step (no coarser "PM step" cadence, unlike
// real Gadget-2), so both force channels already share one cadence and can be kicked with the
// same gravkick factor over the same interval; confirmed by reading gpu_iterate.cpp's PM call
// site directly (unconditional every iteration), not assumed.
//
// Table domain is log(a) from TimeBegin to TimeMax (Gadget-2's own `Time` field IS the scale
// factor `a` directly under comoving integration), log-uniformly spaced, storing the CUMULATIVE
// integral from TimeBegin up to each grid point -- get_drift_factor()/get_gravkick_factor() return
// the difference of two such lookups, i.e. the definite integral over an arbitrary [time0,time1]
// sub-interval.
//
// Deliberately does NOT use GSL (avoids a new external dependency for this port): integration is
// done with a composite Simpson's rule, fine enough (32 sub-panels per table cell, matching the
// table's own log-uniform grid resolution) to comfortably exceed GSL qag's own 1e-8 absolute
// tolerance for these smooth, singularity-free integrands -- validated against an independently
// written host-side reference (not GSL) in this port's own test suite, so bit-identical agreement
// with GSL was never the goal, matching the ticket's own stated exit criterion.
constexpr int GADGET_DRIFT_TABLE_LENGTH = 1000;

struct GadgetDriftTables
{
  double logTimeBegin = 0.0;
  double logTimeMax   = 0.0;
  float  driftTable[GADGET_DRIFT_TABLE_LENGTH]    = {0};
  float  gravKickTable[GADGET_DRIFT_TABLE_LENGTH] = {0};
};

// init_drift_table(), driftfac.c:26-60. Builds both cumulative tables in one pass.
void gadget_init_drift_tables(double timeBegin, double timeMax, double omega0,
                               double omegaLambda, double hubble, GadgetDriftTables &out);

// get_drift_factor()/get_gravkick_factor(), driftfac.c:67-174 (structurally identical for both --
// this is the shared lookup+interpolate+difference logic). `time0`/`time1` are scale factors `a`
// directly (this port's own native time representation, not Gadget-2's integer TIMEBASE ticks --
// see PLAN.md Phase 5.4's own documented reasoning for not building that machinery).
double gadget_get_drift_factor(const GadgetDriftTables &tables, double time0, double time1);
double gadget_get_gravkick_factor(const GadgetDriftTables &tables, double time0, double time1);
