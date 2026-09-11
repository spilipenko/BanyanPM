#pragma once
#include <vector>

// Phase 5 ticket 04 (PLAN.md): shared cosmology helpers, factored out here (not into
// gadget_softening.h) because ticket 05 (comoving integration / driftfac.c tables) will reuse
// gadget_hubble_a() too -- prefactored now rather than duplicated later.

// H(a) = Hubble * sqrt(Omega0/a^3 + (1-Omega0-OmegaLambda)/a^2 + OmegaLambda), exactly as
// computed in Gadget-2's own timestep.c:52-55 (and driftfac.c's integrands, which use the
// identical formula -- PHASE5_ZOOM_COMOVING_SPEC.md Sec 2.1/3.1). `Hubble` here is
// `All.Hubble = HUBBLE_cgs * UnitTime_in_s` (ticket 01's `GadgetParams::Hubble`), not H0 in
// physical units.
double gadget_hubble_a(double Omega0, double OmegaLambda, double Hubble, double time);

// find_dt_displacement_constraint(), timestep.c:566-644, now including the PMGRID/
// PLACEHIGHRESREGION Asmth[0]/[1] refinement (timestep.c:623-628) -- previously deferred here
// (Ticket 04's own scope note stubbed it pending Ticket 07's tree-side Rcut/Asmth plumbing, PLAN.md
// Phase 5.4/5.6), now implemented: for each type, `asmth = havePMGRID ? asmth0 : <unused>`,
// switched to `asmth1` for any type in `highResMask`'s bitmask (Gadget-2's own PLACEHIGHRESREGION
// bit test, `(1<<type) & highResMask`), and if `asmth < dmean` the displacement ceiling uses
// `asmth` in place of `dmean` (a TIGHTER, never-looser bound) -- exact source formula, not an
// approximation. `havePMGRID=false` skips the refinement entirely, matching source's own
// `#ifdef PMGRID` gate (the function still exactly reproduces Gadget-2's non-PMGRID behavior then).
//
// Returns `maxSizeTimestep` unconditionally (matching source exactly) if `comovingIntegrationOn`
// is false, or if no particle type has any members. `types`/`mass`/`velSqr` are per-particle
// (Gadget-2 type 0-5, particle mass, and |velocity|^2) -- single-process only (this port has no
// MPI, so the MPI_Allreduce/MPI_Allgather sums in the source collapse to a single local pass,
// matching Gadget-2's own NTask==1 degenerate case exactly).
double gadget_find_dt_displacement_constraint(
    bool comovingIntegrationOn, double omega0, double omegaBaryon, double G, double hubble,
    double maxSizeTimestep, double maxRMSDisplacementFac, double hfac,
    const std::vector<int> &types, const std::vector<float> &mass,
    const std::vector<float> &velSqr,
    bool havePMGRID = false, double asmth0 = 0.0, double asmth1 = 0.0,
    unsigned int highResMask = 0);

// T35 performance: the same function taking the PER-TYPE AGGREGATES directly, instead of three
// per-particle vectors it would immediately reduce to exactly these. The per-particle overload
// above is unchanged and still used where n is small or the caller has the arrays anyway; this one
// exists so the reduction can be done on the device, where the data already lives. At 512^3 the
// per-particle form cost 1.06 s per rebuild in host allocation, a 4.8 GB device-to-host copy and a
// 134M-element loop, to produce eighteen numbers.
//
// `count`, `vSum` and `minMass` are exactly the three arrays the per-particle overload builds:
// count[t] = number of particles of type t, vSum[t] = sum of |v|^2 over them, minMass[t] = their
// smallest mass (1e30 for an empty type, matching the source's own initialiser).
double gadget_find_dt_displacement_constraint_agg(
    bool comovingIntegrationOn, double omega0, double omegaBaryon, double G, double hubble,
    double maxSizeTimestep, double maxRMSDisplacementFac, double hfac,
    const long long count[6], const double vSum[6], const double minMass[6],
    bool havePMGRID = false, double asmth0 = 0.0, double asmth1 = 0.0,
    unsigned int highResMask = 0);
