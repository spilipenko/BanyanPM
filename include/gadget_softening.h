#pragma once

// Phase 5 ticket 03 (PLAN.md): Gadget-2 per-type, comoving-aware softening
// (PHASE5_ZOOM_COMOVING_SPEC.md Sec 4.2, gravtree.c:454-504's set_softenings()). Array index is
// the Gadget-2 particle type, 0-5 (0=gas,1=halo,2=disk,3=bulge,4=stars,5=bndry) -- matches
// GadgetParams'/GadgetParticleData's own type convention exactly (Phase 5 tickets 01/02).

// Raw `.param`-supplied values (comoving length + max-physical-length cap), per type.
struct GadgetSofteningParams
{
  double softening[6]        = {0, 0, 0, 0, 0, 0}; // SofteningGas/Halo/Disk/Bulge/Stars/Bndry
  double softeningMaxPhys[6] = {0, 0, 0, 0, 0, 0}; // the corresponding *MaxPhys values
};

// The currently-in-effect values, recomputed whenever `time` (the scale factor `a`, under
// comoving integration) changes meaningfully -- at the same granularity Gadget-2 itself
// recomputes them (tree-rebuild granularity, gravtree.c:51 -- NOT every timestep).
struct GadgetSofteningState
{
  double softeningTable[6] = {0, 0, 0, 0, 0, 0}; // the actual, currently-in-effect softening
  double forceSoftening[6] = {0, 0, 0, 0, 0, 0}; // = 2.8 * softeningTable[t] (the spline kernel's h)
};

// set_softenings(), gravtree.c:454-504, exact formula. Under comoving integration, holds the
// softening fixed in COMOVING units (so it shrinks in physical units as 1/a) until the implied
// physical softening (`softening[t]*time`) would exceed `softeningMaxPhys[t]`, at which point the
// comoving value is capped so the physical softening saturates exactly at the ceiling
// (`softeningTable[t] = softeningMaxPhys[t]/time`). Without comoving integration, the comoving
// value is used directly and the cap never applies -- matches source exactly, not an
// approximation.
void gadget_set_softenings(const GadgetSofteningParams &params, bool comovingIntegrationOn,
                            double time, GadgetSofteningState &out);
