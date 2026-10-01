#ifndef GADGET_UNITS_H
#define GADGET_UNITS_H

// T44 item 1 -- making a unit convention executable instead of documented.
//
// THE CONVENTION. The tree kernel and every PM call produce G-FREE accelerations, mirroring
// GADGET-2's forcetree.c, and the factor G = 43.007106 (for the usual unit system) is applied ONCE
// downstream -- by pm_scale_acc_masked for bodies_acc1, and by the consumer itself for anything that
// does not pass through acc1.
//
// THE TRAP. Twice now a path has reached a consumer without meeting that multiplication:
//
//   T4-FIX-G   the whole per-step dynamics silently ran at G=1. Invisible in any test at G=1,
//              because there "forgot to multiply by G" and "multiplied by 1" agree.
//   T42        the Phase 3 long-range kick wrote `gpmx[id] * dtGravKick`. GravPM is stored before
//              pm_scale_acc_masked and the cadence path never enters acc1, so it never met the
//              scaling: the impulse was 43x too small. It reproduced "no long-range force at all"
//              while passing every check on the impulse itself -- right particle, right direction,
//              right count, right time factor, and the one scale factor nobody multiplied by. The
//              comment three lines above the scaling call even names the number 43; it was read
//              during the investigation and the bug was still written. A day to find.
//
// WHY A TYPE AND NOT A COMMENT. Prose cannot stop you writing `gpmx[id] * dtGravKick`. This can:
// GFreeAcc has no conversion to float and no operator* for ordinary scalars. The only multiplication
// it admits is by a GCarrier -- a scalar that already includes G -- so the factor has to be named
// somewhere, by someone, before the expression compiles.
//
// WHY THE G LIVES IN THE MULTIPLIER rather than in an acc.with_G(G) accessor: it keeps the
// arithmetic bit-for-bit what it was. v * (dt*G) and (v*G) * dt are not the same float. Putting G in
// the time factor, where the host already computed it, means this whole change is provably
// behaviour-neutral -- verified by binary comparison, see T44 item 2's note on that instrument.
//
// ESCAPES are deliberately verbose. If a value legitimately stays G-free because something
// downstream will scale it, say so at the point where it is unwrapped.

struct GCarrier;

// One component (x, y, z) of an acceleration that does NOT yet include G.
// Layout-identical to float, trivially copyable, usable as a kernel parameter type.
struct GFreeAcc
{
  float v;

  // The only arithmetic there is. Multiplying by anything else does not compile, which is the
  // entire point of this file.
  __host__ __device__ inline float operator*(GCarrier k) const;

  // For the path where G really is applied later, by pm_scale_acc_masked, together with the tree
  // force this is being added to. Named at length so that using it is a decision, not a reflex.
  __host__ __device__ float raw_because_acc1_is_G_scaled_downstream() const { return v; }

  // Diagnostics only: rms, probes, host dumps. Never for anything that reaches a trajectory.
  __host__ __device__ float raw_for_diagnostics() const { return v; }
};

// A scalar that already carries the factor G. The named constructors are the only way to make one,
// so "which G did this come from" always has an answer at the call site.
struct GCarrier
{
  // dt_gravkick * G -- the long-range kick factor, and the energy diagnostic's drift correction.
  static __host__ __device__ GCarrier gravkick_times_G(float dtGravKickTimesG)
  { GCarrier c; c.v = dtGravKickTimesG; return c; }

  // Bare G, for consumers that combine a G-free force with an already-G-scaled one (the Springel
  // MAC's aold: GADGET writes GravAccel + GravPM/All.G, gravtree.c:309-311).
  static __host__ __device__ GCarrier just_G(float G)
  { GCarrier c; c.v = G; return c; }

  // A deliberate no-op, for code paths that must run with the long-range term switched off (the
  // Phase 3 bisect probes). Spelled out so it cannot be reached by accident.
  static __host__ __device__ GCarrier zero_because_long_range_is_disabled()
  { GCarrier c; c.v = 0.0f; return c; }

  float v;
};

__host__ __device__ inline float GFreeAcc::operator*(GCarrier k) const { return v * k.v; }

#endif
