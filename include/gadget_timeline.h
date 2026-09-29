#ifndef GADGET_TIMELINE_H
#define GADGET_TIMELINE_H

#include <cmath>

// GADGET-2 keeps simulation time on an INTEGER timeline: Ti counts ticks of Timebase_interval
// between TimeBegin and TimeMax (allvars.h: `int Ti_Current`, `int Ti_begstep`). Two particles
// that should re-synchronise therefore hold bit-identical values by construction, and a
// power-of-two bin boundary is exact.
//
// This port stored the scale factor `a` as float32 instead, which makes a rendezvous approximate:
// a coarse-bin particle and the fine-bin particles it should meet disagree by one ULP, the meeting
// is missed, and the run spends an entire system step -- tree rebuild, PM solve, kick -- advancing
// the clock by that ULP. Measured: 1 step in 600 on a uniform box (tolerable), but 84-96% of all
// steps on a multi-species zoom IC, 68,780 steps against GADGET-2's 2,690 for the same physics
// (T24, T38, ZOOM-REPORT-2026-09-17).
//
// Ticks are the state of record. `a` is DERIVED from ticks at the boundaries that genuinely need a
// scale factor (drift/kick table lookups, snapshot headers, output-epoch comparisons); nothing
// stores `a` as state.

// Matches GADGET-2's own TIMEBASE. The span is mapped onto [0, TIMEBASE], which must be a power of
// two so that bin boundaries divide it exactly.
#define GADGET_TIMEBASE (1 << 28)

typedef int gadget_tick_t;   // as GADGET-2: `int Ti_Current`

// The timeline is uniform in log(a) under comoving integration and linear in t otherwise, matching
// GADGET-2's Timebase_interval (init.c:51/56).
struct GadgetTimeline
{
  double logTimeBegin;   // log(TimeBegin), or TimeBegin itself when not comoving
  double logTimeMax;     // log(TimeMax),   or TimeMax   itself when not comoving
  double dPerTick;       // (logTimeMax - logTimeBegin) / TIMEBASE
  double timeBegin;      // kept verbatim so the endpoints are returned EXACTLY
  double timeMax;
  int    comoving;

  void init(double timeBegin_, double timeMax_, int comovingOn)
  {
    comoving     = comovingOn;
    timeBegin    = timeBegin_;
    timeMax      = timeMax_;
    logTimeBegin = comovingOn ? std::log(timeBegin_) : timeBegin_;
    logTimeMax   = comovingOn ? std::log(timeMax_)   : timeMax_;
    dPerTick     = (logTimeMax - logTimeBegin) / (double) GADGET_TIMEBASE;
  }

  // Ticks -> scale factor (or linear time). Always evaluated in double.
  double toTime(gadget_tick_t ti) const
  {
    // exp(log(x)) is not exactly x in double, so returning the stored endpoints keeps the IC epoch
    // check and the TimeMax truncation (C-D-10) exact rather than one ULP off.
    if (ti == 0)               return timeBegin;
    if (ti == GADGET_TIMEBASE) return timeMax;
    const double x = logTimeBegin + (double) ti * dPerTick;
    return comoving ? std::exp(x) : x;
  }

  // Scale factor (or linear time) -> nearest tick. Used only at boundaries: reading an IC epoch,
  // an output list, a restart file written by an older version.
  gadget_tick_t toTick(double t) const
  {
    const double x = comoving ? std::log(t) : t;
    double n = (x - logTimeBegin) / dPerTick;
    n = std::floor(n + 0.5);
    if (n < 0.0) n = 0.0;
    if (n > (double) GADGET_TIMEBASE) n = (double) GADGET_TIMEBASE;
    return (gadget_tick_t) n;
  }

  // The log-space width of a tick interval, which is what the drift and kick factor tables want.
  double logSpan(gadget_tick_t tiFrom, gadget_tick_t tiTo) const
  {
    return (double) (tiTo - tiFrom) * dPerTick;
  }
};

// Largest power-of-two tick count not exceeding `want`, clamped to [1, TIMEBASE]. GADGET-2 derives
// a particle's step this way (timestep.c:348-350) so that every bin length divides TIMEBASE and
// the bins form a nested lattice -- which is what makes the synchronisation rule satisfiable.
__host__ __device__ inline gadget_tick_t gadget_tick_pow2_floor(double want)
{
  if (!(want > 0.0)) return 1;
  gadget_tick_t s = GADGET_TIMEBASE;
  while ((double) s > want && s > 1) s >>= 1;
  return s;
}

#endif // GADGET_TIMELINE_H
