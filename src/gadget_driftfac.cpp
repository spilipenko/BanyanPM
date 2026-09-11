#include "gadget_driftfac.h"
#include "gadget_cosmology.h"

#include <cmath>
#include <functional>

namespace {

// Composite Simpson's rule, `panels` (even) sub-intervals -- fine enough for these smooth,
// singularity-free cosmological integrands (see gadget_driftfac.h's own note on why this replaces
// GSL qag rather than matching it bit-for-bit).
double simpsonIntegrate(const std::function<double(double)> &f, double a, double b, int panels)
{
  if (panels % 2 != 0) panels++;
  const double h = (b - a) / panels;
  double sum = f(a) + f(b);
  for (int i = 1; i < panels; ++i)
  {
    const double x = a + i * h;
    sum += (i % 2 == 0 ? 2.0 : 4.0) * f(x);
  }
  return sum * h / 3.0;
}

double lookupTable(const float *table, double logTimeBegin, double logTimeMax, double time)
{
  const double loga = std::log(time);
  const double u = (loga - logTimeBegin) / (logTimeMax - logTimeBegin) * GADGET_DRIFT_TABLE_LENGTH;
  int i = (int) u;
  if (i >= GADGET_DRIFT_TABLE_LENGTH) i = GADGET_DRIFT_TABLE_LENGTH - 1;
  // driftfac.c's own edge case: with fewer than 2 real table entries behind us, use a linear
  // proxy from the origin instead of interpolating between two (possibly out-of-range) entries.
  if (i <= 1)
    return u * table[0];
  return table[i - 1] + (table[i] - table[i - 1]) * (u - i);
}

} // namespace

void gadget_init_drift_tables(double timeBegin, double timeMax, double omega0,
                               double omegaLambda, double hubble, GadgetDriftTables &out)
{
  out.logTimeBegin = std::log(timeBegin);
  out.logTimeMax   = std::log(timeMax);
  const double dloga = (out.logTimeMax - out.logTimeBegin) / GADGET_DRIFT_TABLE_LENGTH;

  auto Hofa = [&](double a) { return gadget_hubble_a(omega0, omegaLambda, hubble, a); };
  // Substituting x=log(a): da = a*dx, so integrating in x needs an extra factor of `a`.
  auto driftIntegrand = [&](double loga) {
    const double a = std::exp(loga);
    return a / (a * a * a * Hofa(a));
  };
  auto kickIntegrand = [&](double loga) {
    const double a = std::exp(loga);
    return a / (a * a * Hofa(a));
  };

  double cumDrift = 0.0, cumKick = 0.0;
  double prevLoga = out.logTimeBegin;
  for (int i = 0; i < GADGET_DRIFT_TABLE_LENGTH; ++i)
  {
    const double curLoga = out.logTimeBegin + dloga * (i + 1);
    cumDrift += simpsonIntegrate(driftIntegrand, prevLoga, curLoga, 32);
    cumKick  += simpsonIntegrate(kickIntegrand,  prevLoga, curLoga, 32);
    out.driftTable[i]    = (float) cumDrift;
    out.gravKickTable[i] = (float) cumKick;
    prevLoga = curLoga;
  }
}

double gadget_get_drift_factor(const GadgetDriftTables &tables, double time0, double time1)
{
  return lookupTable(tables.driftTable, tables.logTimeBegin, tables.logTimeMax, time1) -
         lookupTable(tables.driftTable, tables.logTimeBegin, tables.logTimeMax, time0);
}

double gadget_get_gravkick_factor(const GadgetDriftTables &tables, double time0, double time1)
{
  return lookupTable(tables.gravKickTable, tables.logTimeBegin, tables.logTimeMax, time1) -
         lookupTable(tables.gravKickTable, tables.logTimeBegin, tables.logTimeMax, time0);
}
