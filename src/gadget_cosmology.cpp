#include "gadget_cosmology.h"

#include <cmath>

double gadget_hubble_a(double Omega0, double OmegaLambda, double Hubble, double time)
{
  // timestep.c:52-55, exact formula.
  const double h2 = Omega0 / (time * time * time) +
                     (1.0 - Omega0 - OmegaLambda) / (time * time) + OmegaLambda;
  return Hubble * std::sqrt(h2);
}

double gadget_find_dt_displacement_constraint(
    bool comovingIntegrationOn, double omega0, double omegaBaryon, double G, double hubble,
    double maxSizeTimestep, double maxRMSDisplacementFac, double hfac,
    const std::vector<int> &types, const std::vector<float> &mass,
    const std::vector<float> &velSqr,
    bool havePMGRID, double asmth0, double asmth1, unsigned int highResMask)
{
  double dt_displacement = maxSizeTimestep;
  if (!comovingIntegrationOn)
    return dt_displacement;

  double vSum[6]   = {0, 0, 0, 0, 0, 0};
  double minMass[6] = {1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30, 1.0e30};
  long long count[6] = {0, 0, 0, 0, 0, 0};

  const size_t n = types.size();
  for (size_t i = 0; i < n; ++i)
  {
    const int t = types[i];
    vSum[t] += velSqr[i];
    if (mass[i] < minMass[t]) minMass[t] = mass[i];
    count[t]++;
  }

  // T35: everything past this point depends on the per-particle data only through these three
  // arrays, so it is shared with the aggregate overload rather than duplicated.
  return gadget_find_dt_displacement_constraint_agg(
      comovingIntegrationOn, omega0, omegaBaryon, G, hubble, maxSizeTimestep,
      maxRMSDisplacementFac, hfac, count, vSum, minMass, havePMGRID, asmth0, asmth1, highResMask);
}

double gadget_find_dt_displacement_constraint_agg(
    bool comovingIntegrationOn, double omega0, double omegaBaryon, double G, double hubble,
    double maxSizeTimestep, double maxRMSDisplacementFac, double hfac,
    const long long count[6], const double vSum[6], const double minMass[6],
    bool havePMGRID, double asmth0, double asmth1, unsigned int highResMask)
{
  double dt_displacement = maxSizeTimestep;
  if (!comovingIntegrationOn)
    return dt_displacement;

  for (int type = 0; type < 6; ++type)
  {
    if (count[type] <= 0) continue;

    // timestep.c:606-618 -- rho_crit = 3*Hubble^2/(8*pi*G); the type's mean density is its own
    // Omega share of that; dmean is the mean interparticle spacing implied by that density and
    // the type's own minimum particle mass.
    const double rhoCrit = 3.0 * hubble * hubble / (8.0 * M_PI * G);
    const double omegaType = (type == 0) ? omegaBaryon : (omega0 - omegaBaryon);
    const double dmean = std::cbrt(minMass[type] / (omegaType * rhoCrit));

    const double meanV = std::sqrt(vSum[type] / (double) count[type]);
    double dt = maxRMSDisplacementFac * hfac * dmean / meanV;

    // timestep.c:623-628, exact formula: PMGRID/PLACEHIGHRESREGION tightens the ceiling to
    // whichever grid's own Asmth this type belongs to, IF that's a finer scale than dmean.
    if (havePMGRID)
    {
      double asmth = asmth0;
      if ((1u << type) & highResMask) asmth = asmth1;
      if (asmth < dmean) dt = maxRMSDisplacementFac * hfac * asmth / meanV;
    }

    if (dt < dt_displacement) dt_displacement = dt;
  }

  return dt_displacement;
}
