#include "gadget_softening.h"

void gadget_set_softenings(const GadgetSofteningParams &params, bool comovingIntegrationOn,
                            double time, GadgetSofteningState &out)
{
  for (int t = 0; t < 6; ++t)
  {
    if (comovingIntegrationOn)
    {
      // gravtree.c:460-488's pattern, generalized over the loop index instead of unrolled per
      // type -- same formula for every type.
      if (params.softening[t] * time > params.softeningMaxPhys[t])
        out.softeningTable[t] = params.softeningMaxPhys[t] / time;
      else
        out.softeningTable[t] = params.softening[t];
    }
    else
    {
      out.softeningTable[t] = params.softening[t];
    }
    out.forceSoftening[t] = 2.8 * out.softeningTable[t];
  }
}
