// T51 unit test for the periodic centre + shift, compiled with plain g++ against
// pm_zoom_region.cpp -- the whole reason that file is kept free of HIP. The cases that matter are
// the wrapped ones: a naive centre-of-mass gets case 2 exactly wrong (it returns the far side of
// the box), which is the bug this helper exists to avoid.
#include "pm_zoom_region.h"
#include <cstdio>
#include <cmath>
#include <vector>

static int failures = 0;

static void check(const char *what, double got, double want, double tol)
{
  const bool ok = std::fabs(got - want) <= tol;
  if (!ok) failures++;
  printf("  %-44s got %-12.6g want %-12.6g %s\n", what, got, want, ok ? "PASS" : "FAIL");
}

static void checkBool(const char *what, bool got, bool want)
{
  const bool ok = (got == want);
  if (!ok) failures++;
  printf("  %-44s got %-12s want %-12s %s\n", what, got ? "true" : "false",
         want ? "true" : "false", ok ? "PASS" : "FAIL");
}

// Build a cube of high-res (type 1) particles centred on `c` with half-width `h`, wrapped into the
// box, plus some low-res (type 2) filler spread over the whole box to make sure it is ignored.
static void makeClump(double c[3], double h, double L, int perAxis,
                      std::vector<float> &pos, std::vector<int> &types)
{
  pos.clear(); types.clear();
  for (int i = 0; i < perAxis; ++i)
    for (int j = 0; j < perAxis; ++j)
      for (int k = 0; k < perAxis; ++k)
      {
        const double f[3] = { (perAxis == 1) ? 0.0 : -1.0 + 2.0 * i / (perAxis - 1),
                              (perAxis == 1) ? 0.0 : -1.0 + 2.0 * j / (perAxis - 1),
                              (perAxis == 1) ? 0.0 : -1.0 + 2.0 * k / (perAxis - 1) };
        for (int a = 0; a < 3; ++a)
        {
          double x = c[a] + h * f[a];
          while (x >= L) x -= L;
          while (x < 0)  x += L;
          pos.push_back((float) x);
        }
        pos.push_back(1.0f);          // mass, i.e. stride 4 like float4
        types.push_back(1);
      }
  // low-res filler across the whole box -- must not influence the answer
  for (int i = 0; i < 50; ++i)
  {
    for (int a = 0; a < 3; ++a) pos.push_back((float)(L * i / 50.0));
    pos.push_back(1.0f);
    types.push_back(2);
  }
}

int main()
{
  const double L = 64.0;
  const unsigned mask = (1u << 1);   // type 1 only, as --zoom-mask 0x2
  double c[3], e[3];
  const char *err = NULL;
  std::vector<float> pos; std::vector<int> types;

  printf("case 1: clump well inside the box (centre 20,30,40, half-width 4)\n");
  { double cc[3] = {20, 30, 40}; makeClump(cc, 4.0, L, 5, pos, types); }
  checkBool("returns true", pm_zoom_periodic_center(pos.data(), 4, types.data(), types.size(),
                                                    mask, L, c, e, &err), true);
  check("centre x", c[0], 20.0, 1e-4); check("centre y", c[1], 30.0, 1e-4);
  check("centre z", c[2], 40.0, 1e-4); check("extent x", e[0], 8.0, 1e-4);

  printf("\ncase 2: clump STRADDLING the boundary (centre 0,0,0) -- naive COM gives L/2 here\n");
  { double cc[3] = {0, 0, 0}; makeClump(cc, 4.0, L, 5, pos, types); }
  checkBool("returns true", pm_zoom_periodic_center(pos.data(), 4, types.data(), types.size(),
                                                    mask, L, c, e, &err), true);
  // centre may come back as 0 or as L depending on which member is the reference; both are the
  // same point, so accept either.
  check("centre x folded to 0", std::fmin(c[0], L - c[0]), 0.0, 1e-3);
  check("centre y folded to 0", std::fmin(c[1], L - c[1]), 0.0, 1e-3);
  check("extent x", e[0], 8.0, 1e-3);

  printf("\ncase 3: clump straddling on ONE axis only (centre 63.5, 30, 0.5)\n");
  { double cc[3] = {63.5, 30.0, 0.5}; makeClump(cc, 1.0, L, 5, pos, types); }
  checkBool("returns true", pm_zoom_periodic_center(pos.data(), 4, types.data(), types.size(),
                                                    mask, L, c, e, &err), true);
  check("centre x", std::fmin(std::fabs(c[0] - 63.5), std::fabs(c[0] + L - 63.5)), 0.0, 1e-3);
  check("centre y", c[1], 30.0, 1e-3);
  check("centre z", std::fmin(std::fabs(c[2] - 0.5), std::fabs(c[2] - L - 0.5)), 0.0, 1e-3);
  check("extent", e[0], 2.0, 1e-3);

  printf("\ncase 4: species spread over the whole box -- precondition must be REPORTED\n");
  { double cc[3] = {32, 32, 32}; makeClump(cc, 31.0, L, 5, pos, types); }
  checkBool("returns false", pm_zoom_periodic_center(pos.data(), 4, types.data(), types.size(),
                                                     mask, L, c, e, &err), false);
  printf("  reported: %s\n", err ? err : "(no message!)");
  if (!err) failures++;

  printf("\ncase 5: no particle matches the mask\n");
  { double cc[3] = {20, 20, 20}; makeClump(cc, 2.0, L, 3, pos, types); }
  checkBool("returns false", pm_zoom_periodic_center(pos.data(), 4, types.data(), types.size(),
                                                     (1u << 5), L, c, e, &err), false);
  printf("  reported: %s\n", err ? err : "(no message!)");

  printf("\ncase 6: shift then un-shift round-trips, and the shift really centres the clump\n");
  { double cc[3] = {63.5, 1.0, 0.5}; makeClump(cc, 1.0, L, 5, pos, types); }
  std::vector<float> orig = pos;
  pm_zoom_periodic_center(pos.data(), 4, types.data(), types.size(), mask, L, c, e, &err);
  double shift[3], back[3];
  for (int a = 0; a < 3; ++a)
  {
    shift[a] = 0.5 * L - c[a];
    while (shift[a] >= L) shift[a] -= L;
    while (shift[a] < 0)  shift[a] += L;
    back[a] = L - shift[a];
    while (back[a] >= L) back[a] -= L;
  }
  pm_zoom_shift_positions(pos.data(), 4, types.size(), shift, L);
  double c2[3], e2[3];
  checkBool("centre recomputes after shift", pm_zoom_periodic_center(pos.data(), 4, types.data(),
                                               types.size(), mask, L, c2, e2, &err), true);
  check("shifted centre x == L/2", c2[0], 0.5 * L, 1e-3);
  check("shifted centre y == L/2", c2[1], 0.5 * L, 1e-3);
  check("shifted centre z == L/2", c2[2], 0.5 * L, 1e-3);
  check("extent unchanged by the shift", e2[0], e[0], 1e-3);
  pm_zoom_shift_positions(pos.data(), 4, types.size(), back, L);
  double worst = 0.0;
  for (size_t i = 0; i < pos.size(); ++i)
  {
    double d = std::fabs((double) pos[i] - (double) orig[i]);
    if (d > 0.5 * L) d = L - d;          // a coordinate at exactly 0 may come back as L-eps
    if (d > worst) worst = d;
  }
  check("round-trip max coordinate error", worst, 0.0, 1e-4);

  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures,
         failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
