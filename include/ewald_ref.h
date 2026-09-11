#pragma once
#include <cmath>

// Exact periodic (infinite-lattice) Newtonian gravitational force between two point masses in a
// cubic periodic box, via Ewald summation -- G=1 convention, matching this port's own tree/PM
// unit system. Faithful, independently-verified port of Gadget-2's own ewald_force()/ewald_psi()
// (Gadget-2.0.7/Gadget2/forcetree.c:3163-3253): same alpha=2.0 and +-4 real-space/reciprocal-space
// summation range Gadget-2 itself ships (used there to build its own Ewald correction table).
// Host-only, no HIP dependency -- exists purely as ground truth for --pm-test-treepm-force to
// check the combined tree(short-range)+PM(long-range) force against, independent of both.
//
// Self-checked (scratch/ewald_ref.cpp, not shipped) before use here: converges to plain
// Newtonian x/r^2 as boxSize->infinity at fixed physical separation (relative deviation shrinks
// proportionally to r/boxSize, as expected), and gives symmetric components at a symmetric
// (diagonal) test point.
namespace ewald_ref
{

// Correction force (full periodic lattice minus minimum-image direct term), box-size=1 units,
// for source-relative-to-target position x with all |x_i| <= 0.5 (minimum image already applied
// by the caller).
inline void correction(const double x[3], double force[3])
{
    const double alpha = 2.0;
    force[0] = force[1] = force[2] = 0.0;

    const double r2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
    const double r  = std::sqrt(r2);
    for (int i = 0; i < 3; i++)
        force[i] += x[i] / (r2 * r);

    for (int n0 = -4; n0 <= 4; n0++)
    for (int n1 = -4; n1 <= 4; n1++)
    for (int n2 = -4; n2 <= 4; n2++)
    {
        const double dx[3] = { x[0]-n0, x[1]-n1, x[2]-n2 };
        const double rr = std::sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
        const double val = std::erfc(alpha*rr) + 2.0*alpha*rr/std::sqrt(M_PI) * std::exp(-alpha*alpha*rr*rr);
        for (int i = 0; i < 3; i++)
            force[i] -= dx[i] / (rr*rr*rr) * val;
    }

    for (int h0 = -4; h0 <= 4; h0++)
    for (int h1 = -4; h1 <= 4; h1++)
    for (int h2 = -4; h2 <= 4; h2++)
    {
        const int h2sq = h0*h0 + h1*h1 + h2*h2;
        if (h2sq == 0) continue;
        const double hdotx = x[0]*h0 + x[1]*h1 + x[2]*h2;
        const double val = 2.0/(double)h2sq * std::exp(-M_PI*M_PI*h2sq/(alpha*alpha)) * std::sin(2*M_PI*hdotx);
        force[0] -= h0*val;
        force[1] -= h1*val;
        force[2] -= h2*val;
    }
}

// Total exact periodic force (G=1, source mass folded in by the caller via outForce scaling, or
// pass mass=1 and scale outside) on a target at the origin from a unit-mass source at physical
// separation d[3] in a cubic periodic box of side boxSize.
inline void total_force(const double d[3], double boxSize, double outForce[3])
{
    double x[3];
    for (int i = 0; i < 3; i++)
    {
        double xi = d[i] / boxSize;
        xi -= std::round(xi);
        x[i] = xi;
    }
    const double r2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
    const double r  = std::sqrt(r2);
    double corr[3];
    correction(x, corr);

    const double invL2 = 1.0 / (boxSize*boxSize);
    for (int i = 0; i < 3; i++)
        outForce[i] = (x[i]/(r2*r) + corr[i]) * invL2;
}

} // namespace ewald_ref
