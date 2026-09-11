#pragma once
// Gadget-2's cubic-spline softened pair force and potential, in ONE place.
//
// Why this header exists (contract C-C-03): `--direct` computed a flat Plummer force
// (`distSqr += eps2`, dev_direct_gravity.cu) with the CLI `eps` as the length scale -- a different
// kernel shape AND a length scale 2.8x off from the tree path's `ForceSoftening = 2.8 * Softening`.
// That matters more than its DIAGNOSTIC label suggests: `--direct` is what one reaches for as
// ground truth when the tree looks wrong, so a tree-vs-direct comparison measured neither the tree
// nor the true force. It produced at least one false result in this audit (the quadrupole test,
// where "4.3x worse" turned out to be Plummer-vs-spline, not the quadrupole).
//
// The formula is Gadget-2's, `forcetree.c` (stock :1708-1723 force, :2590-2606 potential),
// verified coefficient-by-coefficient against the reference under contracts C-C-13/C-C-14. The
// literals are the exactly-rounded float32 of the true rationals (32/3, 64/3, 1/15, 16/3, 32/15),
// bit-identical to the reference's 12-digit doubles.
//
// NOTE ON THE REMAINING COPIES: the tree walk still carries its own `add_acc` (two overloads) and
// `add_acc_d`. They are not migrated here in this change, deliberately -- editing the hot walk to
// share this definition is a separate, riskier refactor that acceptance would have to police. This
// header is the canonical definition going forward; the walk copies should be folded into it.

// Softened pair force/potential magnitudes for a source of mass `massj` at separation `r`
// (r2 = r*r), with compact-support softening length `h` (= Gadget's ForceSoftening = 2.8*Softening).
//   *mrinv  -> potential magnitude: caller does  acc.w -= *mrinv
//   *mrinv3 -> force  magnitude:    caller does  acc.xyz += *mrinv3 * dr
static __device__ __forceinline__ void gadget_spline_pair(
    float r, float r2, float massj, float h, float *mrinv, float *mrinv3)
{
  if (r >= h)
  {
    // Exact Newtonian; also the branch taken when h <= 0.
    const float rinv  = (r2 > 0.0f) ? rsqrtf(r2) : 0.0f;
    const float rinv3 = rinv * rinv * rinv;
    *mrinv  = massj * rinv;
    *mrinv3 = massj * rinv3;
  }
  else
  {
    const float h_inv  = 1.0f / h;
    const float h3_inv = h_inv * h_inv * h_inv;
    const float u      = r * h_inv;
    float wp;
    if (u < 0.5f)
    {
      *mrinv3 = massj * h3_inv * (10.666666667f + u*u*(32.0f*u - 38.4f));
      wp      = -2.8f + u*u*(5.333333333f + u*u*(6.4f*u - 9.6f));
    }
    else
    {
      *mrinv3 = massj * h3_inv * (21.333333333f - 48.0f*u + 38.4f*u*u -
                                   10.666666667f*u*u*u - 0.066666667f/(u*u*u));
      wp      = -3.2f + 0.066666667f/u +
                u*u*(10.666666667f + u*(-16.0f + u*(9.6f - 2.133333333f*u)));
    }
    *mrinv = -massj * h_inv * wp;
  }
}
