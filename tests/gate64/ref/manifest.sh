# Reference data and tolerances for the 64^3 gate. Sourced by gate64.sh.
#
# Every tolerance here is a MEASURED floor, not a preference. See README.md for the measurements and
# tickets/T46-nondeterminism-noise-floor.md for why a tighter number would fail at random.

# The initial conditions. Not stored in the repo (7.3 MB); checksummed so a different IC cannot be
# silently compared against these reference energies. Regeneration: see README.md 'Provenance'.
GATE64_IC=${GATE64_IC:-/home/sergey/work/gadget-runs/ics/IC_ngenic64_id0}
GATE64_IC_SHA256=81825fbeb322244ef9c170a349b8b93797ddd64eac5e9d17e447068f61bba4ce

# Ekin vs CPU GADGET-2 (the EXTERNAL oracle), percent. Necessarily loose: the port and GADGET-2
# genuinely disagree by up to 1.880% (every-step) and 1.534% (cadence) at 64^3, measured over 8 and 3
# known-good runs. Every parameter has been diffed against the reference run's
# parameters-usedvalues and matches, so this is real resolution-limited disagreement, not a setup
# error. T44 item 4 claimed "reproduces CPU GADGET to 0.7%" -- that was wrong, taken from a partial
# run, and a gate set there would have failed on correct code.
GATE64_EKIN_TOL=2.5

# Ekin vs the stored per-epoch mean of known-good runs OF THIS PORT, percent. This is where the
# sharpness lives: the ~1.9% offset against GADGET is a stable property of the configuration, so a
# change in it is detectable far below the absolute tolerance. Run-to-run noise is 0.42-0.51%
# (T46), so 1.0% is ~2x the floor and ~2.5x sharper than the absolute check.
GATE64_ENVELOPE_TOL=1.0

# sigma(delta) on a 32^3 grid from the z=0 snapshot. Mean of 5 known-good runs; their spread is
# 0.31%, so 3% is ~10x the noise.
GATE64_SIGMA=3.6814
GATE64_SIGMA_TOL=3.0

# Minimum separation for the must-differ checks, percent. Measured effects are 74% (no long-range
# force) and larger (10x impulse), against a 0.31% noise floor.
GATE64_MIN_DIFF=20.0
