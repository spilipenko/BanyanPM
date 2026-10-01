#!/bin/bash
# T44 items 4 and 5 -- the 64^3 commit gate.
#
# Four runs of a 262144-particle cosmological box to z=0, about 30 s each, checked against a CPU
# GADGET-2 reference. Total ~2.5 minutes. The point is to turn a bad assumption into a failed test
# in minutes instead of a 7.5-hour 512^3 run, or a day of bisection.
#
# It would have caught T42's missing factor G on the first run: without the long-range force
# sigma(delta) at z=0 falls from 3.68 to 0.95.
#
# Usage:  gate64.sh <path-to-gadget_hip>   [--keep]
#         gate64.sh --help
#
# The binary must be a PMGRID=64, PERIODIC=ON, MAC_SPRINGEL=ON build. The gate checks this by
# reading the port's own startup banner rather than trusting the caller.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/ref/manifest.sh"

if [ "${1:-}" = "--help" ] || [ -z "${1:-}" ]; then
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0
fi
BIN=$1; shift
KEEP=0; [ "${1:-}" = "--keep" ] && KEEP=1
WORK=${GATE64_WORKDIR:-$(mktemp -d /tmp/gate64.XXXXXX)}
mkdir -p "$WORK"
FAILED=0
step() { printf '\n[%s] %s\n' "$1" "$2"; }
fail() { FAILED=$((FAILED+1)); }

# ---------------------------------------------------------------- preflight
step PREFLIGHT "binary, IC, reference data, python"
[ -x "$BIN" ] || { echo "  FAIL  not executable: $BIN"; exit 2; }
if [ ! -f "$GATE64_IC" ]; then
  echo "  FAIL  IC not found: $GATE64_IC"
  echo "        Regenerate with N-GenIC -- see README.md 'Provenance'."
  exit 2
fi
ic_sha=$(sha256sum "$GATE64_IC" | cut -d' ' -f1)
if [ "$ic_sha" != "$GATE64_IC_SHA256" ]; then
  echo "  FAIL  IC checksum mismatch -- the reference energies do not describe this IC."
  echo "        expected $GATE64_IC_SHA256"
  echo "        found    $ic_sha"
  exit 2
fi
echo "  ok    IC $GATE64_IC (sha256 matches)"
python3 -c 'import numpy' 2>/dev/null || { echo "  FAIL  python3 + numpy required"; exit 2; }
echo "  ok    python3 + numpy"

# ---------------------------------------------------------------- run helper
# $1 = name, rest = environment assignments
run() {
  local name=$1; shift
  local d="$WORK/$name"
  rm -rf "$d"; mkdir -p "$d/out"
  sed -e "s|@IC@|$GATE64_IC|" -e "s|@OUTDIR@|$d/out|" "$HERE/ref/gate64.param.in" > "$d/run.param"
  local t0=$SECONDS
  ( cd "$d" && env "$@" "$BIN" --param run.param --restart-file "$d/restart.bin" \
       > run.log 2>&1 )
  local rc=$?
  local steps=$(grep -c '^Begin Step' "$d/run.log" 2>/dev/null || echo 0)
  printf '  %-10s rc=%d  steps=%-5s  %ds\n' "$name" "$rc" "$steps" "$((SECONDS-t0))"
  if [ $rc -ne 0 ]; then
    echo "  FAIL  $name exited $rc; last lines:"; tail -5 "$d/run.log" | sed 's/^/        /'
    return 1
  fi
  return 0
}

step BUILD-CHECK "the binary is the configuration this gate describes"
run probe GADGET_HIP_PM_CADENCE=1 >/dev/null 2>&1 || true
banner="$WORK/probe/run.log"
if ! grep -q 'PMGRID=64' "$banner" 2>/dev/null; then
  echo "  FAIL  binary is not a PMGRID=64 build (no 'PMGRID=64' in its [PERIODIC] banner)"
  grep -m1 '\[PERIODIC\]' "$banner" 2>/dev/null | sed 's/^/        /'
  exit 2
fi
grep -m1 '\[PERIODIC\]' "$banner" | sed 's/^/  ok    /'
grep -m1 'MAC:' "$banner" | tr -s '\t' ' ' | sed 's/^/  ok    /'

# ---------------------------------------------------------------- the runs
step RUNS "4 x 64^3 to z=0"
run every                                              || fail
run cadence  GADGET_HIP_PM_CADENCE=1                   || fail
run nopm     GADGET_HIP_PM_CADENCE=1 GADGET_HIP_PM_NO_KICK=1   || fail
run kick10   GADGET_HIP_PM_CADENCE=1 GADGET_HIP_PM_KICK_SCALE=10 || fail

# ---------------------------------------------------------------- oracles
# Two references per path, answering different questions -- see tools/check_energy.py.
#
# NOTE the `out=$(...)` form rather than piping into sed: the exit status of a pipeline is the status
# of its LAST command, so `python3 ... | sed ... || fail` can never fail, and the check would report
# PASS/FAIL text while always counting as a pass. That bug was written here first.
check_energy() {   # name, log, reference, tolerance
  local label=$1 log=$2 ref=$3 tol=$4 out rc
  out=$(python3 "$HERE/tools/check_energy.py" "$log" "$ref" --tol "$tol" --quiet); rc=$?
  printf '    %-8s %s\n' "$label" "$(echo "$out" | sed 's/^ *//')"
  [ $rc -eq 0 ] || fail
}

step ORACLE "Ekin vs CPU GADGET-2 (independent code), tol ${GATE64_EKIN_TOL}%"
for p in every cadence; do
  check_energy "$p" "$WORK/$p/run.log" "$HERE/ref/energy_cpu_gadget.txt" "$GATE64_EKIN_TOL"
done

step ENVELOPE "Ekin vs this port's known-good mean, tol ${GATE64_ENVELOPE_TOL}%"
for p in every cadence; do
  check_energy "$p" "$WORK/$p/run.log" "$HERE/ref/ekin_port_$p.txt" "$GATE64_ENVELOPE_TOL"
done

step REGRESSION "structure at z=0 (sigma(delta) = $GATE64_SIGMA +- ${GATE64_SIGMA_TOL}%)"
python3 "$HERE/tools/check_structure.py" "$WORK/every/out/snapshot_000" \
        --expect "$GATE64_SIGMA" --tol "$GATE64_SIGMA_TOL" || fail
python3 "$HERE/tools/check_structure.py" "$WORK/cadence/out/snapshot_000" \
        --expect "$GATE64_SIGMA" --tol "$GATE64_SIGMA_TOL" || fail

# ---------------------------------------------------------------- must-differ (T44 item 5)
# A must-differ check is only meaningful if the expected difference EXCEEDS the run-to-run noise
# floor. This port is not reproducible (T46: 0.42% in Ekin, 0.31% in sigma at z=0), so "the outputs
# differ" is true of any two runs and proves nothing. Both checks below are chosen for effects
# measured at 70%+, i.e. two orders of magnitude above the floor. T44's original suggestion --
# quartering the PM interval, which moved the 7th significant digit -- is NOT usable here for
# exactly this reason.
step MUST-DIFFER "configurations that cannot legitimately agree"
echo "    no long-range force must change structure by >=${GATE64_MIN_DIFF}%"
python3 "$HERE/tools/check_structure.py" "$WORK/nopm/out/snapshot_000" \
        --differ-from "$WORK/every/out/snapshot_000" --min-diff "$GATE64_MIN_DIFF" || fail
echo "    a 10x long-range impulse must change structure by >=${GATE64_MIN_DIFF}%"
python3 "$HERE/tools/check_structure.py" "$WORK/kick10/out/snapshot_000" \
        --differ-from "$WORK/cadence/out/snapshot_000" --min-diff "$GATE64_MIN_DIFF" || fail

# ---------------------------------------------------------------- refusal audit (T44 item 3)
# T44 item 3 says a feature that declines to engage must say so. That only works if you also check
# that it said ANYTHING: a missing line means either the feature never ran or the audit is broken,
# and both are failures. The first version of this check looked only for 'OFF' and so passed
# vacuously when the line was absent.
#
# No '^' anchor: the port block-buffers stdout and leaves stderr unbuffered, so a stderr line lands
# spliced into the middle of a stdout line --
#   "Begin Step 86, Time: 0.0469759, ... Dlo[PM-ENERGY] long-range term ... ACTIVE"
# An anchored pattern silently matched nothing. (main.cpp now line-buffers stdout, which stops the
# splicing; the unanchored pattern stays, because a gate should not depend on that staying true.)
step REFUSAL "features that declined to engage must have said so"
for tag in PM-ENERGY T45-AOLD-PM; do
  line=$(grep -hoE "\[$tag\].*" "$WORK/cadence/run.log" | head -1)
  if [ -z "$line" ]; then
    echo "  FAIL  [$tag] never reported -- feature did not run, or the audit no longer matches it"
    fail
  elif echo "$line" | grep -qiE 'OFF|refus|not '; then
    echo "  FAIL  $line"
    fail
  else
    echo "  ok    $line"
  fi
done

# ---------------------------------------------------------------- self-test
# A gate nobody has seen fail is not a gate. Two checks written for this file today were VACUOUS --
# one took the exit status of `sed` instead of python, the other used an anchored pattern against a
# log whose stderr lands mid-line -- and both reported success while testing nothing.
#
# So reproduce a known-bad configuration and require the checks to reject it. The bug chosen is the
# real one: T42's long-range kick with G never applied, i.e. an impulse 1/43.007106 = 0.02325 of
# correct. Measured: Ekin -94.9%, sigma(delta) 0.954 against 3.681.
step SELF-TEST "the checks must reject a known-bad run (T42's missing G, reproduced)"
if run selftest GADGET_HIP_PM_CADENCE=1 GADGET_HIP_PM_KICK_SCALE=0.0232519712; then
  bad=0
  python3 "$HERE/tools/check_energy.py" "$WORK/selftest/run.log" "$HERE/ref/energy_cpu_gadget.txt" \
          --tol "$GATE64_EKIN_TOL" --quiet >/dev/null 2>&1 && bad=1
  python3 "$HERE/tools/check_structure.py" "$WORK/selftest/out/snapshot_000" \
          --expect "$GATE64_SIGMA" --tol "$GATE64_SIGMA_TOL" >/dev/null 2>&1 && bad=1
  if [ $bad -eq 1 ]; then
    echo "  FAIL  a check PASSED a run with the long-range force 43x too small."
    echo "        The gate is broken, not the code under test. Do not trust a PASS from it."
    fail
  else
    echo "  ok    energy and structure checks both reject it"
  fi
else
  fail
fi

# ---------------------------------------------------------------- verdict
echo
if [ $FAILED -eq 0 ]; then
  echo "GATE64: PASS  (${SECONDS}s)"
else
  echo "GATE64: FAIL  ($FAILED check(s) failed, ${SECONDS}s)"
  echo "  run directory kept for inspection: $WORK"; KEEP=1
fi
[ $KEEP -eq 1 ] || rm -rf "$WORK"
exit $([ $FAILED -eq 0 ] && echo 0 || echo 1)
