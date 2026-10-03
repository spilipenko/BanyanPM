#!/bin/bash
# T51: the periodic-centre unit test. Needs no GPU and no HIP -- pm_zoom_region.cpp is deliberately
# HIP-free so this compiles with plain g++, which is what makes the wrap cases cheap to test.
#   tests/zoom/run.sh
set -u
cd "$(dirname "$0")/../.."
g++ -O2 -DGADGET_HIP_HIGHRES -I include -o /tmp/test_periodic_center \
    tests/zoom/test_periodic_center.cpp src/pm_zoom_region.cpp || exit 1
/tmp/test_periodic_center
