cmake_minimum_required(VERSION 3.21)

# HIP-native fork of bonsai/runtime, stripped to a headless TreePM core
# (no renderer/OpenGL/dust/galactics). See AMD_PORT_PLAN.md.

# ROCm's own clang++ is used as the host C++ compiler, not stock GCC: hip_runtime_api.h on this
# install hits a real compile error under g++ 16 (hipHostMallocDefault "not declared" at its use
# site despite being #define'd earlier in the same header) that doesn't reproduce under clang.
set(CMAKE_CXX_COMPILER /usr/lib64/rocm/llvm/bin/clang++)

project(gadget-hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
# C++17, not the original bonsai's C++14: rocThrust/rocPRIM's headers on this ROCm version use
# std::is_convertible_v / std::disjunction (C++17), unconditionally, via my_cuda_rt.h's
# thrust/device_ptr.h include.

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx1151)
endif()

find_package(hip REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# hipCUB/rocPRIM are header-only here and already live under /usr/include (default search
# path) -- deliberately NOT using find_package(hipcub)/find_package(rocprim) + their imported
# targets: those pull in hip::device, whose INTERFACE_COMPILE_OPTIONS force "-x hip" onto every
# CXX-language source in the consuming target (fine for a single-language hipcc-as-CXX build,
# but wrong here since we use CMake's native dual CXX/HIP language mode) -- it broke g++
# compilation of the plain-CXX host files (e.g. build.cpp) with "invalid language hip".

# USE_MPI removed: multi-rank does not work in this port and never did. There is not a single MPI
# collective on any Gadget-specific quantity (no reduction of the timestep ladder, the sync point,
# or the energy statistics -- all 11 collectives are inherited Bonsai load-balance/IO plumbing);
# the PM solver explicitly refuses nProcs>1 and skips long-range gravity entirely; and an actual
# 2-rank run aborts in the first tree build ("tree has become too deep") before any physics runs.
# Dropping the define compiles out every USE_MPI block, including the whole body of parallel.cpp,
# which is what frees bodies_h (its remaining holder was the nProcs>1 particle exchange).
# Re-add -DUSE_MPI to restore the code; making it actually WORK is a separate project.
add_definitions(-D_MPIMT -DUSE_CUB -D__HIP_PLATFORM_AMD__=1)
# __HIP_PLATFORM_AMD__ is needed globally, not just on GPU-touching files: hipify-perl -inplace
# unconditionally prepends "#include hip/hip_runtime.h" to every file it processes, including
# ones (e.g. anyoption.h) that never used CUDA/HIP at all.

# Phase 3 (AMD_PORT_PLAN.md): Springel(2005)/Gadget-2 relative acceleration MAC as a build-time
# alternative to the improved-BH opening-angle MAC (default). A CMake option rather than a header
# #define toggle so both variants can be configured into separate out-of-source build dirs
# (e.g. build-release/ vs build-springel/) and A/B compared directly, without editing source
# between builds.
option(MAC_SPRINGEL "Use Springel(2005) relative-acceleration MAC instead of improved-BH" OFF)
if(MAC_SPRINGEL)
  add_definitions(-D_MAC_SPRINGEL_)
endif()

# T28 (C-A-04/C-C-08/C-B-15): Gadget-2's mixed-softening machinery, and like Gadget-2 it is a
# compile-time option (its own UNEQUALSOFTENINGS, forcetree.c:427/465/513/1654). A run whose
# particle types all share one softening -- every uniform cosmological box -- pays nothing for it:
# the per-node (maxSoft, mixed) buffer is not allocated, the reduction is not compiled, and the
# walk carries no extra load or branch. Zoom runs with per-type softenings need it and accept the
# cost. Guarded at startup: a build WITHOUT this option refuses a parameter file that actually
# specifies more than one distinct softening (see main.cpp), so the saving can never turn into
# silently wrong physics.
option(UNEQUAL_SOFTENINGS "Enable Gadget-2's mixed-softening node summary + forced descent" OFF)
if(UNEQUAL_SOFTENINGS)
  add_definitions(-DUNEQUALSOFTENINGS)
endif()

# Phase 3 NCRIT=32-vs-64 experiment: group size (i-particle group / GOTHIC pseudo-particle size),
# same CMake-option convention as MAC_SPRINGEL above -- node_specs.h defaults to 64 if NCRIT isn't
# defined on the command line.
set(GADGET_HIP_NCRIT "" CACHE STRING "Override NCRIT (group size); empty = node_specs.h default (32)")
if(GADGET_HIP_NCRIT)
  add_definitions(-DNCRIT=${GADGET_HIP_NCRIT})
endif()

# T9 (wayfinder map, launch-config-dependence ticket): NTHREAD (node_specs.h) controls how many
# threads/warps cooperate per block in the tree-walk kernel (dev_approximate_gravity_warp_new.cu's
# treewalk<nWarp2,...> dispatch). Same override convention as GADGET_HIP_NCRIT above -- node_specs.h
# defaults to 128 if NTHREAD isn't defined on the command line.
set(GADGET_HIP_NTHREAD "" CACHE STRING "Override NTHREAD (warp-cooperative launch width); empty = node_specs.h default (128)")
if(GADGET_HIP_NTHREAD)
  add_definitions(-DNTHREAD=${GADGET_HIP_NTHREAD})
endif()

# De-grouping experiment (2026-08-30, PLAN.md Close-encounter root-cause audit / LOG.md Sec 48-51):
# NLEAF (octree leaf size, node_specs.h) is independent of NCRIT in the actual kernels but the
# header enforces NCRIT>=NLEAF, so testing small NCRIT (down to a true per-particle NCRIT=1) needs
# NLEAF lowered to match -- same override convention as GADGET_HIP_NCRIT.
set(GADGET_HIP_NLEAF "" CACHE STRING "Override NLEAF (octree leaf size); empty = node_specs.h default (16)")
if(GADGET_HIP_NLEAF)
  add_definitions(-DNLEAF=${GADGET_HIP_NLEAF})
endif()

# Found 2026-08-28: Bonsai's own inherited kernel had quadrupole moments hardcoded ON (a bare
# `#if 1`, not a build flag) -- meaning all of Phase 3's validation ran with it active, unnoticed.
# Gadget-2 is monopole-only by deliberate design (GADGET2_NOTES.md), and the user decided not to
# use quadrupole -- default OFF here, matching Gadget-2, with the prior behavior kept reachable
# via this flag rather than deleted (same convention as MAC_SPRINGEL/NCRIT above).
# C-B-19 probe: disable the Rcut early-rejection so its conservativeness can be A/B tested by
# bit-comparison of the forces. Diagnostic only -- it removes an optimisation, so the B side is
# slower and is never a production configuration.
option(GADGET_HIP_CB19_NO_RCUT_REJECT "C-B-19 probe: disable Rcut node rejection" OFF)
if(GADGET_HIP_CB19_NO_RCUT_REJECT)
  add_definitions(-DGADGET_HIP_CB19_NO_RCUT_REJECT)
endif()

option(GADGET_HIP_QUADRUPOLE "Enable Bonsai's quadrupole moment correction (off = Gadget-2's own monopole-only behavior)" OFF)
if(GADGET_HIP_QUADRUPOLE)
  add_definitions(-DGADGET_HIP_QUADRUPOLE)
endif()

# Ticket 20 (tickets/T20-test-summation-order-sensitivity.md): test-only, opt-in double-precision
# force accumulator for the warp-cooperative tree walk (acc_i[k] in
# dev_approximate_gravity_warp_new.cu), to test whether float32 summation-order/precision in that
# accumulation explains Ticket 17's secular non-cosmological energy drift. Default OFF (identical
# to every other build) -- only ever set ON in a dedicated test build (build-t20-doubleforce),
# never in a build another ticket depends on. Same convention as GADGET_HIP_QUADRUPOLE above.
option(GADGET_HIP_T20_DOUBLE_FORCE_ACCUM "Ticket 20 test-only: accumulate acc_i in double4 instead of float4 in the warp-cooperative tree walk (off = original float4 behavior)" OFF)
if(GADGET_HIP_T20_DOUBLE_FORCE_ACCUM)
  add_definitions(-DGADGET_HIP_T20_DOUBLE_FORCE_ACCUM)
endif()

# Phase 5 ticket 02 (PLAN.md): Gadget-2 snapshot I/O. LONGIDS selects an 8-byte (vs. default
# 4-byte) on-disk particle-ID block, matching Gadget-2's own LONGIDS compile flag exactly -- a
# build-time choice there too, not a `.param` value (PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 1.4/2.4).
# Same CMake-option convention as GADGET_HIP_QUADRUPOLE above.
option(GADGET_HIP_LONGIDS "Use 8-byte particle IDs in Gadget-2 snapshot I/O (off = Gadget-2's own default 4-byte IDs)" OFF)
if(GADGET_HIP_LONGIDS)
  add_definitions(-DGADGET_HIP_LONGIDS)
endif()

# Phase 4 (PLAN.md): PM long-range gravity solver. PMGRID: PM mesh size (mesh cells per side), a
# compile-time constant in Gadget-2 too (its PMGRID is a Makefile #define, not a runtime .param
# value -- GADGET2_NOTES.md). Same CMake-option convention as NCRIT/MAC_SPRINGEL above. Empty
# means "PM not compiled in" -- deliberately no silent default grid size, since a wrong-but-
# plausible default here would be a much more consequential silent bug than NCRIT's tuning-only
# default. rocFFT is only required when PM is actually enabled, matching every other feature in
# this file being purely additive/opt-in -- a non-PM build dir shouldn't need rocFFT installed.
set(GADGET_HIP_PMGRID "" CACHE STRING "PM mesh size (cells per side); empty = PM support not compiled in")
set(PM_EXTRA_CCFILES "")
set(PM_EXTRA_CUFILES "")
if(GADGET_HIP_PMGRID)
  add_definitions(-DPMGRID=${GADGET_HIP_PMGRID})
  # rocFFT's own CMake package config is not discoverable via bare find_package() on this native
  # ROCm 7.1.1 install -- its config lives under /usr/lib64/cmake/, a subtree cmake doesn't walk
  # into automatically here (see PLAN.md §5 for the full investigation, which hit the identical
  # issue for hipcub/rocprim). Set the prefix explicitly rather than relying on the caller's
  # environment, so a plain `cmake -S . -B build -DGADGET_HIP_PMGRID=...` keeps working the way
  # every other build dir in this project already does.
  list(APPEND CMAKE_PREFIX_PATH "/usr/lib64" "/usr/share/cmake")
  find_package(rocfft REQUIRED)
  # pm_fft.cpp (rocFFT plan management) and pm_greens.cu (periodic Green's-function multiply) are
  # only meaningful as part of the PM pipeline (unlike pm_cic.cu, which is a generic, reusable
  # primitive with no PM-specific compile dependency) -- gated together with the rocFFT dependency
  # itself, not added unconditionally.
  list(APPEND PM_EXTRA_CCFILES src/pm_fft.cpp src/pm_solve.cpp src/pm_solve_isolated.cpp)
  list(APPEND PM_EXTRA_CUFILES CUDAkernels/pm_greens.cu CUDAkernels/pm_isolated.cu)
endif()

# Tree-side periodic minimum-image walk + TreePM short-range/long-range split (Gadget-2's own
# PERIODIC macro name, forcetree.c). Independent of PMGRID in Gadget-2 itself (PLAN.md's Phase 4
# architecture note: periodicity and "use PM" are orthogonal there), but THIS port only
# implements the TreePM combination (minimum-image tree + PM handling the long-range remainder) --
# PERIODIC without PMGRID would silently drop all long-range force with nothing computing it, so
# that combination is refused at configure time rather than silently producing wrong physics.
option(PERIODIC "Enable periodic minimum-image tree walk + TreePM short-range/long-range split" OFF)
if(PERIODIC)
  if(NOT GADGET_HIP_PMGRID)
    message(FATAL_ERROR "PERIODIC requires GADGET_HIP_PMGRID: this port's periodic tree walk "
      "only implements the TreePM combination (minimum-image tree + PM for the long-range "
      "remainder) -- PERIODIC alone would silently drop long-range force entirely.")
  endif()
  add_definitions(-DPERIODIC)
endif()

# Phase 5 ticket 06 (PLAN.md): PLACEHIGHRESREGION zoom dual-grid PM. Requires GADGET_HIP_PMGRID
# (zoom is a PM refinement, meaningless without PM at all) -- same "refused at configure time
# rather than silently producing wrong physics" reasoning PERIODIC already applies above. Refused
# together with GADGET_HIP_QUADRUPOLE too: the tree-side per-type Rcut/Asmth split (Ticket 07)
# only extends the monopole add_acc() path, and there is no Gadget-2 reference for a
# quadrupole+zoom combination anyway (same reasoning already applied to PERIODIC+QUADRUPOLE).
# Works with or without PERIODIC (PHASE5_ZOOM_COMOVING_SPEC.md Sec 1.2 -- both are real Gadget-2
# modes), so no dependency on that option specifically.
option(GADGET_HIP_HIGHRES "Enable PLACEHIGHRESREGION zoom dual-grid PM (fine grid refinement)" OFF)
if(GADGET_HIP_HIGHRES)
  if(NOT GADGET_HIP_PMGRID)
    message(FATAL_ERROR "GADGET_HIP_HIGHRES requires GADGET_HIP_PMGRID: zoom is a refinement of "
      "the PM long-range solve, meaningless without PM compiled in at all.")
  endif()
  if(GADGET_HIP_QUADRUPOLE)
    message(FATAL_ERROR "GADGET_HIP_HIGHRES is refused together with GADGET_HIP_QUADRUPOLE: no "
      "Gadget-2 reference exists for this combination (same reasoning as PERIODIC+QUADRUPOLE).")
  endif()
  add_definitions(-DGADGET_HIP_HIGHRES)
  list(APPEND PM_EXTRA_CCFILES src/pm_zoom_region.cpp src/pm_solve_zoom.cpp)
  list(APPEND PM_EXTRA_CUFILES CUDAkernels/pm_zoom.cu)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CORE_CCFILES
  src/build.cpp
  src/compute_properties.cpp
  src/gpu_iterate.cpp
  src/libraryInterface.cpp
  src/load_kernels.cpp
  src/anyoption.cpp
  src/octree.cpp
  src/parallel.cpp
  src/sort_bodies_gpu.cpp
  src/log.cpp
  src/hostConstruction.cpp
  src/tipsyIO.cpp
  src/gadget_params.cpp
  src/gadget_snapshot.cpp
  src/gadget_softening.cpp
  src/gadget_cosmology.cpp
  src/gadget_driftfac.cpp
  ${PM_EXTRA_CCFILES}
  )
# main.cpp is deliberately excluded from CORE_CCFILES and built only into the gadget_hip
# executable below (not into gadget_hip_core too) -- an earlier version compiled the full file
# set into both a STATIC lib and an executable, and a compile-flag (-march=native, needed for
# parallel.cpp's SSE builtins) applied to only one of the two duplicate compilations, which is
# exactly the kind of drift that caused a real build failure. One compilation, one set of flags.

set(CORE_CUFILES
  CUDAkernels/build_tree.cu
  CUDAkernels/compute_propertiesD.cu
  CUDAkernels/parallel.cu
  CUDAkernels/sortKernels.cu
  CUDAkernels/scanKernels.cu
  CUDAkernels/timestep.cu
  CUDAkernels/depthSort.cu
  CUDAkernels/dev_direct_gravity.cu
  CUDAkernels/dev_approximate_gravity_warp_new.cu
  CUDAkernels/pm_cic.cu
  CUDAkernels/pm_force.cu
  ${PM_EXTRA_CUFILES}
  )
# pm_cic.cu (CIC mass assignment + force interpolation) and pm_force.cu (finite-difference force
# extraction) take grid size/pitch as runtime arguments, not compile-time constants, so they build
# unconditionally regardless of GADGET_HIP_PMGRID -- only the rocFFT-calling PM orchestration code
# (PM_EXTRA_CCFILES/PM_EXTRA_CUFILES above) needs PM compiled in.

set_source_files_properties(${CORE_CUFILES} PROPERTIES LANGUAGE HIP)

add_library(gadget_hip_core STATIC
  ${CORE_CCFILES}
  ${CORE_CUFILES}
  )

target_include_directories(gadget_hip_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gadget_hip_core PUBLIC hip::host MPI::MPI_CXX OpenMP::OpenMP_CXX)
if(GADGET_HIP_PMGRID)
  target_link_libraries(gadget_hip_core PUBLIC roc::rocfft)
endif()
target_compile_options(gadget_hip_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fPIE -Wno-main>)
# -march=native/-msse4 apply ONLY to parallel.cpp (the one file using SSE intrinsics for its
# LET/domain SIMD helpers) -- NOT blanket across the whole target. Blanket -march=native was
# briefly tried and caused octree::iterate_once to SIGSEGV on entry (deterministic, every run):
# the extra ISA widens what the compiler treats as "natural" alignment for spills/locals beyond
# the x86_64 ABI's guaranteed 16-byte stack alignment, and that assumption doesn't hold on an
# OpenMP-worker-thread stack -- so any file that doesn't actually need the wider ISA should not
# opt into it.
set_source_files_properties(src/parallel.cpp PROPERTIES COMPILE_OPTIONS "-msse4;-march=native")

add_executable(gadget_hip src/main.cpp)
target_include_directories(gadget_hip PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(gadget_hip PRIVATE -fPIE -Wno-main)
target_link_libraries(gadget_hip PRIVATE gadget_hip_core hip::host MPI::MPI_CXX OpenMP::OpenMP_CXX -lrt)

add_executable(bonsai_clrshm src/bonsai_clrshm.cpp)
target_link_libraries(bonsai_clrshm PRIVATE MPI::MPI_CXX -lrt)

add_executable(bonsai_io src/bonsai_io.cpp src/anyoption.cpp)
target_include_directories(bonsai_io PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bonsai_io PRIVATE MPI::MPI_CXX -lrt)

add_executable(bonsai_driver src/driver.cpp)
target_link_libraries(bonsai_driver PRIVATE MPI::MPI_CXX -ldl -lrt)
