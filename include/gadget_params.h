#pragma once
#include <string>
#include <vector>

// Phase 5 ticket 01 (PLAN.md): Gadget-2 `.param` file parser + central config struct.
//
// This struct is this port's rough equivalent of Gadget-2's global `All` (struct
// global_data_all_processes, allvars.h) -- the single source of truth for every value a `.param`
// file supplies. Field names deliberately match Gadget-2's own `All.<field>` names exactly (see
// PHASE5_PARAMS_SNAPSHOT_SPEC.md, Sec 1.5) rather than this port's own naming conventions, so the
// mapping to the source-of-truth spec/citations stays direct and greppable.
//
// Scope, per PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 0: `read_parameter_file()` (begrun.c:280-752)
// registers all 65 tags with ZERO compile-time (#ifdef) gating -- every tag is required in every
// real `.param` file regardless of PERIODIC/PMGRID/etc, confirmed against both shipped example
// files. So this struct holds all 65 fields unconditionally; which ones actually drive behavior
// in this DM-only, SPH-less, MPI-less port is a matter for the *consumers* of this struct (later
// Phase 5 tickets), not for the parser itself -- SPH-only/MPI-only tags are still parsed and
// stored here, never rejected as unknown, matching real-world `.param` files that list all 65
// keys unconditionally (spec Sec 1.6).
struct GadgetParams
{
  // --- File paths / logs (STRING tags 1-10) ---
  std::string InitCondFile;
  std::string OutputDir;
  std::string SnapshotFileBase;
  std::string EnergyFile;
  std::string CpuFile;
  std::string InfoFile;
  std::string TimingsFile;
  std::string RestartFile;
  std::string ResubmitCommand;
  std::string OutputListFilename;

  // --- Output-list / cosmology / box (tags 11-18) ---
  int    OutputListOn = 0;
  double Omega0 = 0.0;
  double OmegaBaryon = 0.0;          // SPH/cooling-only in this source tree; parsed, never consumed
  double OmegaLambda = 0.0;
  double HubbleParam = 0.0;          // "little h" -- header/output bookkeeping only, not used to
                                      // rescale internal G/Hubble (spec Sec 1.5, tag 15)
  double BoxSize = 0.0;
  int    PeriodicBoundariesOn = 0;
  double TimeOfFirstSnapshot = 0.0;

  // --- Timing (tags 19-23) ---
  double CpuTimeBetRestartFile = 0.0;
  double TimeBetStatistics = 0.0;
  double TimeBegin = 0.0;
  double TimeMax = 0.0;
  double TimeBetSnapshot = 0.0;

  // --- Unit system (tags 24-26) ---
  double UnitVelocity_in_cm_per_s = 0.0;
  double UnitLength_in_cm = 0.0;
  double UnitMass_in_g = 0.0;

  // --- Domain/timestep/gravity accuracy (tags 27-30) ---
  double TreeDomainUpdateFrequency = 0.0; // MPI-domain-decomposition-only, no-op single-process
  double ErrTolIntAccuracy = 0.0;
  double ErrTolTheta = 0.0;
  double ErrTolForceAcc = 0.0;

  // --- SPH-only (tags 31, 35-38, 64-65) -- parsed and stored, never consumed by this port ---
  double MinGasHsmlFractional = 0.0;
  double ArtBulkViscConst = 0.0;
  double CourantFac = 0.0;
  double DesNumNgb = 0.0;
  double MaxNumNgbDeviation = 0.0;
  double InitGasTemp = 0.0;
  double MinGasTemp = 0.0;

  // --- Timestep bounds (tags 32-34) ---
  double MaxSizeTimestep = 0.0;
  double MinSizeTimestep = 0.0;
  double MaxRMSDisplacementFac = 0.0;

  // --- Mode flags / formats (tags 39-46) ---
  int ComovingIntegrationOn = 0;
  int ICFormat = 1;
  int SnapFormat = 1;
  int NumFilesPerSnapshot = 1;
  int NumFilesWrittenInParallel = 1;   // MPI-only; this port requires exactly 1 (validated)
  int ResubmitOn = 0;
  int TypeOfTimestepCriterion = 0;     // only 0 is ever valid, in stock Gadget-2 too (validated)
  int TypeOfOpeningCriterion = 0;

  double TimeLimitCPU = 0.0;

  // --- Per-type softening (tags 48-59) ---
  double SofteningGas = 0.0;
  double SofteningHalo = 0.0;
  double SofteningDisk = 0.0;
  double SofteningBulge = 0.0;
  double SofteningStars = 0.0;
  double SofteningBndry = 0.0;
  double SofteningGasMaxPhys = 0.0;
  double SofteningHaloMaxPhys = 0.0;
  double SofteningDiskMaxPhys = 0.0;
  double SofteningBulgeMaxPhys = 0.0;
  double SofteningStarsMaxPhys = 0.0;
  double SofteningBndryMaxPhys = 0.0;

  // --- Memory sizing (tags 60-62) ---
  int    BufferSize = 0;
  double PartAllocFactor = 0.0;
  double TreeAllocFactor = 0.0;

  // --- Gravity constant override (tag 63) ---
  double GravityConstantInternal = 0.0;

  // --- Derived by deriveUnits(), NOT read directly from the .param file --- (set_units(),
  // begrun.c:149-191, PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 1.3). Only the subset actually needed by
  // a DM-only port -- UnitPressure_in_cgs/UnitCoolingRate_in_cgs/UnitEnergy_in_cgs/MinEgySpec are
  // SPH-only per the spec and deliberately not ported.
  double UnitTime_in_s = 0.0;
  double UnitTime_in_Megayears = 0.0;
  double UnitDensity_in_cgs = 0.0;
  double G = 0.0;        // All.G
  double Hubble = 0.0;   // All.Hubble = HUBBLE * UnitTime_in_s

  // Derives G/Hubble/UnitTime_in_s/UnitTime_in_Megayears/UnitDensity_in_cgs from the raw unit
  // triplet + GravityConstantInternal, exactly matching set_units() (begrun.c:149-191). Call once
  // after a successful parse().
  void deriveUnits();
};

// Parses a Gadget-2 `.param` file into `out`, matching read_parameter_file()'s exact semantics
// (begrun.c:280-752, PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 1.1): whitespace-tokenized `tag value`
// pairs, a line is a comment only if its first token starts with '%', each of the 65 known tags
// must appear exactly once (duplicate or unrecognized tag is a hard error), and a tag missing by
// EOF is a hard error -- there is no default-value mechanism anywhere in this parser, matching the
// source exactly.
//
// On success, returns true and `out` holds every parsed value (deriveUnits() is called
// automatically before returning). On failure, returns false and `errors` holds one human-readable
// message per problem found (multiple problems are collected, not just the first, to make a
// malformed file easier to fix in one pass) -- `out` is not guaranteed fully populated in this
// case.
//
// Also runs the compile-time-relevant post-parse validation gates from begrun.c Sec 1.2 that are
// still meaningful for this single-process, TypeOfTimestepCriterion==0-only port:
// NumFilesWrittenInParallel must be exactly 1 (this port has no MPI, so "must be between 1 and
// NTask" collapses to this), TypeOfTimestepCriterion must be 0, and PeriodicBoundariesOn must
// match `builtWithPeriodic` (this build's compile-time periodicity mode, passed in by the caller
// since this header cannot see the caller's PERIODIC macro state).
bool gadget_params_parse(const std::string &filename, bool builtWithPeriodic, GadgetParams &out,
                          std::vector<std::string> &errors);
