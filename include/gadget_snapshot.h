#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Phase 5 ticket 02 (PLAN.md): Gadget-2 snapshot I/O, format 1 and 2 (NOT HDF5 -- deferred past
// Phase 6). Output (gadget_snapshot_write) is single-file only (NumFilesPerSnapshot=1, no
// `.0`/`.filenr` suffix -- this single-process port never needs to split its own output).
// Input (gadget_snapshot_read) DOES support Gadget-2's real multi-file convention (Phase 5
// ticket 08 (PLAN.md): both this project's own Phase 0 IC and a real zoom IC need it -- the
// latter genuinely, not just as an MPI-era artifact, since each resolution level of a zoom IC is
// its own file) -- see gadget_snapshot.cpp's own multi-file assembly comment for the details.
// Only the block set a gravity-only DM run actually produces with default Makefile flags: HEAD,
// POS, VEL, ID, [MASS if any type lacks a fixed per-type mass]. OUTPUTPOTENTIAL/
// OUTPUTACCELERATION/OUTPUTTIMESTEP's optional POT/ACCE/TSTP blocks are a documented,
// non-blocking follow-on (PLAN.md Phase 5.2), not implemented here.

// Mirrors Gadget-2's on-disk `struct io_header` exactly (allvars.h:609-632), field-for-field, at
// its exact 256-byte layout (PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.1). Serialized/deserialized at
// fixed byte offsets by gadget_snapshot.cpp rather than relying on this struct's own C++ padding
// to happen to match Gadget-2's C layout -- portable regardless of compiler packing choices.
struct GadgetSnapshotHeader
{
  int          npart[6]              = {0, 0, 0, 0, 0, 0};
  double       mass[6]                = {0, 0, 0, 0, 0, 0};
  double       time                   = 0.0;
  double       redshift               = 0.0;
  int          flag_sfr               = 0;
  int          flag_feedback          = 0;
  unsigned int npartTotal[6]          = {0, 0, 0, 0, 0, 0};
  int          flag_cooling           = 0;
  int          num_files              = 1;
  double       BoxSize                = 0.0;
  double       Omega0                 = 0.0;
  double       OmegaLambda            = 0.0;
  double       HubbleParam            = 0.0;
  int          flag_stellarage        = 0;
  int          flag_metals            = 0;
  unsigned int npartTotalHighWord[6]  = {0, 0, 0, 0, 0, 0};
  int          flag_entropy_instead_u = 0;
};

// In-memory particle data, always grouped by type in ascending order 0..5 (matching Gadget-2's
// own on-disk ordering, PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.5) -- gadget_snapshot_write() takes
// care of sorting an arbitrarily-ordered input into this shape; gadget_snapshot_read() always
// produces data already in this order (the file itself is grouped this way).
struct GadgetParticleData
{
  std::vector<float>    pos;   // 3*N: x0,y0,z0, x1,y1,z1, ...
  std::vector<float>    vel;   // 3*N, same layout. See velocity comoving-convention note below.
  std::vector<uint64_t> id;    // N -- widened to 64-bit in memory regardless of on-disk LONGIDS
  std::vector<float>    mass;  // N -- always fully populated on read (expanded from mass[type]
                                // for types using a fixed table mass), consumed as-is on write
  std::vector<int>      type;  // N -- Gadget-2 particle type, 0..5 (0=gas,1=halo,2=disk,3=bulge,
                                // 4=stars,5=bndry). This port has no SPH, so type 0 should not
                                // appear in practice, but nothing here enforces that.
};

// Writes a DM-only-shaped Gadget-2 snapshot: HEAD, POS, VEL, ID, and MASS only for particle types
// that don't get a fixed per-type mass. `header.npart`/`npartTotal`/`npartTotalHighWord`/
// `num_files` are computed here from `data.type` and OVERWRITTEN regardless of what the caller
// put in them (single source of truth: the actual particle data, not a caller-maintained count)
// -- every other header field is taken from `header` as given. `header.mass[t]` is likewise
// computed here, not taken from the caller: for each type t present, if every particle of that
// type shares the exact same mass value, `mass[t]` is set to it and that type's masses are
// omitted from the MASS block entirely (matching Gadget-2's own convention exactly,
// PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.4); otherwise `mass[t]=0` and per-particle masses are
// written for every particle of that type.
//
// `snapFormat` must be 1 or 2. `longIds` selects a 4-byte (unsigned int) or 8-byte
// (unsigned long long) on-disk ID block, matching Gadget-2's own LONGIDS compile flag -- a
// build-time choice in this port too (GADGET_HIP_LONGIDS CMake option), not autodetected.
//
// `comovingIntegrationOn` and `header.time` (the scale factor `a`) together select the on-disk
// VEL convention: when on, every velocity component is DIVIDED by `sqrt(a)*a` before writing.
// Confirmed by direct source citation (Phase 5 ticket 08 (PLAN.md), and a real self-caught
// correction within this same ticket -- an earlier pass at this citation stopped reading `io.c`
// one line too early and concluded there was no scaling at all): `io.c`'s `IO_VEL` case
// (`fill_write_buffer()`, ~150-241) computes `P[pindex].Vel[k] + kick_terms` and THEN multiplies
// by `sqrt(a3inv)` where `a3inv = 1/a^3` under comoving integration -- i.e. divides by
// `sqrt(a)*a` overall. The effect: a regular Gadget-2 snapshot's on-disk VEL ends up in the SAME
// peculiar-velocity convention as a fresh initial-conditions file, not the tiny internal
// "canonical momentum" magnitude -- this function's divide is the exact mutual inverse of
// gadget_snapshot_read's IC-load multiply below.
//
// Returns true on success; on failure, returns false and `error` holds a human-readable message.
bool gadget_snapshot_write(const std::string &path, int snapFormat, bool longIds,
                            bool comovingIntegrationOn, GadgetSnapshotHeader header,
                            const GadgetParticleData &data, std::string &error);

// Reads a Gadget-2 snapshot written in format `icFormat` (1 or 2 -- like Gadget-2 itself, the
// format is NOT autodetected from the file; the caller must already know it, matching
// `ICFormat`/`SnapFormat`'s role as a `.param` value, PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.2).
// `longIds` must match how the file was written (again a build-time fact in real Gadget-2, not a
// property recoverable from the file itself). Both single-file and Gadget-2's real multi-file
// convention (`<path>.0`, `<path>.1`, ...; detected exactly like `read_ic.c`'s own `find_files()`,
// by whether `<path>.0` opens -- not by trusting any header's `num_files` field up front) are
// supported (Phase 5 ticket 08 (PLAN.md)); multi-file input is bucketed by particle type and
// reassembled into one ascending-type-order in-memory set, since a real zoom IC's files need not
// themselves be in ascending type order (see gadget_snapshot.cpp's own assembly comment).
//
// On success, `header` and `data` are fully populated -- `data.mass` is expanded to one entry per
// particle regardless of whether that particle's type used a fixed table mass or an explicit
// MASS-block entry, and `data.vel` has already had Gadget-2's own real initial-conditions velocity
// transform applied (multiplying by `sqrt(a)*a`, using the header's own `time` field as `a`,
// exactly matching `init.c:80`) when `comovingIntegrationOn` is set -- the exact mutual inverse of
// gadget_snapshot_write's own divide above, so this port's own write-then-read round-trips exactly.
//
// Returns true on success; on failure, returns false and `error` holds a human-readable message.
bool gadget_snapshot_read(const std::string &path, int icFormat, bool longIds,
                           bool comovingIntegrationOn, GadgetSnapshotHeader &header,
                           GadgetParticleData &data, std::string &error);

// Self-contained round-trip correctness test (Phase 5 ticket 02, `--snapshot-test` CLI flag,
// same one-shot-diagnostic idiom as Phase 4's `--pm-test-*` flags). Synthesizes a small
// multi-type particle set -- one type where every particle shares the exact same mass (exercises
// the Massarr fixed-mass path, MASS block should omit that type) and one type with distinct
// per-particle masses (exercises the explicit MASS-block path) -- writes it to `tmpPath` in the
// given format, reads it back, and checks every field round-trips correctly. Internally runs the
// check twice, with `ComovingIntegrationOn` off and on (a synthetic mid-range scale factor), to
// also exercise the velocity (de)conversion in both directions. Returns true iff every check
// passes; `report` always gets a human-readable pass/fail summary regardless of the return value.
bool gadget_snapshot_selftest(const std::string &tmpPath, int snapFormat, bool longIds,
                               std::string &report);
