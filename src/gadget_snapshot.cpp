#include "gadget_snapshot.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

// Phase 5 ticket 02 (PLAN.md) -- see gadget_snapshot.h for design rationale. Implementation
// structured to mirror io.c/read_ic.c's own record-marker/block-descriptor protocol
// (PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.2) as directly as possible, so it stays easy to audit
// against the source.

namespace {

// Fixed byte offsets/sizes for the 256-byte header, per PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.1
// (allvars.h:609-632). Serializing/deserializing at these exact offsets, rather than reading/
// writing the GadgetSnapshotHeader struct's own raw bytes, sidesteps any risk of this compiler's
// struct packing not matching Gadget-2's C layout bit-for-bit.
constexpr size_t HEADER_BYTES = 256;

void serializeHeader(const GadgetSnapshotHeader &h, uint8_t buf[HEADER_BYTES])
{
  std::memset(buf, 0, HEADER_BYTES);
  size_t o = 0;
  std::memcpy(buf + o, h.npart, 6 * sizeof(int)); o += 24;
  std::memcpy(buf + o, h.mass, 6 * sizeof(double)); o += 48;
  std::memcpy(buf + o, &h.time, sizeof(double)); o += 8;
  std::memcpy(buf + o, &h.redshift, sizeof(double)); o += 8;
  std::memcpy(buf + o, &h.flag_sfr, sizeof(int)); o += 4;
  std::memcpy(buf + o, &h.flag_feedback, sizeof(int)); o += 4;
  std::memcpy(buf + o, h.npartTotal, 6 * sizeof(unsigned int)); o += 24;
  std::memcpy(buf + o, &h.flag_cooling, sizeof(int)); o += 4;
  std::memcpy(buf + o, &h.num_files, sizeof(int)); o += 4;
  std::memcpy(buf + o, &h.BoxSize, sizeof(double)); o += 8;
  std::memcpy(buf + o, &h.Omega0, sizeof(double)); o += 8;
  std::memcpy(buf + o, &h.OmegaLambda, sizeof(double)); o += 8;
  std::memcpy(buf + o, &h.HubbleParam, sizeof(double)); o += 8;
  std::memcpy(buf + o, &h.flag_stellarage, sizeof(int)); o += 4;
  std::memcpy(buf + o, &h.flag_metals, sizeof(int)); o += 4;
  std::memcpy(buf + o, h.npartTotalHighWord, 6 * sizeof(unsigned int)); o += 24;
  std::memcpy(buf + o, &h.flag_entropy_instead_u, sizeof(int)); o += 4;
  // remaining 60 bytes: fill[], left zeroed by the memset above.
  (void)o;
}

void deserializeHeader(const uint8_t buf[HEADER_BYTES], GadgetSnapshotHeader &h)
{
  size_t o = 0;
  std::memcpy(h.npart, buf + o, 6 * sizeof(int)); o += 24;
  std::memcpy(h.mass, buf + o, 6 * sizeof(double)); o += 48;
  std::memcpy(&h.time, buf + o, sizeof(double)); o += 8;
  std::memcpy(&h.redshift, buf + o, sizeof(double)); o += 8;
  std::memcpy(&h.flag_sfr, buf + o, sizeof(int)); o += 4;
  std::memcpy(&h.flag_feedback, buf + o, sizeof(int)); o += 4;
  std::memcpy(h.npartTotal, buf + o, 6 * sizeof(unsigned int)); o += 24;
  std::memcpy(&h.flag_cooling, buf + o, sizeof(int)); o += 4;
  std::memcpy(&h.num_files, buf + o, sizeof(int)); o += 4;
  std::memcpy(&h.BoxSize, buf + o, sizeof(double)); o += 8;
  std::memcpy(&h.Omega0, buf + o, sizeof(double)); o += 8;
  std::memcpy(&h.OmegaLambda, buf + o, sizeof(double)); o += 8;
  std::memcpy(&h.HubbleParam, buf + o, sizeof(double)); o += 8;
  std::memcpy(&h.flag_stellarage, buf + o, sizeof(int)); o += 4;
  std::memcpy(&h.flag_metals, buf + o, sizeof(int)); o += 4;
  std::memcpy(h.npartTotalHighWord, buf + o, 6 * sizeof(unsigned int)); o += 24;
  std::memcpy(&h.flag_entropy_instead_u, buf + o, sizeof(int)); o += 4;
  (void)o;
}

// --- Format-1/2 record-marker protocol (io.c/read_ic.c, PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.2)
// --- Every record (header or data block) is bracketed by a 4-byte int record-length marker
// before and after the payload, ALWAYS a plain 4-byte int (never 8-byte/long), matching source.

bool writeRecord(std::ofstream &f, const void *data, int nbytes)
{
  f.write(reinterpret_cast<const char *>(&nbytes), sizeof(int));
  if (nbytes > 0) f.write(reinterpret_cast<const char *>(data), nbytes);
  f.write(reinterpret_cast<const char *>(&nbytes), sizeof(int));
  return (bool)f;
}

// Format-2's per-block descriptor: 8-byte record whose payload is a 4-char label + the size of
// the record that follows it (payload bytes + its own 2 markers), io.c:798-803,832-839.
bool writeF2Descriptor(std::ofstream &f, const char label[4], int followingRecordTotalBytes)
{
  char payload[8];
  std::memcpy(payload, label, 4);
  std::memcpy(payload + 4, &followingRecordTotalBytes, 4);
  return writeRecord(f, payload, 8);
}

bool readRecordHeader(std::ifstream &f, int &nbytes)
{
  f.read(reinterpret_cast<char *>(&nbytes), sizeof(int));
  return (bool)f;
}

bool readRecordTrailer(std::ifstream &f, int expected, std::string &error)
{
  int trailer = 0;
  f.read(reinterpret_cast<char *>(&trailer), sizeof(int));
  if (!f) { error = "unexpected EOF reading record trailer"; return false; }
  if (trailer != expected)
  {
    error = "incorrect block-sizes detected (leading=" + std::to_string(expected) +
            ", trailing=" + std::to_string(trailer) + ")";
    return false;
  }
  return true;
}

// Reads one full record (marker, payload, marker) into `payload`, resizing it to the declared
// size. Returns false (with `error` set) on any EOF/size mismatch.
bool readRecord(std::ifstream &f, std::vector<uint8_t> &payload, std::string &error)
{
  int n = 0;
  if (!readRecordHeader(f, n)) { error = "unexpected EOF reading record header"; return false; }
  payload.resize(n > 0 ? n : 0);
  if (n > 0) f.read(reinterpret_cast<char *>(payload.data()), n);
  if (!f) { error = "unexpected EOF reading record payload (" + std::to_string(n) + " bytes)"; return false; }
  return readRecordTrailer(f, n, error);
}

// Reads a format-2 block descriptor and checks its label matches `expectedLabel` (read_ic.c:
// 429-438's positional label check -- a consistency check against the fixed iteration order,
// not a lookup key). `followingRecordTotalBytes` is returned for informational purposes only;
// the actual data record is still read/validated independently via readRecord().
bool readF2Descriptor(std::ifstream &f, const char expectedLabel[4], std::string &error)
{
  std::vector<uint8_t> payload;
  if (!readRecord(f, payload, error)) return false;
  if (payload.size() != 8)
  {
    error = "format-2 block descriptor has unexpected size " + std::to_string(payload.size());
    return false;
  }
  if (std::memcmp(payload.data(), expectedLabel, 4) != 0)
  {
    char got[5] = {0};
    std::memcpy(got, payload.data(), 4);
    error = std::string("incorrect block-structure: expected label '") + std::string(expectedLabel, 4) +
            "', got '" + got + "'";
    return false;
  }
  return true;
}

// Sum of npart[] over the six particle types -- used throughout as the total particle count.
long totalParticles(const int npart[6])
{
  long n = 0;
  for (int t = 0; t < 6; ++t) n += npart[t];
  return n;
}

} // namespace

bool gadget_snapshot_write(const std::string &path, int snapFormat, bool longIds,
                            bool comovingIntegrationOn, GadgetSnapshotHeader header,
                            const GadgetParticleData &data, std::string &error)
{
  const size_t N = data.type.size();
  if (data.pos.size() != 3 * N || data.vel.size() != 3 * N || data.id.size() != N ||
      data.mass.size() != N)
  {
    error = "gadget_snapshot_write: inconsistent array sizes in GadgetParticleData";
    return false;
  }
  if (snapFormat != 1 && snapFormat != 2)
  {
    error = "gadget_snapshot_write: SnapFormat must be 1 or 2 (HDF5/3 not supported)";
    return false;
  }

  // Stable-sort a permutation by type -- matches Gadget-2's own on-disk ordering (grouped by
  // type, ascending, PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.5). Stable so that, for equal types,
  // relative order (e.g. particle ID order) is preserved -- a nicety, not a correctness
  // requirement, but costs nothing.
  std::vector<size_t> order(N);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                    [&](size_t a, size_t b) { return data.type[a] < data.type[b]; });

  // header.npart/npartTotal/npartTotalHighWord/num_files are derived from the actual data, not
  // taken from the caller -- single source of truth (gadget_snapshot.h's own documented contract).
  for (int t = 0; t < 6; ++t) header.npart[t] = 0;
  for (size_t i = 0; i < N; ++i)
  {
    int t = data.type[i];
    if (t < 0 || t > 5) { error = "gadget_snapshot_write: particle type out of range [0,5]"; return false; }
    header.npart[t]++;
  }
  for (int t = 0; t < 6; ++t)
  {
    header.npartTotal[t] = (unsigned int)header.npart[t];
    header.npartTotalHighWord[t] = 0; // single-file, well under 2^32 particles
  }
  header.num_files = 1;

  // Per-type fixed-mass detection (PHASE5_PARAMS_SNAPSHOT_SPEC.md Sec 2.4): if every particle of
  // type t shares the exact same mass, use a fixed table mass and omit type t from the MASS
  // block entirely; otherwise mass[t]=0 and every particle of that type is written explicitly.
  bool typeHasFixedMass[6] = {false, false, false, false, false, false};
  for (int t = 0; t < 6; ++t) header.mass[t] = 0.0;
  {
    size_t idx = 0;
    for (int t = 0; t < 6; ++t)
    {
      int cnt = header.npart[t];
      if (cnt == 0) continue;
      float m0 = data.mass[order[idx]];
      bool allEqual = true;
      for (int k = 1; k < cnt; ++k)
        if (data.mass[order[idx + k]] != m0) { allEqual = false; break; }
      if (allEqual)
      {
        header.mass[t] = (double)m0;
        typeHasFixedMass[t] = true;
      }
      idx += cnt;
    }
  }

  const double a = header.time;
  const double velScale = comovingIntegrationOn ? std::sqrt(a) * a : 1.0;

  std::ofstream f(path, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!f.is_open()) { error = "cannot open '" + path + "' for writing"; return false; }

  uint8_t headerBuf[HEADER_BYTES];
  serializeHeader(header, headerBuf);

  if (snapFormat == 2 && !writeF2Descriptor(f, "HEAD", (int)HEADER_BYTES + 8))
  { error = "write failed (HEAD descriptor)"; return false; }
  if (!writeRecord(f, headerBuf, (int)HEADER_BYTES)) { error = "write failed (header)"; return false; }

  // POS
  {
    std::vector<float> buf(3 * N);
    for (size_t i = 0; i < N; ++i)
      for (int c = 0; c < 3; ++c) buf[3 * i + c] = data.pos[3 * order[i] + c];
    int nbytes = (int)(3 * N * sizeof(float));
    if (snapFormat == 2 && !writeF2Descriptor(f, "POS ", nbytes + 8)) { error = "write failed (POS descriptor)"; return false; }
    if (!writeRecord(f, buf.data(), nbytes)) { error = "write failed (POS)"; return false; }
  }

  // VEL -- Phase 5 ticket 08 (PLAN.md): DIVIDES by velScale. Corrected from an earlier, INCOMPLETE
  // reading of Gadget-2's own source (a real self-caught mistake this same ticket, not hypothetical):
  // io.c's IO_VEL case (fill_write_buffer(), ~209-241) doesn't stop at
  // `fp[k] = P[pindex].Vel[k] + kick_terms` -- three lines further down (io.c:241,
  // `fp[k] *= sqrt(a3inv);` with `a3inv = 1/a^3` under comoving integration, io.c:150) it also
  // multiplies by `sqrt(a3inv) = 1/(sqrt(a)*a) = 1/velScale`, i.e. DIVIDES by velScale -- missed on
  // the first read of this file (the view that informed the earlier "no scaling" comment happened
  // to stop just above this line). The net effect: Gadget-2's regular snapshot OUTPUT ends up in
  // the SAME peculiar-velocity convention as a fresh initial-conditions file (write undoes exactly
  // what init.c:80's `*= sqrt(a)*a` IC-load transform does), which is what actually explains this
  // port's own empirical finding while validating ticket 08 -- a real stock-Gadget-2 snapshot's own
  // VEL block has RMS magnitude matching the ORIGINAL IC's raw VEL values, not the tiny internal
  // "canonical momentum" magnitude. This restores write/read as true mutual inverses (matching
  // gadget_snapshot_read's own `* velScale` on the IC-load path) -- the self-test below is back to
  // expecting an exact round trip, not the asymmetric relationship an earlier, incomplete fix here
  // called for.
  {
    std::vector<float> buf(3 * N);
    for (size_t i = 0; i < N; ++i)
      for (int c = 0; c < 3; ++c)
        buf[3 * i + c] = (float)(data.vel[3 * order[i] + c] / velScale);
    int nbytes = (int)(3 * N * sizeof(float));
    if (snapFormat == 2 && !writeF2Descriptor(f, "VEL ", nbytes + 8)) { error = "write failed (VEL descriptor)"; return false; }
    if (!writeRecord(f, buf.data(), nbytes)) { error = "write failed (VEL)"; return false; }
  }

  // ID -- 4-byte (unsigned int) or 8-byte (unsigned long long) depending on longIds, matching
  // Gadget-2's own LONGIDS compile flag (a build-time choice here too, GADGET_HIP_LONGIDS).
  {
    int elemBytes = longIds ? 8 : 4;
    int nbytes = (int)(N * elemBytes);
    if (snapFormat == 2 && !writeF2Descriptor(f, "ID  ", nbytes + 8)) { error = "write failed (ID descriptor)"; return false; }
    if (longIds)
    {
      std::vector<uint64_t> buf(N);
      for (size_t i = 0; i < N; ++i) buf[i] = data.id[order[i]];
      if (!writeRecord(f, buf.data(), nbytes)) { error = "write failed (ID)"; return false; }
    }
    else
    {
      std::vector<uint32_t> buf(N);
      for (size_t i = 0; i < N; ++i) buf[i] = (uint32_t)data.id[order[i]];
      if (!writeRecord(f, buf.data(), nbytes)) { error = "write failed (ID)"; return false; }
    }
  }

  // MASS -- only particles of a type without a fixed table mass, in the same type-then-index
  // order as everything else. Entirely absent from the file if every present type has a fixed
  // mass (Sec 2.3/2.4).
  {
    std::vector<float> buf;
    buf.reserve(N);
    for (size_t i = 0; i < N; ++i)
    {
      int t = data.type[order[i]];
      if (!typeHasFixedMass[t]) buf.push_back(data.mass[order[i]]);
    }
    if (!buf.empty())
    {
      int nbytes = (int)(buf.size() * sizeof(float));
      if (snapFormat == 2 && !writeF2Descriptor(f, "MASS", nbytes + 8)) { error = "write failed (MASS descriptor)"; return false; }
      if (!writeRecord(f, buf.data(), nbytes)) { error = "write failed (MASS)"; return false; }
    }
  }

  if (!f) { error = "write failed (stream error)"; return false; }
  return true;
}

namespace {

// Reads exactly one physical file's HEAD/POS/VEL/ID/MASS blocks, raw (no comoving velocity
// conversion -- the public gadget_snapshot_read() below applies that once, after any multi-file
// concatenation, since every physical file of one logical snapshot shares the same header.time
// by Gadget-2's own convention). Does not reject header.num_files>1: that field just reports the
// logical snapshot's total file count, which is a property of the *set*, not a defect in this
// one physical file.
bool readOneFile(const std::string &path, int icFormat, bool longIds,
                  GadgetSnapshotHeader &header, GadgetParticleData &data, std::string &error)
{
  std::ifstream f(path, std::ios::binary | std::ios::in);
  if (!f.is_open()) { error = "cannot open '" + path + "' for reading"; return false; }

  if (icFormat == 2 && !readF2Descriptor(f, "HEAD", error)) return false;
  {
    std::vector<uint8_t> payload;
    if (!readRecord(f, payload, error)) return false;
    if (payload.size() != HEADER_BYTES)
    {
      error = "header record is " + std::to_string(payload.size()) + " bytes, expected " +
              std::to_string(HEADER_BYTES);
      return false;
    }
    deserializeHeader(payload.data(), header);
  }

  const long N = totalParticles(header.npart);
  data = GadgetParticleData();
  data.pos.resize(3 * N);
  data.vel.resize(3 * N);
  data.id.resize(N);
  data.mass.resize(N);
  data.type.resize(N);
  {
    size_t idx = 0;
    for (int t = 0; t < 6; ++t)
      for (int k = 0; k < header.npart[t]; ++k) data.type[idx++] = t;
  }

  // POS
  {
    if (icFormat == 2 && !readF2Descriptor(f, "POS ", error)) return false;
    std::vector<uint8_t> payload;
    if (!readRecord(f, payload, error)) return false;
    if ((long)payload.size() != 3 * N * (long)sizeof(float))
    { error = "POS block size mismatch"; return false; }
    std::memcpy(data.pos.data(), payload.data(), payload.size());
  }

  // VEL -- left as raw on-disk values; the inverse comoving conversion is applied once by the
  // public wrapper below, after any multi-file concatenation.
  {
    if (icFormat == 2 && !readF2Descriptor(f, "VEL ", error)) return false;
    std::vector<uint8_t> payload;
    if (!readRecord(f, payload, error)) return false;
    if ((long)payload.size() != 3 * N * (long)sizeof(float))
    { error = "VEL block size mismatch"; return false; }
    std::memcpy(data.vel.data(), payload.data(), payload.size());
  }

  // ID
  {
    if (icFormat == 2 && !readF2Descriptor(f, "ID  ", error)) return false;
    std::vector<uint8_t> payload;
    if (!readRecord(f, payload, error)) return false;
    long expected = N * (longIds ? 8 : 4);
    if ((long)payload.size() != expected) { error = "ID block size mismatch"; return false; }
    if (longIds)
    {
      std::vector<uint64_t> buf(N);
      std::memcpy(buf.data(), payload.data(), payload.size());
      for (long i = 0; i < N; ++i) data.id[i] = buf[i];
    }
    else
    {
      std::vector<uint32_t> buf(N);
      std::memcpy(buf.data(), payload.data(), payload.size());
      for (long i = 0; i < N; ++i) data.id[i] = buf[i];
    }
  }

  // MASS -- only present if at least one type has mass[type]==0 (Sec 2.3/2.4); entirely absent
  // from the file otherwise. Every particle's mass[] entry is populated either way: fixed-table
  // types get header.mass[type] directly, others come from this block.
  long massBlockEntries = 0;
  for (int t = 0; t < 6; ++t)
    if (header.mass[t] == 0.0) massBlockEntries += header.npart[t];

  std::vector<float> massBlock;
  if (massBlockEntries > 0)
  {
    if (icFormat == 2 && !readF2Descriptor(f, "MASS", error)) return false;
    std::vector<uint8_t> payload;
    if (!readRecord(f, payload, error)) return false;
    if ((long)payload.size() != massBlockEntries * (long)sizeof(float))
    { error = "MASS block size mismatch"; return false; }
    massBlock.resize(massBlockEntries);
    std::memcpy(massBlock.data(), payload.data(), payload.size());
  }
  {
    size_t idx = 0, massIdx = 0;
    for (int t = 0; t < 6; ++t)
    {
      for (int k = 0; k < header.npart[t]; ++k)
      {
        data.mass[idx] = (header.mass[t] != 0.0) ? (float)header.mass[t] : massBlock[massIdx++];
        ++idx;
      }
    }
  }

  return true;
}

} // namespace

bool gadget_snapshot_read(const std::string &path, int icFormat, bool longIds,
                           bool comovingIntegrationOn, GadgetSnapshotHeader &header,
                           GadgetParticleData &data, std::string &error)
{
  if (icFormat != 1 && icFormat != 2)
  {
    error = "gadget_snapshot_read: ICFormat must be 1 or 2 (HDF5/3 not supported)";
    return false;
  }

  // Mirrors Gadget-2's own find_files() (read_ic.c:615-622): a multi-file snapshot's pieces are
  // named "<path>.0", "<path>.1", ...; a single-file snapshot is the bare, unsuffixed path. Which
  // convention is in play is decided purely by whether "<path>.0" opens -- never by inspecting
  // any header's own num_files field first, exactly like the reference implementation.
  const std::string firstPath = path + ".0";
  const bool multiFile = std::ifstream(firstPath, std::ios::binary).good();

  // Applies Gadget-2's own initial-conditions velocity transform once, in place, using the
  // assembled header's time -- shared by both branches below since a real Gadget-2 snapshot's
  // header.time is identical across every physical file of one logical snapshot.
  //
  // Phase 5 ticket 08 (PLAN.md): this is a MULTIPLY, not the divide this file had before --
  // confirmed by direct source citation, not assumption. gadget_snapshot_read() has exactly one
  // real caller in this codebase (main.cpp's `haveGadgetIC` branch, loading `InitCondFile`), so
  // it only ever needs to play the role of Gadget-2's own initial-conditions loader, and that
  // role's actual transform is `init.c:80`: `P[i].Vel[j] *= sqrt(All.Time) * All.Time;`, applied
  // unconditionally to every freshly-read IC under comoving integration. (Gadget-2's own regular
  // snapshot *output*, by contrast, applies no VEL transform at all on write -- see
  // gadget_snapshot_write's own VEL comment -- so this function deliberately does NOT try to be a
  // symmetric inverse of gadget_snapshot_write; matching Gadget-2's real, role-dependent asymmetry
  // is the correct behavior here, confirmed against source rather than assumed for symmetry's
  // sake.)
  auto applyIcVelocityTransform = [&]() -> bool {
    if (!comovingIntegrationOn) return true;
    const double a = header.time;
    const double velScale = std::sqrt(a) * a;
    if (!(velScale > 0.0))
    {
      error = "gadget_snapshot_read: comoving integration on but header.time=" +
              std::to_string(a) + " gives a non-positive velocity scale";
      return false;
    }
    for (auto &v : data.vel) v = (float)(v * velScale);
    return true;
  };

  if (!multiFile)
    return readOneFile(path, icFormat, longIds, header, data, error) && applyIcVelocityTransform();

  // Multi-file: bucket every physical file's particles by type (each file is already internally
  // type-ordered, ascending, by itself -- readOneFile()'s own convention) and concatenate the six
  // per-type buckets, ascending, across all files. Plain "read file 0, then file 1, ..." would
  // NOT do this correctly in general: this port's whole codebase (softening, zoom mask, snapshot
  // write) assumes data is grouped by ascending type globally, but a real zoom IC (e.g. the
  // ginnungagap-style multi-resolution set this ticket validates against) assigns each
  // *resolution level* its own file, and distinct levels do not have to appear in ascending
  // particle-type order across files (e.g. observed file order 4,2,2,1 for coarse-to-fine here).
  GadgetSnapshotHeader h0;
  GadgetParticleData d0;
  if (!readOneFile(firstPath, icFormat, longIds, h0, d0, error)) return false;
  const int numFiles = h0.num_files > 0 ? h0.num_files : 1;

  std::vector<float>    posByType[6], velByType[6], massByType[6];
  std::vector<uint64_t> idByType[6];

  auto bucket = [&](const GadgetSnapshotHeader &hf, const GadgetParticleData &df) {
    size_t idx = 0;
    for (int t = 0; t < 6; ++t)
    {
      for (int k = 0; k < hf.npart[t]; ++k, ++idx)
      {
        posByType[t].insert(posByType[t].end(), df.pos.begin() + 3 * idx, df.pos.begin() + 3 * idx + 3);
        velByType[t].insert(velByType[t].end(), df.vel.begin() + 3 * idx, df.vel.begin() + 3 * idx + 3);
        idByType[t].push_back(df.id[idx]);
        massByType[t].push_back(df.mass[idx]);
      }
    }
  };
  bucket(h0, d0);

  for (int fn = 1; fn < numFiles; ++fn)
  {
    const std::string fp = path + "." + std::to_string(fn);
    GadgetSnapshotHeader hf;
    GadgetParticleData   df;
    if (!readOneFile(fp, icFormat, longIds, hf, df, error)) return false;
    bucket(hf, df);
  }

  // Global header fields (time, redshift, BoxSize, Omega0, ...) are identical across every
  // physical file of one logical snapshot by Gadget-2's own convention -- file 0's copy is as
  // good as any. npart[]/npartTotal[]/num_files, in contrast, are deliberately NOT trusted from
  // any single file's header (a real discrepancy was observed between file 0's own npartTotal and
  // the other files' npartTotal for the same IC set) -- they're recomputed from what was actually
  // read, then presented as num_files=1 since the rest of this single-process port only ever
  // consumes one assembled in-memory particle set regardless of how many physical files it came
  // from.
  header = h0;
  header.num_files = 1;
  data = GadgetParticleData();
  for (int t = 0; t < 6; ++t)
  {
    header.npart[t] = (int)idByType[t].size();
    header.npartTotal[t] = (unsigned int)idByType[t].size();
    header.npartTotalHighWord[t] = 0;
    data.pos.insert(data.pos.end(), posByType[t].begin(), posByType[t].end());
    data.vel.insert(data.vel.end(), velByType[t].begin(), velByType[t].end());
    data.id.insert(data.id.end(), idByType[t].begin(), idByType[t].end());
    data.mass.insert(data.mass.end(), massByType[t].begin(), massByType[t].end());
    data.type.insert(data.type.end(), idByType[t].size(), t);
  }

  if (!applyIcVelocityTransform()) return false;

  return true;
}

bool gadget_snapshot_selftest(const std::string &tmpPath, int snapFormat, bool longIds,
                               std::string &report)
{
  std::ostringstream rep;
  bool allOk = true;

  auto checkPass = [&](bool comovingIntegrationOn) {
    const char *label = comovingIntegrationOn ? "comoving=on" : "comoving=off";

    // Synthetic multi-type set: type 1 (Halo, N1 particles) all share mass 2.5 exactly --
    // exercises the Massarr fixed-mass path (MASS block should omit type 1 entirely). Type 4
    // (Stars, N2 particles) each get a distinct mass -- exercises the explicit MASS-block path.
    const int N1 = 10, N2 = 7;
    const int N = N1 + N2;
    GadgetParticleData in;
    in.pos.resize(3 * N);
    in.vel.resize(3 * N);
    in.id.resize(N);
    in.mass.resize(N);
    in.type.resize(N);
    for (int i = 0; i < N; ++i)
    {
      in.pos[3 * i + 0] = 10.0f + 0.37f * i;
      in.pos[3 * i + 1] = -5.0f + 0.11f * i * i;
      in.pos[3 * i + 2] = std::sin((float)i);
      in.vel[3 * i + 0] = 0.01f * i - 0.5f;
      in.vel[3 * i + 1] = std::cos(0.3f * i);
      in.vel[3 * i + 2] = -0.02f * i;
      in.id[i] = 1000000ULL + (uint64_t)i * 97ULL; // exercises >32-bit-unfriendly-looking values
      if (i < N1) { in.type[i] = 1; in.mass[i] = 2.5f; }
      else        { in.type[i] = 4; in.mass[i] = 1.0f + 0.01f * (i - N1); }
    }

    GadgetSnapshotHeader hdr;
    hdr.time = comovingIntegrationOn ? 0.5 : 1.0; // scale factor a=0.5 under comoving, else unused
    hdr.redshift = comovingIntegrationOn ? (1.0 / hdr.time - 1.0) : 0.0;
    hdr.BoxSize = 100.0;
    hdr.Omega0 = 0.3;
    hdr.OmegaLambda = 0.7;
    hdr.HubbleParam = 0.7;

    std::string err;
    if (!gadget_snapshot_write(tmpPath, snapFormat, longIds, comovingIntegrationOn, hdr, in, err))
    {
      rep << "[" << label << "] FAIL: write error: " << err << "\n";
      allOk = false;
      return;
    }

    GadgetSnapshotHeader outHdr;
    GadgetParticleData out;
    if (!gadget_snapshot_read(tmpPath, snapFormat, longIds, comovingIntegrationOn, outHdr, out, err))
    {
      rep << "[" << label << "] FAIL: read error: " << err << "\n";
      allOk = false;
      return;
    }

    bool ok = true;
    if (outHdr.npart[1] != N1 || outHdr.npart[4] != N2)
    {
      rep << "[" << label << "] FAIL: npart mismatch (got [1]=" << outHdr.npart[1]
          << " [4]=" << outHdr.npart[4] << ", want " << N1 << "/" << N2 << ")\n";
      ok = false;
    }
    if (outHdr.mass[1] != 2.5)
    {
      rep << "[" << label << "] FAIL: expected fixed Massarr[1]=2.5, got " << outHdr.mass[1] << "\n";
      ok = false;
    }
    if (outHdr.mass[4] != 0.0)
    {
      rep << "[" << label << "] FAIL: expected Massarr[4]=0 (per-particle masses), got "
          << outHdr.mass[4] << "\n";
      ok = false;
    }
    if ((long)out.type.size() != N)
    {
      rep << "[" << label << "] FAIL: particle count mismatch\n";
      ok = false;
    }
    else
    {
      // Data is written grouped by type ascending (type 1 block first, then type 4), matching
      // `in`'s own layout here since `in` was already built in that order -- direct index
      // comparison is valid.
      for (int i = 0; i < N && ok; ++i)
      {
        if (out.type[i] != in.type[i] || out.id[i] != in.id[i])
        { rep << "[" << label << "] FAIL: type/id mismatch at " << i << "\n"; ok = false; break; }
        if (out.mass[i] != in.mass[i])
        { rep << "[" << label << "] FAIL: mass mismatch at " << i << " (" << out.mass[i]
              << " vs " << in.mass[i] << ")\n"; ok = false; break; }
        for (int c = 0; c < 3 && ok; ++c)
        {
          if (out.pos[3 * i + c] != in.pos[3 * i + c])
          { rep << "[" << label << "] FAIL: pos mismatch at " << i << "." << c << "\n"; ok = false; break; }
          // write divides by velScale (io.c:150+241, a3inv/sqrt(a3inv)) and read multiplies by the
          // same velScale (init.c:80) -- true mutual inverses again, so write-then-read is expected
          // to reproduce `in` exactly (up to float roundoff from the double-precision scale/unscale
          // pair). With comoving off, velScale is exactly 1.0 so it should still be exact.
          float relErr = std::fabs(out.vel[3 * i + c] - in.vel[3 * i + c]) /
                         std::max(1e-6f, std::fabs(in.vel[3 * i + c]));
          float tol = comovingIntegrationOn ? 1e-5f : 0.0f;
          if (relErr > tol)
          { rep << "[" << label << "] FAIL: vel mismatch at " << i << "." << c << " relErr=" << relErr << "\n"; ok = false; break; }
        }
      }
    }

    if (ok) rep << "[" << label << "] PASS (N=" << N << ", format=" << snapFormat
                << ", longIds=" << (longIds ? "8-byte" : "4-byte") << ")\n";
    else    allOk = false;
  };

  checkPass(false);
  checkPass(true);

  report = rep.str();
  return allOk;
}
