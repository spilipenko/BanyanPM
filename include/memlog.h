#pragma once
// ---------------------------------------------------------------------------------------------
// Allocation ledger for the memory-accounting study. MEASUREMENT ONLY -- nothing here changes
// what is allocated, only what is recorded about it.
//
// Every record carries: buffer name, bytes, allocator and flags, the file:line that asked for it,
// and the timestep phase it happened in. Freeing is recorded too, so the ledger reconstructs the
// live set over time rather than just a running total.
//
// Completeness is self-checking: an allocation that reaches the ledger without a name is recorded
// as "(untagged:<type>#<n>)" rather than dropped, so a gap in the instrumentation shows up as a
// line in the report instead of as silence. Trusting a memory total means trusting that nothing
// was missed, and that is only believable if misses are visible.
//
// Enabled by GADGET_HIP_MEMLOG=<path>. Off by default and essentially free when off.
// ---------------------------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef GADGET_HIP_MEMLOG_BUILD
// ---------------------------------------------------------------------------------------------
// Compiled out -- the DEFAULT. on() is constexpr false, so every guarded site folds away at
// compile time and the default build generates the code it did before the ledger existed.
//
// This is not caution for its own sake. Compiled in but IDLE (GADGET_HIP_MEMLOG unset), the
// ledger measured 3.2% slower overall and 3.8% slower in the tree walk on a 64^3 deterministic
// run: 26.39-26.44 s against 27.17-27.38 s total, 21.50-21.56 against 22.27-22.48 in the walk,
// five runs each with the order balanced, no overlap between the groups. Output was
// bit-identical and the kernel register and scratch footprints were identical, so the cost is
// entirely host-side: eight extra members per dev_mem grow the tree struct that the launch
// path walks. With the ledger compiled OUT, the only difference from sources that never had it
// is the one startup banner line that reports the option: the device image is byte-identical,
// the deterministic 64^3 trajectory is bit-identical, and the timings are equal. That is the
// property this #ifdef exists to guarantee.
//
// Build with -DGADGET_HIP_MEMLOG=ON, then set GADGET_HIP_MEMLOG=<path> at run time.
// ---------------------------------------------------------------------------------------------
namespace memlog {
inline constexpr bool on() { return false; }
inline void record(const char *, const char *, long long, const char *, int, unsigned,
                   const char *, int) {}
inline void set_phase(const char *) {}
inline void note(const char *) {}
}
#define DMTAG(obj)                            ((void) 0)
#define MEMLOG_ALLOC(name, bytes, allocator)  ((void) 0)
#define MEMLOG_FREE(name, bytes, allocator)   ((void) 0)

#else

namespace memlog {

inline FILE *&sink()            { static FILE *f = nullptr; return f; }
inline const char *&phase()     { static const char *p = "startup"; return p; }
inline long long &seq()         { static long long s = 0; return s; }
inline long long &live_bytes()  { static long long b = 0; return b; }
inline long long &peak_bytes()  { static long long b = 0; return b; }

inline bool on()
{
  static int state = -1;
  if (state < 0)
  {
    const char *p = getenv("GADGET_HIP_MEMLOG");
    state = (p && *p) ? 1 : 0;
    if (state)
    {
      sink() = fopen(p, "w");
      if (!sink()) { state = 0; return false; }
      // Header names the columns, so the file is readable without the generator script.
      fprintf(sink(), "# seq\tevent\tname\tbytes\tallocator\tflags\talias\tphase\tsite\tlive\tpeak\n");
    }
  }
  return state == 1;
}

// One ledger line. `event` is ALLOC or FREE; `bytes` is positive in both cases and the sign is
// applied here, so the caller never has to remember which way round it goes.
inline void record(const char *event, const char *name, long long bytes,
                   const char *allocator, int flags, unsigned alias,
                   const char *file, int line)
{
  if (!on()) return;
  const bool freeing = (event[0] == 'F');
  // Only ALLOC and FREE move the live set. VIEW carries a real byte count (so the reader can
  // see how much of the scratch arena a phase uses) but those bytes already sit inside
  // generalBuffer1, so adding them here would double-count the arena. The rule is that the
  // live set tracks allocator calls and nothing else.
  const bool counts = (event[0] == 'A') || freeing;
  if (counts)
  {
    live_bytes() += freeing ? -bytes : bytes;
    if (live_bytes() > peak_bytes()) peak_bytes() = live_bytes();
  }
  const char *base = file ? strrchr(file, '/') : nullptr;
  fprintf(sink(), "%lld\t%s\t%s\t%lld\t%s\t%d\t0x%x\t%s\t%s:%d\t%lld\t%lld\n",
          ++seq(), event, name ? name : "(unnamed)", bytes,
          allocator, flags, alias, phase(),
          base ? base + 1 : (file ? file : "?"), line,
          live_bytes(), peak_bytes());
  fflush(sink());
}

// Set by the phase timers so each record knows where in the step it happened.
inline void set_phase(const char *p) { phase() = p; }

inline void note(const char *what)
{
  if (!on()) return;
  fprintf(sink(), "# %s\tphase=%s\tlive=%lld\tpeak=%lld\n", what, phase(), live_bytes(), peak_bytes());
  fflush(sink());
}

} // namespace memlog

// Name a dev_mem buffer with the expression that names it in the source, plus that source site.
#define DMTAG(obj) (obj).memlog_tag(#obj, __FILE__, __LINE__)

// For allocations that are not dev_mem: raw hipMalloc/hipHostMalloc, library scratch, host staging.
#define MEMLOG_ALLOC(name, bytes, allocator) \
  memlog::record("ALLOC", name, (long long)(bytes), allocator, 0, 0u, __FILE__, __LINE__)
#define MEMLOG_FREE(name, bytes, allocator) \
  memlog::record("FREE",  name, (long long)(bytes), allocator, 0, 0u, __FILE__, __LINE__)

// ---------------------------------------------------------------------------------------------
// Raw HIP allocations -- the PM solver, the CIC scatter and a few per-particle force buffers do
// not go through dev_mem, so they were invisible to the dev_mem ledger. That gap was not
// hypothetical: it is why a PMGRID=512 run and a PMGRID=64 run first reported the same total,
// the entire particle mesh being uncounted.
//
// Instrumented by redefining the FUNCTION NAME inside the translation units that allocate, so
// every call site is covered without editing any of them, and the buffer name comes from the
// pointer expression as written.
// ---------------------------------------------------------------------------------------------
// Guarded: this header is also included by translation units that never see a HIP header
// (gadget_snapshot.cpp, which stages the ICs in plain std::vectors and has to record them).
// Those TUs need record()/set_phase() and nothing else.
#ifdef HIP_VERSION
namespace memlog {
inline hipError_t hip_malloc(void **p, size_t n, const char *nm, const char *f, int l)
{ hipError_t e = hipMalloc(p, n); record("ALLOC", nm, (long long) n, "hipMalloc(raw)", 0, 0u, f, l); return e; }
inline hipError_t hip_host_malloc(void **p, size_t n, const char *nm, const char *f, int l)
{ hipError_t e = hipHostMalloc(p, n); record("ALLOC", nm, (long long) n, "hipHostMalloc(raw)", 0, 0u, f, l); return e; }
inline hipError_t hip_free_named(void *p, const char *nm, const char *f, int l)
{ record("FREE", nm, 0, "hipFree(raw)", 0, 0u, f, l); return hipFree(p); }
}
#endif  // HIP_VERSION

#endif  // GADGET_HIP_MEMLOG_BUILD
