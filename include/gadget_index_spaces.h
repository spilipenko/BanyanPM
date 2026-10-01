#ifndef GADGET_INDEX_SPACES_H
#define GADGET_INDEX_SPACES_H

// T44 item 1, second half -- the three index spaces, made executable.
//
// THE PROBLEM (see include/buffer_registry.h for the full derivation and the four bugs it cost).
// Particle buffers live in up to three index spaces at once:
//
//   CurrentOrder   the post-sort slot. sort_bodies permutes bodies_key, bodies_Ppos, bodies_ids
//                  and bodies_h on its default per-step path; anything produced after the sort is
//                  also in this space.
//   OriginalOrder  NOT permuted by the per-step sort, so it must be reached through
//                  oriParticleOrder: bodies_vel, bodies_Pvel, bodies_acc0, bodies_time.
//   IdSpace        indexed by the particle id. GravPM is stored here precisely so it survives the
//                  re-sort between PM steps.
//
// All three are `uint`-shaped, so every wrong pairing is in range, silent, and plausible. The
// confirmed cost so far: T12, T23 Bug 2, T27 (a MISdiagnosis that added a translation where none
// belonged), C-D-03b (99.6% of iterations did nothing for weeks), and Phase 3 bug 4.
//
// WHY THE TYPE GOES ON THE KERNEL PARAMETER, NOT ON dev_mem.
// The space of a buffer is NOT a property of the buffer -- it is a property of WHERE YOU ARE in the
// step. buffer_registry.h's own invariant: in Phase 1 (predict, before sort_bodies)
// oriParticleOrder is the identity and every buffer may be indexed directly; the spaces only
// diverge in Phase 2; correct_particles re-unifies them in Phase 3. A tag on dev_mem would
// therefore be false for a third of the step. A tag on a kernel parameter is not: each kernel runs
// in one known phase, and the declaration is exactly where the author decides which space it means.
//
// WHAT THIS BUYS. `vel[idx]` where vel is OriginalOrder and idx is a CurrentIdx does not compile.
// That is T23 Bug 2 and C-D-03b.

// ALL THREE ARE UNSIGNED, and deliberately so.
//
// These are array subscripts: they are never negative, and an unsigned subscript cannot be
// sign-extended by accident. Before this header existed the codebase spelled the same quantity --
// the thread's slot index, blockIdx*blockDim + threadIdx -- three different ways: `int` in 10
// kernels, `uint` in 5, `size_t` in 2. That mixture is invisible in behaviour (every count here is
// far below 2^31) but it means the generated address arithmetic alternates between sign- and
// zero-extension for no reason, and a reader cannot tell which convention a given kernel follows.
//
// Note what these are NOT: none of them is the particle ID. ParticleId is the id read from
// bodies_ids, which is `ullong` and always has been. CurrentIdx and OriginalIdx are SLOT numbers --
// positions in an array -- and the distinction between them is which permutation the array is in.
// A GROUP id, not a particle slot. Groups are the walk's work units; group arrays are n_groups long,
// not n_bodies long, so a particle index used on one is out of range as well as wrong. The dangerous
// step is the CONVERSION: body2group_list is particle-indexed (SPACE_CURRENT) and its VALUE is a
// group id, so `valid_list[body2grouplist[idx]]` crosses two spaces on one line. group_of() makes
// that crossing a named call.
//
// `valid_list` is the reason this is worth typing: the same parameter name means a GROUP-indexed
// array in setActiveGroups and a NODE-indexed one in build_tree.cu's leaf/node validity kernels.
struct GroupIdx    { unsigned int v; };             // group id (0 .. n_groups-1)
struct CurrentIdx  { unsigned int v; };             // post-sort slot number
struct OriginalIdx { unsigned int v; };             // slot number in a buffer the sort did not permute
struct ParticleId  { unsigned long long v; };       // the particle's own id (bodies_ids is ullong)

// Each wrapper holds exactly one pointer and is standard-layout, so it is layout-identical to that
// pointer: passing it to a kernel costs nothing, and dev_mem::as<>() below is a valid reinterpret.
template <class T> struct CurrentOrder {
  T *p;
  __host__ __device__ T& operator[](CurrentIdx i) const { return p[i.v]; }
  __host__ __device__ bool is_null() const { return p == 0; }
};
template <class T> struct OriginalOrder {
  T *p;
  __host__ __device__ T& operator[](OriginalIdx i) const { return p[i.v]; }
  __host__ __device__ bool is_null() const { return p == 0; }
};
template <class T> struct IdSpace {
  T *p;
  __host__ __device__ T& operator[](ParticleId i) const { return p[i.v]; }
  __host__ __device__ bool is_null() const { return p == 0; }
};
template <class T> struct GroupOrder {
  T *p;
  __host__ __device__ T& operator[](GroupIdx i) const { return p[i.v]; }
  __host__ __device__ bool is_null() const { return p == 0; }
};

// The only two translations there are. Naming them means a space change is always visible as a
// call, never as an implicit reuse of the same integer.
// Templated on the pointee so they accept both const and non-const views: correct_particles both
// READS oriParticleOrder and resets it to the identity, so its parameter cannot be const.
template <class U>
__host__ __device__ inline OriginalIdx to_original(CurrentIdx i, CurrentOrder<U> oriParticleOrder)
{ OriginalIdx o; o.v = oriParticleOrder[i]; return o; }

template <class U>
__host__ __device__ inline ParticleId id_at(CurrentIdx i, CurrentOrder<U> ids)
{ ParticleId p; p.v = ids[i]; return p; }

// particle slot -> the group that owns it. body2group_list is particle-indexed and its value is a
// group id, so this is the third space crossing, and the one C-D-03b got wrong in the other
// direction (it paired a group-derived index with an unpermuted particle buffer).
template <class U>
__host__ __device__ inline GroupIdx group_of(CurrentIdx i, CurrentOrder<U> body2group)
{ GroupIdx g; g.v = body2group[i]; return g; }

#endif
