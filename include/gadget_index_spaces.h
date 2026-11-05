#ifndef GADGET_INDEX_SPACES_H
#define GADGET_INDEX_SPACES_H

// T44 item 1, second half -- the index spaces, made executable.
//
// THE PROBLEM (see include/buffer_registry.h for the full derivation and the bugs it cost).
// Particle buffers live in up to two index spaces at once:
//
//   CurrentOrder   the post-sort slot. sort_bodies permutes every particle buffer
//                  (bodies_key, bodies_Ppos, bodies_vel, bodies_ids, bodies_acc0, bodies_time,
//                  and bodies_h under BONSAI_DENSITY) on its per-step path; anything produced after
//                  the sort is also in this space. Since T81 item B there is no other particle
//                  space: nothing survives the sort unpermuted, so there is no "original order"
//                  and no permutation map to consult.
//   IdSpace        indexed by the particle id. GravPM is stored here precisely so it survives the
//                  re-sort between PM steps.
//
// Both are `uint`-shaped, so every wrong pairing is in range, silent, and plausible. The confirmed
// cost so far: T12, T23 Bug 2, T27 (a MISdiagnosis that added a translation where none belonged),
// C-D-03b (99.6% of iterations did nothing for weeks), and Phase 3 bug 4 -- all of which existed
// because a buffer that sort_bodies did not permute was indexed with the current sorted slot.
// T81 item B removed that whole class by permuting every particle buffer every step.
//
// WHY THE TYPE GOES ON THE KERNEL PARAMETER, NOT ON dev_mem.
// The space of a buffer is NOT a property of the buffer -- it is a property of WHERE YOU ARE in the
// step. In Phase 1 (predict, before sort_bodies) every buffer is in the previous step's order and
// may be indexed directly; sort_bodies permutes them; correct_particles updates them in place. A
// tag on dev_mem would therefore be false for part of the step. A tag on a kernel parameter is not:
// each kernel runs in one known phase, and the declaration is exactly where the author decides
// which space it means.
//
// WHAT THIS BUYS. `vel[idx]` where vel is an IdSpace array and idx is a CurrentIdx does not
// compile. That is the class of bug that T23 Bug 2 and C-D-03b were.

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
// bodies_ids, which is `ullong` and always has been. CurrentIdx is a SLOT number -- a position in
// an array. A GROUP id, not a particle slot. Groups are the walk's work units; group arrays are
// n_groups long, not n_bodies long, so a particle index used on one is out of range as well as
// wrong. The dangerous step is the CONVERSION: body2group_list is particle-indexed (SPACE_CURRENT)
// and its VALUE is a group id, so `valid_list[body2grouplist[idx]]` crosses two spaces on one line.
// group_of() makes that crossing a named call.
//
// `valid_list` is the reason this is worth typing: the same parameter name means a GROUP-indexed
// array in setActiveGroups and a NODE-indexed one in build_tree.cu's leaf/node validity kernels.
struct GroupIdx    { unsigned int v; };             // group id (0 .. n_groups-1)
struct CurrentIdx  { unsigned int v; };             // post-sort slot number
struct ParticleId  { unsigned long long v; };       // the particle's own id (bodies_ids is ullong)

// Each wrapper holds exactly one pointer and is standard-layout, so it is layout-identical to that
// pointer: passing it to a kernel costs nothing, and dev_mem::as<>() below is a valid reinterpret.
template <class T> struct CurrentOrder {
  T *p;
  __host__ __device__ T& operator[](CurrentIdx i) const { return p[i.v]; }
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

// The one translation there still is. Naming it means a space change is always visible as a call,
// never as an implicit reuse of the same integer.
template <class U>
__host__ __device__ inline ParticleId id_at(CurrentIdx i, CurrentOrder<U> ids)
{ ParticleId p; p.v = ids[i]; return p; }

// particle slot -> the group that owns it. body2group_list is particle-indexed and its value is a
// group id, so this is the space crossing (C-D-03b got it wrong: it paired a group-derived index
// with an unpermuted particle buffer).
template <class U>
__host__ __device__ inline GroupIdx group_of(CurrentIdx i, CurrentOrder<U> body2group)
{ GroupIdx g; g.v = body2group[i]; return g; }

#endif
