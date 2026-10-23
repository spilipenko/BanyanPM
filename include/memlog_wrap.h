// Redirect raw hipMalloc/hipHostMalloc in ONE translation unit to the ledger.
//
// Include this LAST, after every other header. Two reasons, both learned the hard way:
//   * memlog.h is pulled in transitively (octree.h -> my_cuda_rt.h), so an #ifdef inside it is
//     evaluated before any opt-in macro in the .cpp could be seen -- the wrapper silently never
//     activated, and a PMGRID=512 run reported the same total as PMGRID=64 with the entire
//     particle mesh uncounted.
//   * dev_mem calls hipMalloc internally. If these macros were active while my_cuda_rt.h is
//     compiled, every dev_mem allocation would be counted twice.
// NO include guard on purpose: this header defines macros only, and including it twice is
// harmless, while a guard would make the "include last" rule silently fail under transitivity.
#include "memlog.h"   // safe: #pragma once, and this header is included after my_cuda_rt.h

#ifdef GADGET_HIP_MEMLOG_BUILD
#define hipMalloc(p, n)     memlog::hip_malloc((void**)(p), (n), #p, __FILE__, __LINE__)
#define hipHostMalloc(p, n) memlog::hip_host_malloc((void**)(p), (n), #p, __FILE__, __LINE__)
#endif
