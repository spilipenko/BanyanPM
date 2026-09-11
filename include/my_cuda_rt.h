#include "hip/hip_runtime.h"
#ifndef _MY_CUDA_H_
#define _MY_CUDA_H_

#include <cmath>


//#include <sys/time.h>

#if defined(_WIN32) && !defined(_WIN64)
  //Only use this on 32bit Windows builds
  #include <typeinfo.h>
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <vector>
#include <map>
#include <hip/hip_runtime.h>
#include <unistd.h>
#ifdef __HIPCC__
#include <thrust/device_ptr.h>
#endif

#include <iostream>
#include "log.h"

//Some easy to use typedefs
typedef float4 real4;
typedef float real;
#define make_real4 make_float4
typedef unsigned int uint;

using namespace std;


#define cl_mem void*


#  define CU_SAFE_CALL_KERNEL( call , kernel )       CU_SAFE_CALL(call);

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define CU_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(hipError_t err, const char *file, const int line )
{
    if(hipSuccess != err)
    {
//        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, hipGetErrorString( err ) );
      LOGF(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, hipGetErrorString( err ) );
      fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, hipGetErrorString( err ) );
      ::exit(-1);
    }
}

// This will output the proper error string when calling hipGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    hipError_t err = hipGetLastError();
    if (hipSuccess != err)
    {
//        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, hipGetErrorString( err ) );
        LOGF(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, hipGetErrorString( err ) );
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, hipGetErrorString( err ) );
        ::exit(-1);
    }
}

// end of CUDA Helper Functions



//OpenCL to CUDA macro / functions
__inline__ hipError_t clFinish(int param)
{
  #if CUDART_VERSION >= 4000
          return hipDeviceSynchronize();
  #else
          return hipDeviceSynchronize();
  #endif
}


static int getNumberOfCUDADevices()
{
  // Get number of devices supporting CUDA
  int temp = 0;
  CU_SAFE_CALL(hipGetDeviceCount(&temp));
  return temp;
}


namespace my_dev {

  class context {
  protected:
    size_t dev;

    int ciDeviceCount;

    bool hContext_flag;
    bool hInit_flag;
    bool logfile_flag;
    bool disable_timing;

    std::ostream *logFile;

    int logID;  //Unique ID to every log line


    //Events:
    hipEvent_t start, stop;

    //Compute capability, important for default compilation mode
    int ccMajor;
    int ccMinor;
    int defaultComputeMode;


    std::string logPrepend;


  public:

     int multiProcessorCount;   //Required to configure parts of the code

    context() {
      hContext_flag     = false;
      hInit_flag        = false;
      logfile_flag      = false;
      disable_timing    = false;

      hInit_flag        = true;
    }
    ~context() {
      if (hContext_flag )
      {
        CU_SAFE_CALL(hipDeviceReset());
      }
    }

    int getComputeCapability() const { return 100 * ccMajor + 10 * ccMinor; }
    int getComputeCapabilityMajor() const {return ccMajor;}
    int getComputeCapabilityMinor() const {return ccMajor;}


    int create(std::ostream &log, bool disableTiming = false)
    {
      disable_timing = disableTiming;
      logfile_flag   = true;
      logFile        = &log;
      logID          = 0;
      return create(disable_timing);
    }


    int create(bool disableT = false) {
      assert(hInit_flag);

      disable_timing = disableT;

      LOG("Creating CUDA context \n");

      // Get number of devices supporting CUDA
      ciDeviceCount = 0;
      CU_SAFE_CALL(hipGetDeviceCount(&ciDeviceCount));

      LOG("Found %d suitable devices: \n",ciDeviceCount);
      for(int i=0; i < ciDeviceCount; i++)
      {
        hipDeviceProp_t deviceProp;
        hipGetDeviceProperties(&deviceProp, i);
        LOG(" %d: %s\n",i, deviceProp.name);
      }
      return ciDeviceCount;
    }

    void createQueue(size_t dev = 0, int ctxCreateFlags = 0)
    {
      //use hipDeviceMapHost as flag for zero-copy memory
      //Here we finally create and assign the context to a device
      assert(!hContext_flag);
      assert(hInit_flag);
      this->dev = dev;
      assert((int)dev < ciDeviceCount);

      LOG("Trying to use device: %d ...", (int)dev);
      //Faster and async kernel launches when using large size arrays of local memory
      //ctxCreateFlags |= hipDeviceLmemResizeToMax;

      //Create the context for this device handle
      //CU_SAFE_CALL(hipCtxCreate(&hContext, ctxCreateFlags, hDevice));

      int res = hipSetDevice((int)dev);
      if(res != hipSuccess)
      {
        LOG("failed (error #: %d), now trying all devices starting at 0 \n", res);

	      for(int i=0; i < ciDeviceCount; i++)
        {
	        LOG("Trying device: %d  ...", i);
          if(hipSetDevice(i) != hipSuccess)
          {
            LOG("failed!\n");
            if(i+1 == ciDeviceCount)
            {
              LOG("All devices failed, exit! \n");
              ::exit(0);
            }
          }
          else
          {
            LOG("success! \n");
            this->dev = i;
            break;
          }
        }
      }
      else
      {
        LOG("success!\n");
      }

      hipDeviceProp_t deviceProp;
      CU_SAFE_CALL(hipGetDeviceProperties(&deviceProp, (int)dev));
      //Get the number of multiprocessors of the device
      multiProcessorCount = deviceProp.multiProcessorCount;
      ccMajor = deviceProp.major;
      ccMinor = deviceProp.minor;

      hContext_flag = true;
    }

    void startTiming(hipStream_t stream=0)
    {
      if(disable_timing) return;
      int eventflags =  hipEventDefault;
      CU_SAFE_CALL(hipEventCreateWithFlags(&start, eventflags));
      CU_SAFE_CALL(hipEventCreateWithFlags(&stop, eventflags));
      CU_SAFE_CALL(hipEventRecord(start, stream));
    }

    //Text and ID to be printed with the log message on screen / in the file
    void stopTiming(const char *text, int type = -1, hipStream_t stream=0)
    {
      if(disable_timing) return;

      CU_SAFE_CALL(hipEventRecord(stop, stream));
      CU_SAFE_CALL(hipEventSynchronize(stop));
      float time;
      CU_SAFE_CALL(hipEventElapsedTime(&time, start, stop));
      CU_SAFE_CALL(hipEventDestroy(start));
      CU_SAFE_CALL(hipEventDestroy(stop));

      LOG("%s took:\t%f\t millisecond\n", text, time);

      if(logfile_flag)
      {
        (*logFile) << logPrepend << logID++ << "\t"  << type << "\t" << text << "\t" << time << std::endl;
      }
    }

    void writeLogEvent(const char *text)
    {
      if(disable_timing) return;
      if(logfile_flag)
      {
        (*logFile) << logPrepend << text;
      }
    }

    void setLogPreamble(std::string text)
    {
      logPrepend = text;
    }

    //This function returns the currently recorded log-data and will clear the log-buffer
    std::string getLogData()
    {
      std::stringstream temp;
      temp << logFile->rdbuf();
      return temp.str();
    }
  };


  ////////////////////////////////////////

    //Class to handle streams / queues

  class dev_stream
  {
    private:
      hipStream_t stream;

    public:
      dev_stream(unsigned int flags = 0)
      {
        createStream(flags);
      }


      void createStream(unsigned int flags = 0)
      {
         CU_SAFE_CALL(hipStreamCreate(&stream));
      }

      void destroyStream()
      {
        CU_SAFE_CALL(hipStreamDestroy(stream));
      }

      void sync()
      {
        CU_SAFE_CALL(hipStreamSynchronize(stream));
      }

      bool isFinished()
      {
        hipError_t res = hipStreamQuery(stream);
        if(res == hipSuccess) return true;
        if(res == hipErrorNotReady) return false;

        //Some other result
        CU_SAFE_CALL(res);
        return false;
      }

      hipStream_t s()
      {
        return stream;
      }


      ~dev_stream() {
      destroyStream();
    }
  };


  ///////////////////////

  class base_mem
  {
    public:
    //Memory usage counters
    static long long currentMemUsage;
    static long long maxMemUsage;

    void increaseMemUsage(int bytes)
    {
      currentMemUsage +=  bytes;

      if(currentMemUsage > maxMemUsage)
        maxMemUsage = currentMemUsage;
    }

    void decreaseMemUsage(int bytes)
    {
      currentMemUsage -=  bytes;
    }

    static void printMemUsage()
    {
      LOG("Current usage: %lld bytes ( %lld MB) \n", currentMemUsage, currentMemUsage / (1024*1024));
      LOG("Maximum usage: %lld bytes ( %lld MB) \n", maxMemUsage, maxMemUsage / (1024*1024));

      size_t free, total;
      hipMemGetInfo(&free, &total);
      LOG("Build-in usage: free: %ld bytes ( %ld MB , total: %ld) \n", free, free / (1024*1024), total / (1024*1024));

    }

    static long long getMaxMemUsage()
    {
      return maxMemUsage;
    }

  };


  template<class T>
  class dev_mem : base_mem {
  protected:

    size_t size;
    T           *hDeviceMem;
    T           *host_ptr;
    void        *DeviceMemPtr;

    hipEvent_t  asyncCopyEvent = nullptr;

    bool pinned_mem, flags;
    bool hDeviceMem_flag;
    bool childMemory; //Indicates that this is a shared buffer that will be freed by a parent

    bool eventSet;

    void cuda_free() {
      if(childMemory) //Only free if we are NOT a child
      {
        return;
      }

      if (hDeviceMem_flag)
      {
    	  assert(size > 0);
    	  (hipFree(hDeviceMem));
    	  decreaseMemUsage(size*sizeof(T));

    	  if(pinned_mem){
    		  (hipHostFree((void*)host_ptr));
    	  } else {
    		  free(host_ptr);
    	  }
          hDeviceMem_flag = false;
      }
    } //cuda_free

  public:


    ///////// Constructors

    dev_mem(): hDeviceMem(NULL), DeviceMemPtr(NULL), flags(0){
      size              = 0;
      pinned_mem        = false;
      hDeviceMem_flag   = false;
      host_ptr          = NULL;
      childMemory       = false;

      CU_SAFE_CALL(hipEventCreate(&asyncCopyEvent));
      eventSet = true;
    }

    void free_mem()
    {
      cuda_free();
    }

    //////// Destructor

    ~dev_mem() {
      if(eventSet) hipEventDestroy(asyncCopyEvent);
      cuda_free();
    }



    ///////////
    //Return the number of elements (of type uint) to be padded
    //to get to the correct address boundary
    static int getGlobalMemAllignmentPadding(int n)
    {
      const int allignBoundary = 128*sizeof(uint); //CC 2.X and 3.X ,128 bytes

      int offset = 0;
      //Compute the number of bytes
      offset = n*sizeof(uint);
      //Compute number of allignBoundary byte blocks
      offset = (offset / allignBoundary) + (((offset % allignBoundary) > 0) ? 1 : 0);
      //Compute the number of bytes padded / offset
      offset = (offset * allignBoundary) - n*sizeof(uint);
      //Back to the actual number of elements
      offset = offset / sizeof(uint);

      return offset;
    }

    //Get the reference of memory allocated by another piece of memory
    //sourcemem -> The memory buffer that acts as the parent
    //n         -> The number of elements of type T for the child
    //offset    -> The offset, this *MUST* be the return value of previous calls
    //             to this function to ensure alignment. Note this is the number
    //             of elements in type uint
    int  cmalloc_copy(dev_mem<uint> &sourcemem, const size_t n, const int offset)
    {
      //The properties
      this->pinned_mem  = sourcemem.get_pinned();
      this->flags       = sourcemem.get_flags();
      this->childMemory = true;
      this->size 		= n;


      void* ParentHost_ptr = &sourcemem[offset];
      host_ptr = (T*) ((char*)ParentHost_ptr);


      /* jbedorf: fixed so address is correct, casting before adding gives a different
       * result. So divide by size of object */
      void *cudaMem     = sourcemem.get_devMem();
      hDeviceMem        = (T*)cudaMem + ((offset*sizeof(uint)) / sizeof(T));
      DeviceMemPtr      = (void*)(size_t)(hDeviceMem);
      hDeviceMem_flag   = true;

      //Compute the alignment
      int currentOffset = offset + ((n*sizeof(T)) / sizeof(uint));
      int padding       = getGlobalMemAllignmentPadding(currentOffset);

      // The "TODO for safety" that used to sit here, now implemented -- it is load-bearing.
      // generalBuffer1 is a shared scratch arena carved up by several unrelated consumers
      // (build()'s per-level loop, sort_bodies, dataReorder, compute_properties, the energy
      // reduction), and NONE of them bounds-checked. Its size merely happened to be adequate
      // because the tree-node scratch inside it was sized at a very pessimistic 2*N -- so when
      // that was reduced to TreeAllocFactor*N, sort_bodies (which needs ~9*N uints, entirely
      // independently of the node count) silently ran off the end and the only symptom was a
      // bare "Memory access fault by GPU node-1" with no indication of which buffer or why.
      // Fail here, naming the consumer, instead.
      const size_t endUint = (size_t)currentOffset + (size_t)padding;
      if (endUint > sourcemem.get_size())
      {
        fprintf(stderr,
          "\nFATAL: scratch arena overflow while sub-allocating %zu element(s) of size %zu.\n"
          "  arena capacity : %zu uints\n"
          "  required       : %zu uints (offset %d + this allocation + alignment)\n"
          "This is a shared scratch buffer; the consumer that overflowed it is the caller of\n"
          "cmalloc_copy immediately above this line in the backtrace. If the run is large,\n"
          "raise TreeAllocFactor in the parameter file; otherwise the arena sizing in\n"
          "octree::allocateParticleMemory / reallocateParticleMemory needs to account for this\n"
          "consumer explicitly.\n\n",
          (size_t)n, sizeof(T), (size_t)sourcemem.get_size(), endUint, offset);
        exit(1);
      }

      return currentOffset + padding;
    }

    // PLAN 4.1 deliverable 4: poison freshly allocated DEVICE memory when GADGET_HIP_POISON=1,
    // so a read of never-written memory yields an immediate NaN instead of a plausible number.
    // 0x7FC00000 is a quiet NaN as float32 (and an obviously-wrong 2143289344 as int32).
    //
    // Deliberately NOT applied to ccalloc(). Several buffers depend on documented zero-init, and
    // at least one of those zeros is load-bearing DYNAMICS: bodies_acc0 starts zeroed, and
    // groupMaxAcc == 0 is precisely how the Springel MAC detects "no prior acceleration yet" and
    // falls back to the geometric criterion for the bootstrap iteration
    // (dev_approximate_gravity_warp_new.cu, Ticket 23's fix). Poisoning that would change the
    // trajectory rather than expose a defect -- the opposite of what this is for.
    static int poisonLevel()
    {
      static int on = -1;
      if (on < 0) { const char *e = getenv("GADGET_HIP_POISON"); on = e ? atoi(e) : 0; }
      return on;
    }
    static bool poisonEnabled() { return poisonLevel() != 0; }

    // Level 2 additionally poisons ccalloc(), which is NOT safe -- it deliberately destroys the
    // zero-init that bodies_acc0's MAC bootstrap depends on. It exists purely as a positive
    // control: if level 2 does not change results, the poison is not actually being written and a
    // clean level-1 run proves nothing. Never use it for a real run.
    static bool poisonZeroInit() { return poisonLevel() >= 2; }

    void poisonDeviceRange(size_t firstElem, size_t nElem)
    {
      if (!poisonEnabled() || nElem == 0 || !hDeviceMem_flag) return;
      const size_t bytes = nElem * sizeof(T);
      if (bytes % 4) return;                       // pattern is 32-bit; skip odd-sized types
      char *base = (char*) hDeviceMem + firstElem * sizeof(T);
      hipError_t perr = hipMemsetD32((hipDeviceptr_t) base, (int) 0x7FC00000, bytes / 4);

      // One-shot self-test. A poison run that reports no NaNs proves nothing unless the poison was
      // actually written, so verify once that the pattern lands and say so on stderr. Without this
      // the whole diagnostic is unfalsifiable.
      static bool verified = false;
      if (!verified)
      {
        verified = true;
        unsigned int probe = 0;
        hipError_t cerr = hipMemcpy(&probe, base, sizeof(probe), hipMemcpyDeviceToHost);
        fprintf(stderr, "[POISON] self-test: memset=%s copy=%s first word=0x%08x %s\n",
                hipGetErrorString(perr), hipGetErrorString(cerr), probe,
                probe == 0x7FC00000u ? "OK (pattern present)" : "FAILED (pattern NOT written)");
      }
    }

    void cmalloc(size_t n, bool pinned = false, int flags = 0)
    {
      this->pinned_mem = pinned;
      this->flags = (flags == 0) ? false : true;
      if (size > 0) cuda_free();
      size = n;

      if(pinned_mem){
        CU_SAFE_CALL(hipHostMalloc((T**)&host_ptr, size*sizeof(T)));}
      else{
        host_ptr = (T*)malloc(size*sizeof(T));}

      //fprintf(stderr,"host ptr: %p new: %d \n",host_ptr, size);

      CU_SAFE_CALL(hipMalloc((T**)&hDeviceMem, size*sizeof(T)));
      increaseMemUsage(size*sizeof(T));
      DeviceMemPtr = (void*)(size_t)hDeviceMem;

      hDeviceMem_flag = true;
      poisonDeviceRange(0, (size_t) size);
    }

    void ccalloc(size_t n, bool pinned = false, int flags = 0) {
      this->pinned_mem = pinned;
      this->flags = (flags == 0) ? false : true;
      if (size > 0) cuda_free();
      size = n;

      if(pinned_mem)
        hipHostMalloc((T**)&host_ptr, size*sizeof(T));
      else
        host_ptr = (T*)calloc(size, sizeof(T));


      CU_SAFE_CALL(hipMalloc((T**)&hDeviceMem, size*sizeof(T)));
      CU_SAFE_CALL(hipMemset((void*)hDeviceMem, 0, size*sizeof(T)));
      increaseMemUsage(size*sizeof(T));
      DeviceMemPtr = (void*)(size_t)hDeviceMem;

      hDeviceMem_flag = true;
          if (poisonZeroInit()) poisonDeviceRange(0, (size_t) size);
}

     //Set reduce to false to not reduce the size, to speed up pinned memory buffers
    void cresize(size_t n, bool reduce = true)
    {
      if(size == n)     //No need if we are already at the correct size
        return;

      if(size > n && reduce == false) //Do not make the memory size smaller
      {
        return;
      }

//       d2h();    //Get datafrom the device

      if(pinned_mem)
      {
        //No realloc function so do it by hand
        T *tmp_ptr;
        CU_SAFE_CALL(hipHostMalloc((T**)&tmp_ptr, n*sizeof(T)));
        //Copy old content to newly allocated mem
        size_t tmpSize = std::min(size,n);

        //Copy the old data to the new pointer and free the old location
        memcpy (((void*) tmp_ptr), ((void*) host_ptr), tmpSize*sizeof(T));
        CU_SAFE_CALL(hipHostFree((void*)host_ptr));
        host_ptr = tmp_ptr;
      }
      else
      {
        //Resizes the current array
        //New size is smaller, don't do anything with the allocated memory
        host_ptr = (T*)realloc(host_ptr, n*sizeof(T));

        //On my machine realloc somtimes failed, retrying seems to fix it (JB)
        int count = 0;
        while(host_ptr == nullptr)
        {
            usleep(10); count++;
            fprintf(stderr,"Realloc failed, try again, size-old: %d new: %d attempt: %d\n", size,n, count);
            host_ptr = (T*)realloc(host_ptr, n*sizeof(T));
            if(host_ptr == nullptr && count > 10) {fprintf(stderr,"Realloc failed too often, exit\n"); exit(0); }
        }
      }

      //This version compared to the commented out one above, first allocates
      //new memory and then copies the old one in the new one and free's the old one
      T *hDeviceMemNew;
      CU_SAFE_CALL(hipMalloc((T**)&hDeviceMemNew, n*sizeof(T)));
      increaseMemUsage(n*sizeof(T));
      size_t nToCopy = std::min(size, n); //Do not copy more than we have memory
      CU_SAFE_CALL(hipMemcpy(hDeviceMemNew, hDeviceMem, nToCopy*sizeof(T), hipMemcpyDeviceToDevice ));
      //Now free the old memory
      CU_SAFE_CALL(hipFree(hDeviceMem));
      decreaseMemUsage(size*sizeof(T));
      hDeviceMem = hDeviceMemNew;
      DeviceMemPtr = (void*)(size_t)hDeviceMem;
      size = n;
    }

    //Set reduce to false to not reduce the size, to speed up pinned memory buffers
    //This one does not copy/preserve memory, its just a free and realloc no memory cpy
   void cresize_nocpy(size_t n, bool reduce = true)
   {
     if(size == n)     //No need if we are already at the correct size
       return;

     if(size > n && reduce == false) //Do not make the memory size smaller
     {
       return;
     }

     if(pinned_mem)
     {
       //No realloc function so do it by hand
       CU_SAFE_CALL(hipHostFree((void*)host_ptr));
       CU_SAFE_CALL(hipHostMalloc((T**)&host_ptr, n*sizeof(T)));
     }
     else
     {
       //Resizes the current array
       free(host_ptr);
       host_ptr = (T*)malloc(n*sizeof(T));
       //New size is smaller, don't do anything with the allocated memory
       //host_ptr = (T*)realloc(host_ptr, n*sizeof(T));
     }

     CU_SAFE_CALL(hipFree(hDeviceMem));
     decreaseMemUsage(size*sizeof(T));
     CU_SAFE_CALL(hipMalloc((T**)&hDeviceMem, n*sizeof(T)));
     increaseMemUsage(n*sizeof(T));

     DeviceMemPtr = (void*)(size_t)hDeviceMem;
     size = n;
   }


    //Set the memory to zero
    void zeroMem()
    {
      assert(hDeviceMem_flag);

      if(size > 0) {
        memset(host_ptr, 0, size*sizeof(T));
        CU_SAFE_CALL(hipMemset((void*)hDeviceMem, 0, size*sizeof(T)));
      }
    }

    void zeroMemGPUAsync(hipStream_t stream)
    {
      assert(hDeviceMem_flag);
      CU_SAFE_CALL(hipMemsetAsync((void*)hDeviceMem, 0, size*sizeof(T), stream));
    }

    ///////////

    //////////////


    void d2h(bool OCL_BLOCKING = true, hipStream_t stream = 0)   {
      d2h(size, OCL_BLOCKING, stream);
    }

    //D2h that only copies a certain number of items to the host
    void d2h(int number, bool OCL_BLOCKING = true, hipStream_t stream = 0)   {
      assert(hDeviceMem_flag);

      if(number == 0) return;

      assert(size > 0);

      if(OCL_BLOCKING)
      {
        CU_SAFE_CALL(hipMemcpy(&host_ptr[0], hDeviceMem, number*sizeof(T),hipMemcpyDeviceToHost));
      }
      else
      {
        //Async copy, ONLY works for page-locked memory therefore default parameter is blocking.
        assert(pinned_mem);
        CU_SAFE_CALL(hipMemcpyAsync(&host_ptr[0], hDeviceMem, number*sizeof(T),hipMemcpyDeviceToHost, stream));
        CU_SAFE_CALL(hipEventRecord(asyncCopyEvent, stream));
      }
    }

    //Copy to a specified buffer
    void d2h(int number, void* dst, bool OCL_BLOCKING = true, hipStream_t stream = 0)   {
      assert(hDeviceMem_flag);

      if(number == 0) return;

      assert(size > 0);

      if(OCL_BLOCKING)
      {
        CU_SAFE_CALL(hipMemcpy(dst, hDeviceMem, number*sizeof(T),hipMemcpyDeviceToHost));
      }
      else
      {
        //Async copy, ONLY works for page-locked memory therefore default parameter is blocking.
        assert(pinned_mem);
        CU_SAFE_CALL(hipMemcpyAsync(&host_ptr[0], hDeviceMem, number*sizeof(T),hipMemcpyDeviceToHost, stream));
        CU_SAFE_CALL(hipEventRecord(asyncCopyEvent, stream));
      }
    }


    void h2d(bool OCL_BLOCKING  = true, hipStream_t stream = 0)   {
      assert(hDeviceMem_flag);
      assert(size > 0);
      //if (flags & CL_MEM_USE_HOST_PTR == 0) return;
      if(OCL_BLOCKING)
      {
        CU_SAFE_CALL(hipMemcpy(hDeviceMem, &host_ptr[0], size*sizeof(T),hipMemcpyHostToDevice ));
      }
      else
      {
        //Async copy, ONLY works for page-locked memory therefore default parameter
        //is blocking.
        assert(pinned_mem);
        CU_SAFE_CALL(hipMemcpyAsync(hDeviceMem, host_ptr, size*sizeof(T),hipMemcpyHostToDevice , stream));
        CU_SAFE_CALL(hipEventRecord(asyncCopyEvent, stream));
      }
    }

    //D2h that only copies a certain number of items to the host
    void h2d(int number, bool OCL_BLOCKING = true, hipStream_t stream = 0)   {
      assert(hDeviceMem_flag);
      assert(size > 0);

      if(number == 0) return;

      if(OCL_BLOCKING)
      {
        CU_SAFE_CALL(hipMemcpy(hDeviceMem, &host_ptr[0], number*sizeof(T),hipMemcpyHostToDevice));
      }
      else
      {
        //Async copy, ONLY works for page-locked memory therefore default parameter is blocking.
        assert(pinned_mem);
        CU_SAFE_CALL(hipMemcpyAsync(hDeviceMem, host_ptr, number*sizeof(T),hipMemcpyHostToDevice, stream));
        CU_SAFE_CALL(hipEventRecord(asyncCopyEvent, stream));

      }
    }

    void waitForCopyEvent() {
      CU_SAFE_CALL(hipEventSynchronize(asyncCopyEvent));
    }

    void streamWaitForCopyEvent(my_dev::dev_stream &stream) {
      CU_SAFE_CALL(hipStreamWaitEvent(stream.s(), asyncCopyEvent, 0));
    }

    //JB: Modified this so that it copies a device buffer to an other device
    //buffer, and the host buffer to the other host buffer
    void copy(dev_mem &src_buffer, int n, bool OCL_BLOCKING = true)   {
      assert(hDeviceMem_flag);
      if (size < n) {
        cuda_free();
        cmalloc(n, flags);
        size = n;
      }

      //Copy on the device
      CU_SAFE_CALL(hipMemcpy(hDeviceMem, src_buffer.d(), n*sizeof(T), hipMemcpyDeviceToDevice));
      //Copy on the host
      memcpy (((void*) &host_ptr[0]), ((void*) &src_buffer[0]), n*sizeof(T));
    }

    void copy_devonly(dev_mem &src_buffer, int n, int offset = 0)
    {
      //Copy data from src_buffer in our own device buffer
      CU_SAFE_CALL(hipMemcpy((void*)(size_t)(hDeviceMem + offset), src_buffer.d(), n*sizeof(T), hipMemcpyDeviceToDevice));
    }
    void copy_devonly(T* src, const int n, int offset = 0)
    {
      CU_SAFE_CALL(hipMemcpy((void*)(size_t)(hDeviceMem + offset), src, sizeof(T)*n, hipMemcpyDeviceToDevice));
    }
    void copy_devonly_async(dev_mem &src_buffer, const int n, int offset = 0, hipStream_t stream = 0)
    {
      CU_SAFE_CALL(hipMemcpyAsync((void*)(size_t)(hDeviceMem + offset), src_buffer.d(), sizeof(T)*n, hipMemcpyDeviceToDevice, stream));
    }




    /////////

    T& operator[] (int i){ return host_ptr[i]; }

    void*  get_devMem() {return (void*)hDeviceMem;}
    void*  d()          {return (void*)hDeviceMem;}

#ifdef __HIPCC__
    thrust::device_ptr<T> thrustPtr()
    {
        thrust::device_ptr<T> temp = thrust::device_pointer_cast(hDeviceMem);
        return temp;
    }
#endif

    T* raw_p() {return  hDeviceMem;}

    void*   p() {return &hDeviceMem;}
    void*   a(int offset)
    {
      //Calculate the new memory offset
      //This fails because pointer is of type T and not of type void :-)
      //return (void*)(size_t)(hDeviceMem + offset*sizeof(T));
      //This works
      return (void*)(size_t)(hDeviceMem + offset);
    }

    size_t get_size(){return size;}
    bool get_pinned(){return pinned_mem;}
    bool get_flags(){return flags;}
  };     // end of class dev_mem

  ////////////////////


  class kernel {
  protected:
    char       *hKernelFilename;
    char       *hKernelName;
public:
    const void *hKernelPointer;
protected:
    std::vector<size_t> hGlobalWork;
    std::vector<size_t> hLocalWork;

    std::vector<void*> argumentList;
    std::vector<int> argumentOffset;

    //Kernel argument stuff
    #define MAXKERNELARGUMENTS 128
    typedef struct kernelArg
    {
      int alignment;    //Alignment of the variable type
      int sizeoftyp;    //Size of the variable type
      void* ptr;        //The pointer to the memory
      int size;         //The number of elements (incase of shared memory)
    } kernelArg;

    std::vector<kernelArg>      kernelArguments;
    std::vector<void*>          kArguments;
    int argSize=0;

    bool context_flag;
    bool kernel_flag;
    bool program_flag;
    bool work_flag;

    size_t sharedMemorySize;
    int paramOffset;

  public:

    kernel() {
      hKernelName     = (char*)malloc(256);
      hKernelFilename = (char*)malloc(1024);
      hGlobalWork.clear();
      hLocalWork.clear();

      context_flag = false;
      kernel_flag  = false;
      program_flag = false;
      work_flag    = false;

      sharedMemorySize = 0;
      paramOffset      = 0;



      //Kernel argument stuff
      kArguments.resize(MAXKERNELARGUMENTS);
      kernelArguments.resize(MAXKERNELARGUMENTS);
      kernelArg argTemp;
      argTemp.alignment = -1;   argTemp.sizeoftyp = -1;
      argTemp.ptr       = NULL; argTemp.size      = -1;
      kernelArguments.assign(MAXKERNELARGUMENTS, argTemp);

    }
    ~kernel() {
      free(hKernelName);
      free(hKernelFilename);
    }

    kernel(class context &c) {
      hKernelName   = (char*)malloc(256);
      hKernelFilename = (char*)malloc(1024);
      hGlobalWork.clear();
      hLocalWork.clear();

      context_flag = false;
      kernel_flag  = false;
      program_flag = false;
      work_flag    = false;

      //Kernel argument stuff
      kernelArguments.resize(MAXKERNELARGUMENTS);
      kernelArg argTemp;
      argTemp.alignment = -1;   argTemp.sizeoftyp = -1;
      argTemp.ptr       = NULL; argTemp.size      = -1;
      kernelArguments.assign(MAXKERNELARGUMENTS, argTemp);

      sharedMemorySize = 0;
      paramOffset      = 0;
      setContext(c);
    }

    ////////////

    void setContext(class context &c) {
      assert(!context_flag);
      context_flag     = true;
    }

    ////////////

    void load_source(const char *fileName, std::string &ptx_source)
    {
      //Keep for compatibility
    }

    void load_source(const char *kernel_name, const char *subfolder,
                     const char *compilerOptions = "",
                     int maxrregcount = -1,
                     int architecture = 0) {
      assert(context_flag);
      assert(!program_flag);

      //In runtime kept for compatibility

      //In cuda version we assume that the code is already compiled into ptx
      //so that the file loaded/specified is in fact a PTX file
      sprintf(hKernelFilename, "%s%s", subfolder, kernel_name);

      LOG("Loading source: %s ...", hKernelFilename);
      LOG("done!\n");

      program_flag = true;
    }

    void create(const char *kernel_name, const void *funcPointer) {
      //In runtime kept for compatibility
      program_flag = true;
      assert(program_flag);
      assert(!kernel_flag);
      assert(!context_flag);
      context_flag     = true;


      sprintf(hKernelName, kernel_name,"");

      LOG("Setting kernel: %s \n", kernel_name);

      hKernelPointer = funcPointer;

      kernel_flag = true;
    }

    void computeSharedMemorySize()
    {
      //We need to know size of shared memory before we set the arguments
      //so a quick look to only compute the shared memory size
      sharedMemorySize  = 0;
      //Loop over all set arguments and set them
      for(int i=0; i < MAXKERNELARGUMENTS; i++)
      {
        //First of all check if this argument has to be set or that we've finished already
        if(kernelArguments[i].size == -1)
          continue;

        //Now, check if this is a shared memory argument
        if(kernelArguments[i].ptr == NULL && kernelArguments[i].size > 1)
        {
          //Increase the shared memory size
          sharedMemorySize  += (size_t) (kernelArguments[i].size*kernelArguments[i].sizeoftyp);
        }
      }//end for

      if(sharedMemorySize > 0) { fprintf(stderr, "%s uses shared mem\n", hKernelName); }//exit(0);}
    }

    //Overwrite one of the previous set arguments with a new value
    void reset_arg(const int idx, void *arg) {kArguments[idx] = arg; }

    void set_argsb(int idx){}

    template<typename T, typename... Targs>
    void set_argsb(int idx, T arg, Targs... Fargs)
    {
        //Store T and continue with the remaining arguments
        kArguments[idx] = arg;
        set_argsb(++idx, Fargs...);
    }

    /*
     * First argument is the size of the shared-memory reservation in bytes
     * Each following argument will be a pointer to a value that will be send to the kernel
     *
     */

    template<typename T, typename... Targs>
    void set_args(const size_t shMemSize, T arg, Targs... Fargs)
    {
        sharedMemorySize = shMemSize;
        set_argsb(0, arg, Fargs...);
    }




    //'size'  is used for dynamic shared memory
    //Cuda does not have a function like clSetKernelArg
    //therefore we keep track of a vector with arguments
    //that will be processed when we launch the kernel
    template<class T>
    //void set_arg(unsigned int arg, void* ptr, int size = 1)  {
    void set_arg(unsigned int arg, void* ptr, int size = 1)  {
      assert(kernel_flag);

      //TODO have to check / think about if we want size default initialised
      //to 1 or to zero

      kernelArg tempArg;
      tempArg.alignment = __alignof(T);
      tempArg.sizeoftyp = sizeof(T);
      tempArg.ptr       = ptr;
      tempArg.size      = size;

      /* jbedorf, on 32bit Windows float4 reports as being
       * 4 byte aligned while it should be 16 byte. Force this
       * by manually setting alignment if the template argument
       * is a float4 . Note this is a first version work-around
       * for testing.
      */
      #if defined(_WIN32) && !defined(_WIN64)
        if(typeid(T) == typeid(float4&))
        {
          tempArg.alignment = 16;
        }
      #endif

      kernelArguments[arg] = tempArg;

      return;
    }

    void setWork(int items, int n_threads, int blocks = -1)
    {
      //Sets the number of blocks and threads based on the number of items
      //and number of threads per block.
      //TODO see if we can use the new kernel define for thread numbers?
      std::vector<size_t> localWork(2), globalWork(2);

      int nx, ny;

      if(blocks == -1)
      {
        //Calculate dynamic
        int ng = (items) / n_threads + 1;
        nx = (int)sqrt((double)ng);
        ny = (ng -1)/nx +  1;
      }
      else
      {
        //Specified number of blocks and numbers of threads make it a
        //2D grid if necessary
        if(blocks >= 65536)
        {
          nx = (int)sqrt((double)blocks);
          ny = (blocks -1)/nx +  1;
        }
        else
        {
          nx = blocks;
          ny = 1;
        }
      }

      globalWork[0] = nx*n_threads;  globalWork[1] = ny*1;
      localWork [0] = n_threads;     localWork[1]  = 1;
      setWork(globalWork, localWork);
    }


    void setWork(std::vector<size_t> global_work, std::vector<size_t> local_work) {
      assert(kernel_flag);
      assert(global_work.size() == local_work.size());

      hGlobalWork.resize(3);
      hLocalWork. resize(3);

      hLocalWork [0] = local_work[0];
      hLocalWork [1] = (local_work.size()  > 1) ? local_work[1] : 1;
      hLocalWork [2] = (local_work.size()  > 2) ? local_work[2] : 1;

      hGlobalWork[0] = global_work[0];
      hGlobalWork[1] = (global_work.size() > 1) ? global_work[1] : 1;

      //Since the values between CUDA and OpenCL differ:
      //Cuda is specific size of each block, while OpenCL
      //is the combined size of the lower blocks and this block
      //we have to divide the values

      hGlobalWork[0] /= hLocalWork[0];
      hGlobalWork[1] /= hLocalWork[1];
      hGlobalWork[2] /= hLocalWork[2];

      work_flag = true;
    }


    void execute2(hipStream_t hStream = 0, int* event = NULL) {
        dim3 gridDim;
        dim3 blockDim;
        hGlobalWork.resize(3);
        hLocalWork.resize(3);
        gridDim.x = (uint)hGlobalWork[0]; gridDim.y = (uint)hGlobalWork[1]; gridDim.z = 1;
        blockDim.x = (uint)hLocalWork[0]; blockDim.y = (uint)hLocalWork[1]; blockDim.z = (uint)hLocalWork[2];

        if(blockDim.x == 0 || gridDim.x == 0)
          return;

        CU_SAFE_CALL(hipLaunchKernel(reinterpret_cast<const void*>(hKernelPointer), gridDim, blockDim, &kArguments[0], (size_t)sharedMemorySize,hStream));
//        printf("Waiting on kernel: %s to finish...", hKernelName);
//        CU_SAFE_CALL(hipDeviceSynchronize());
//        printf("finished \n");
    }


    void printWorkSize(const char *s)
    {
      LOG("%sBlocks: (%ld, %ld, %ld) Threads: (%ld, %ld, %ld) \n", s,
              hGlobalWork[0], hGlobalWork[1], hGlobalWork[2],
              hLocalWork[0], hLocalWork[1], hLocalWork[2]);
    }

    void printWorkSize()
    {
      printWorkSize("");
    }

#if 0
    void execute(hipStream_t hStream = 0, int* event = NULL) {
      assert(kernel_flag);
      assert(work_flag);

      dim3 gridDim;
      dim3 blockDim;
      hGlobalWork.resize(3);
      hLocalWork.resize(3);
      gridDim.x = (uint)hGlobalWork[0]; gridDim.y = (uint)hGlobalWork[1]; gridDim.z = 1;
      blockDim.x = (uint)hLocalWork[0]; blockDim.y = (uint)hLocalWork[1]; blockDim.z = (uint)hLocalWork[2];

      if(blockDim.x == 0 || gridDim.x == 0)
        return;

      computeSharedMemorySize();  //Has to be done before rest of arguments
      CU_SAFE_CALL(hipConfigureCall(gridDim,
                                     blockDim,
                                     sharedMemorySize,
                                     hStream));
      completeArguments();


//     LOGF(stderr,"Waiting on kernel: %s to finish... ", hKernelName );
      fprintf(stderr,"Waiting on kernel: %s to finish... \n", hKernelName );
#if CUDART_VERSION < 5000
      CU_SAFE_CALL(hipLaunchByPtr((const char*)hKernelPointer));
#else
      CU_SAFE_CALL(hipLaunchByPtr(hKernelPointer));
      //CU_SAFE_CALL(hipLaunchByPtr((const char*)hKernelPointer));
#endif
//      CU_SAFE_CALL(hipDeviceSynchronize());       LOGF(stderr,"finished \n");

    }
#endif
    ////
  };

}     // end of namespace my_cuda


#endif // _MY_CUDA_H_


