#include "hip/hip_runtime.h"
#include "octree.h"

#include "FileIO.h"

#include "SharedMemory.h"
#include "BonsaiSharedData.h"

#ifndef WIN32
    #include <sys/time.h>
#endif


#include "IDType.h"



/*********************************/
/*********************************/
/*********************************/



void octree::set_src_directory(string src_dir) {                                                                                                                                 
    this->src_directory = (char*)src_dir.c_str();                                                                                                                                
}   

double octree::get_time() {
#ifdef WIN32
  if (sysTimerFreq.QuadPart == 0)
  {
    return -1.0;
  }
  else
  {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<double>( (double)(c.QuadPart - sysTimerAtStart.QuadPart) / sysTimerFreq.QuadPart );
  }
#else
  struct timeval Tvalue;
  struct timezone dummy;
  
  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
#endif
}

int octree::getAllignmentOffset(int n)
{
  const int allignBoundary = 128*sizeof(uint); //Fermi,128 bytes 
  
  int offset = 0;
  //Compute the number of bytes  
  offset = n*sizeof(uint); 
  //Compute number of 256 byte blocks  
  offset = (offset / allignBoundary) + (((offset % allignBoundary) > 0) ? 1 : 0); 
  //Compute the number of bytes padded / offset 
  offset = (offset * allignBoundary) - n*sizeof(uint); 
  //Back to the actual number of elements
  offset = offset / sizeof(uint);   
  
  return offset;
}

int octree::getTextureAllignmentOffset(int n, int size)
{
    const int texBoundary = TEXTURE_BOUNDARY; //Fermi
  
    int textOffset = 0;
    //Compute the number of bytes  
    textOffset = n*size; 
    //Compute number of texBoundary byte blocks  
    textOffset = (textOffset / texBoundary) + (((textOffset % texBoundary) > 0) ? 1 : 0); 
    //Compute the number of bytes padded / offset 
    textOffset = (textOffset * texBoundary) - n*size; 
    //Back to the actual number of elements
    textOffset = textOffset / size; 
    
    return textOffset;
}   




/*********************************/
/******** Output functions  ******/
/*********************************/


/*
 * BonsaiIO output routines
 *
 */

using ShmQHeader = SharedMemoryServer<BonsaiSharedQuickHeader>;
using ShmQData   = SharedMemoryServer<BonsaiSharedQuickData>;
using ShmSHeader = SharedMemoryServer<BonsaiSharedSnapHeader>;
using ShmSData   = SharedMemoryServer<BonsaiSharedSnapData>;

static ShmQHeader *shmQHeader = NULL;
static ShmQData   *shmQData   = NULL;
static ShmSHeader *shmSHeader = NULL;
static ShmSData   *shmSData   = NULL;

/*
 *  Signal the IO process that we're finished (this is when tCurrent == -1) *
 *
 */

void octree::terminateIO() const
{
  {
    auto &header = *shmQHeader;
    header.acquireLock();
    header[0].tCurrent = -1;
    header[0].done_writing = false;
    header.releaseLock();
  }
  {
    auto &header = *shmSHeader;
    header.acquireLock();
    header[0].tCurrent = -1;
    header[0].done_writing = false;
    header.releaseLock();
  }
}




