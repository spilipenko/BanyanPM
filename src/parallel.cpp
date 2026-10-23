#include "hip/hip_runtime.h"
#include "octree.h"

//#define USE_MPI



inline int host_float_as_int(float val)
{
  union{float f; int i;} u; //__float_as_int
  u.f           = val;
  return u.i;
}

inline float host_int_as_float(int val)
{
  union{int i; float f;} itof; //__int_as_float
  itof.i           = val;
  return itof.f;
}


//SSE stuff for local tree-walk

void octree::mpiSetup()
{
  char processor_name[] = "Default";


  LOGF(   stderr, "Proc id: %d @ %s , total processes: %d (mpiInit) \n", procId, processor_name, nProcs);
  fprintf(stderr, "Proc id: %d @ %s , total processes: %d (mpiInit) \n", procId, processor_name, nProcs);


  currentRLow          = new double4[nProcs];
  currentRHigh         = new double4[nProcs];
  curSysState          = new sampleRadInfo[nProcs];
  globalGrpTreeCount   = new uint[nProcs];
  globalGrpTreeOffsets = new uint[nProcs];
}



//Utility functions





//end utility



//Main functions


//Functions related to domain decomposition



void octree::exchangeSamplesAndUpdateBoundarySFC(uint4 *sampleKeys2,    int  nSamples2,
    uint4 *globalSamples2, int  *nReceiveCnts2, int *nReceiveDpls2,
    int    totalCount2,   uint4 *parallelBoundaries, float lastExecTime,
    bool initialSetup)
{


#if 0 /* evghenii: disable 1D to enable 2D domain decomposition below */
  {
    //Send actual data
    MPI_Gatherv(&sampleKeys[0],    nSamples*sizeof(uint4), MPI_BYTE,
        &globalSamples[0], nReceiveCnts, nReceiveDpls, MPI_BYTE,
        0, mpiCommWorld);


    if(procId == 0)
    {
      //Sort the keys. Use stable_sort (merge sort) since the separate blocks are already
      //sorted. This is faster than std::sort (quicksort)
      //std::sort(allHashes, allHashes+totalNumberOfHashes, cmp_ph_key());
      double t00 = get_time();

#if 0 /* jb2404 */
      //std::stable_sort(globalSamples, globalSamples+totalCount, cmp_ph_key());
      __gnu_parallel::stable_sort(globalSamples, globalSamples+totalCount, cmp_ph_key());
#else
#if 0
      {
        const int BITS = 32*2;  /*  32*1 = 32 bit sort, 32*2 = 64 bit sort, 32*3 = 96 bit sort */
        typedef RadixSort<BITS> Radix;
        LOGF(stderr,"Boundary :: using %d-bit RadixSort\n", BITS);

        Radix radix(totalCount);
#if 0
        typedef typename Radix::key_t key_t;
#endif

        Radix::key_t *keys;
        posix_memalign((void**)&keys, 64, totalCount*sizeof(Radix::key_t));

#pragma omp parallel for
        for (int i = 0; i < totalCount; i++)
          keys[i] = Radix::key_t(globalSamples[i]);

        radix.sort(keys);

#pragma omp parallel for
        for (int i = 0; i < totalCount; i++)
          globalSamples[i] = keys[i].get_uint4();

        free(keys);

      }
#else
      {
        LOGF(stderr,"Boundary :: using %d-bit RadixSort\n", 64);
        unsigned long long *keys;
        posix_memalign((void**)&keys, 64, totalCount*sizeof(unsigned long long));

#pragma omp parallel for
        for (int i = 0; i < totalCount; i++)
        {
          const uint4 key = globalSamples[i];
          keys[i] =
            static_cast<unsigned long long>(key.y) | (static_cast<unsigned long long>(key.x) << 32);
        }

#if 0
        RadixSort64 r(totalCount);
        r.sort(keys);
#else
        __gnu_parallel::sort(keys, keys+totalCount);
#endif
#pragma omp parallel for
        for (int i = 0; i < totalCount; i++)
        {
          const unsigned long long key = keys[i];
          globalSamples[i] = (uint4){
            (uint)((key >> 32) & 0x00000000FFFFFFFF),
              (uint)((key      ) & 0x00000000FFFFFFFF),
              0,0};
        }
        free(keys);
      }
#endif

#endif
      LOGF(stderr,"Boundary took: %lg  Items: %d\n", get_time()-t00, totalCount);


      //Split the samples in equal parts to get the boundaries


      int procIdx   = 1;

      globalSamples[totalCount] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
      parallelBoundaries[0]     = make_uint4(0x0, 0x0, 0x0, 0x0);
      //Chop in equal sized parts
      for(int i=1; i < nProcs; i++)
      {
        int idx = (size_t(i)*size_t(totalCount))/size_t(nProcs);

        //jb2404
        if(iter == 0){
          if((i%1000) == 0) fprintf(stderr, " Boundary %d taken from : %d \n" ,i, idx);
          if(i >= nProcs-10) fprintf(stderr, " Boundary %d taken from : %d \n" ,i, idx);
        }

        parallelBoundaries[procIdx++] = globalSamples[idx];
      }
#if 0
      int perProc = totalCount / nProcs;
      int tempSum   = 0;
      for(int i=0; i < totalCount; i++)
      {
        tempSum += 1;
        if(tempSum >= perProc)
        {
          //LOGF(stderr, "Boundary at: %d\t%d %d %d %d \t %d \n",
          //              i, globalSamples[i+1].x,globalSamples[i+1].y,globalSamples[i+1].z,globalSamples[i+1].w, tempSum);
          tempSum = 0;
          parallelBoundaries[procIdx++] = globalSamples[i+1];
        }
      }//for totalNumberOfHashes
#endif


      //Force final boundary to be the highest possible key value
      parallelBoundaries[nProcs]  = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

      delete[] globalSamples;
    }

    //Send the boundaries to all processes
    MPI_Bcast(&parallelBoundaries[0], sizeof(uint4)*(nProcs+1), MPI_BYTE, 0, mpiCommWorld);
  }

  //End of 1D
#endif
}

//Uses one communication by storing data in one buffer and communicate required information,
//such as box-sizes and number of sample particles on this process. Nsample is set to 0
//since it is not used in this function/hash-method
void octree::sendCurrentRadiusInfo(real4 &rmin, real4 &rmax)
{

  sampleRadInfo curProcState;

  int nsample               = 0; //Place holder to just use same datastructure
  curProcState.nsample      = nsample;
  curProcState.rmin         = make_double4(rmin.x, rmin.y, rmin.z, rmin.w);
  curProcState.rmax         = make_double4(rmax.x, rmax.y, rmax.z, rmax.w);

  curSysState[0] = curProcState;

  rmin.x                 = (real)(currentRLow[0].x = curSysState[0].rmin.x);
  rmin.y                 = (real)(currentRLow[0].y = curSysState[0].rmin.y);
  rmin.z                 = (real)(currentRLow[0].z = curSysState[0].rmin.z);
  currentRLow[0].w = curSysState[0].rmin.w;

  rmax.x                 = (real)(currentRHigh[0].x = curSysState[0].rmax.x);
  rmax.y                 = (real)(currentRHigh[0].y = curSysState[0].rmax.y);
  rmax.z                 = (real)(currentRHigh[0].z = curSysState[0].rmax.z);
  currentRHigh[0].w = curSysState[0].rmax.w;

  for(int i=1; i < nProcs; i++)
  {
    rmin.x = std::min(rmin.x, (real)curSysState[i].rmin.x);
    rmin.y = std::min(rmin.y, (real)curSysState[i].rmin.y);
    rmin.z = std::min(rmin.z, (real)curSysState[i].rmin.z);

    rmax.x = std::max(rmax.x, (real)curSysState[i].rmax.x);
    rmax.y = std::max(rmax.y, (real)curSysState[i].rmax.y);
    rmax.z = std::max(rmax.z, (real)curSysState[i].rmax.z);

    currentRLow[i].x = curSysState[i].rmin.x;
    currentRLow[i].y = curSysState[i].rmin.y;
    currentRLow[i].z = curSysState[i].rmin.z;
    currentRLow[i].w = curSysState[i].rmin.w;

    currentRHigh[i].x = curSysState[i].rmax.x;
    currentRHigh[i].y = curSysState[i].rmax.y;
    currentRHigh[i].z = curSysState[i].rmax.z;
    currentRHigh[i].w = curSysState[i].rmax.w;
  }
}






//Function that uses the GPU to get a set of particles that have to be
//send to other processes
void octree::gpuRedistributeParticles_SFC(uint4 *boundaries)
{
} //End gpuRedistributeParticles

//Exchange particles with other processes
int octree::gpu_exchange_particles_with_overflow_check_SFC2(tree_structure &tree,
                                                            bodyStruct *particlesToSend,
                                                            int *nparticles, int *nsendDispls,
                                                            int *nreceive, int nToSend)
{

//  localTree.bodies_Ppos.d2h();
//  localTree.bodies_pos.d2h();

//  for(int i=0; i < tree.n; i++)
//  {
//	  LOGF(stderr,"CURRENT: %d %f %f  \t %f %f\n",
//			  i,
//			  localTree.bodies_pos[i].x, localTree.bodies_pos[i].y,
//			  localTree.bodies_Ppos[i].x, localTree.bodies_Ppos[i].y);
//  }


  return 0;
}



/********************************************************
 *                                                      *
 * Functions related to the LET Creation and Exchange   *
 *                                                      *
 ********************************************************/

//Broadcast the group-tree structure (used during the LET creation)
//First we gather the size, so we can create/allocate memory
//and then we broad-cast the final structure
//This basically is a sync-operation and therefore can be quite costly
//Maybe we want to do this in a separate thread
/****** EGABUROV ****/
void octree::sendCurrentInfoGrpTree()
{
}



//////////////////////////////////////////////////////
// ***** Local essential tree functions ************//
//////////////////////////////////////////////////////


//Sum the number of particles on all processes
void octree::mpiSumParticleCount(int numberOfParticles)
{
  nTotalFreq_ull = numberOfParticles;

  if(procId == 0) LOG("Total number of particles: %llu\n", nTotalFreq_ull);
}


