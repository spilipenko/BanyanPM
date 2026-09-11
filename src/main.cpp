#include "hip/hip_runtime.h"
/*

Bonsai V2: A parallel GPU N-body gravitational Tree-code

(c) 2010-2012:
Jeroen Bedorf
Evghenii Gaburov
Simon Portegies Zwart

Leiden Observatory, Leiden University

http://castle.strw.leidenuniv.nl
http://github.com/treecode/Bonsai

*/

/*
 *
 * TODO
 * Close BonsaiIO on destruction to properly close open File handles
 * Fix the block time stepping
 * Add block time-stepping to the multi-GPU code
 *
 */


#ifdef WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
  #include <process.h>
  #define M_PI        3.14159265358979323846264338328

  #include <stdlib.h>
  #include <time.h>
  void srand48(const long seed)
  {
    srand(seed);
  }
  //JB This is not a proper work around but just to get things compiled...
  double drand48()
  {
    return double(rand())/RAND_MAX;
  }
#endif


#ifdef USE_MPI
  #include <omp.h>
  #include <mpi.h>
#endif

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <sys/time.h>
#include <omp.h>
#include "log.h"
#include "anyoption.h"
#include "gadget_params.h"
#include "gadget_snapshot.h"

#include <array>

#include <FileIO.h>
#include <ICGenerators.h>


#if ENABLE_LOG
  bool ENABLE_RUNTIME_LOG;
  bool PREPEND_RANK;
  int  PREPEND_RANK_PROCID;
  int  PREPEND_RANK_NPROCS;
#endif

using namespace std;

#include "../profiling/bonsai_timing.h"

int devID;
int renderDevID;

extern void initTimers()
{
#ifndef CUXTIMER_DISABLE
  // Set up the profiling timing info
  build_tree_init();
  compute_propertiesD_init();
  dev_approximate_gravity_init();
  parallel_init();
  sortKernels_init();
  timestep_init();
#endif
}

extern void displayTimers()
{
#ifndef CUXTIMER_DISABLE
  // Display all timing info on the way out
  build_tree_display();
  
  compute_propertiesD_display();
  //dev_approximate_gravity_display();
  //parallel_display();
  //sortKernels_display();
  //timestep_display();
#endif
}

#include "octree.h"
#include "pm.h"
#include "timestep_test.h"
#include "gadget_cosmology.h"
#include "gadget_driftfac.h"
#include <cmath>
#ifdef GADGET_HIP_HIGHRES
#include "pm_zoom.h"
#endif
#ifdef PERIODIC
#include "ewald_ref.h"
#endif

#ifdef USE_OPENGL
#include "renderloop.h"
#include <cuda_gl_interop.h>
#endif






double get_time_main()
{
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
}


/*
 * This thread watches if the simulation progresses or if it 'hangs'
 * this sometime happens (for unknown) reasons on large runs with file writing.
 * To prevent usage of precious compute cycles this thread kills the program 
 * after 360 seconds of no progress.
 */
void watchThread(octree *bonsai)
{
	double tCurrent = 0;
	do 
	{
	  std::this_thread::sleep_for(std::chrono::seconds(360));
	  double tNew = bonsai->getTime();
	  if(tNew != tCurrent)
	  {
		tCurrent = tNew;
	  }
	  else
	  {
		fprintf(stderr,"No progress detected for 360 seconds, the program will now exit!\n");
		::exit(0);
	  }
	}
	while(true);
}



//Buffers and flags used for the IO thread
volatile IOSharedData_t ioSharedData;

long long my_dev::base_mem::currentMemUsage;
long long my_dev::base_mem::maxMemUsage;

int bonsai_main(int argc, char** argv, MPI_Comm comm, int shrMemPID)
{
  my_dev::base_mem::currentMemUsage = 0;
  my_dev::base_mem::maxMemUsage     = 0;

  vector<real4>   bodyPositions;
  vector<real4>   bodyVelocities;
  vector<ullong>  bodyIDs;
  // Phase 5 ticket 03 (PLAN.md): per-particle Gadget-2 type (0-5), populated only when loading a
  // real Gadget-2 IC (ticket 02's gadget_snapshot_read); left empty otherwise, in which case
  // octree::allocateParticleMemory() below defaults every particle to type 1 (Halo/DM) -- the
  // right default for every existing dev/diagnostic IC generator (Plummer/Cube/Sphere/etc.),
  // none of which have any notion of Gadget-2 particle type.
  vector<int>     bodyTypes;

 
  float eps      = 0.05f;
  float theta    = 0.75f;
  float errTolForceAcc = ERR_TOL_FORCE_ACC;  //Springel MAC tolerance; only used if built with _MAC_SPRINGEL_
  float timeStep = 1.0f / 16.0f;
  float tEnd     = 1;
  int   iterEnd  = (1 << 30);
  devID          = 0;
  renderDevID    = 0;

  string fileName          =  "";
  // Phase 5 ticket 01 (PLAN.md): Gadget-2 `.param` file, additive alongside every existing CLI
  // flag below (those stay as dev/diagnostic entry points; --param is the real production path).
  string paramFile         =  "";
  // Declared here (not inside the option-parsing block below) because the Gadget-2 IC-loading
  // section further down in bonsai_main() -- outside that block's scope -- needs to read them too.
  GadgetParams gadgetParams;
  bool haveGadgetParams = false;
  bool haveGadgetIC     = false;
  // Phase 5 ticket 02 (PLAN.md): one-shot snapshot-format round-trip self-test, same idiom as
  // Phase 4's --pm-test-* diagnostics -- 0 means "don't run it", else the SnapFormat (1 or 2) to
  // test.
  int snapshotTestFormat = 0;
  // Phase 5 ticket 03 (PLAN.md): one-shot softening-kernel self-test, same idiom as above -- >0
  // means "run it" (the value itself is currently unused, matches other --*-test flags' shape).
  int softeningTestSamples = 0;
  // Phase 5 ticket 04 (PLAN.md): one-shot timestep-criterion self-test, same idiom.
  int timestepTestSamples = 0;
  // Phase 5 ticket 05 (PLAN.md): one-shot driftfac device-vs-host lookup self-test, same idiom.
  int driftfacTestSamples = 0;
  // Phase 5 ticket 06 (PLAN.md): one-shot zoom fine-grid Green's-function-kernel self-test, same
  // idiom (value unused, just needs to be nonzero). Only meaningful in a GADGET_HIP_HIGHRES build.
  int zoomKernelTestSamples = 0;
  // Phase 5 ticket 06 (PLAN.md): PLACEHIGHRESREGION zoom config -- mask is a Gadget-2-style
  // bitmask over particle types (bit t set => type t is "high-res", matches
  // pm_nonperiodic.c's own PLACEHIGHRESREGION semantics), 0 means "zoom disabled" (the default).
  unsigned int zoomMask    = 0;
  // Gadget-2's ENLARGEREGION is a Makefile -D that's commented out (Makefile:24, with 1.2 given only
  // as an example value) by default -- pm_nonperiodic.c:93-94 only multiplies TotalMeshSize when
  // ENLARGEREGION is actually #define'd, so the true stock default is "no enlargement" (1.0), not
  // 1.2. Verified directly against Gadget-2.0.7/Gadget2/{Makefile,pm_nonperiodic.c} rather than
  // assumed from prose.
  float        zoomEnlarge = 1.0f;
  string logFileName       = "gpuLog.log";
  string snapshotFile      = "snapshot_";
  std::string bonsaiFileName;
  float snapshotIter       = -1;
  float  remoDistance      = -1.0;
  int rebuild_tree_rate    = 1;
  std::string restartFilePath;           // --restart-file: where the restart state lives
  bool        resumeFromRestart = false; // --resume: resume from it instead of using the IC state
  int reduce_bodies_factor = 1;
  int reduce_dust_factor   = 1;
  string fullScreenMode    = "";
  bool direct     = false;
  bool fullscreen = false;
  bool displayFPS = false;
  bool diskmode   = false;
  bool stereo     = false;
  bool restartSim = false;

  float quickDump  = 0.0;
  float quickRatio = 0.1;
  bool  quickSync  = true;
  bool  useMPIIO = false;

#if ENABLE_LOG
  ENABLE_RUNTIME_LOG = false;
  PREPEND_RANK       = false;
#endif

#ifdef USE_OPENGL
	TstartGlow = 0.0;
	dTstartGlow = 1.0;
#endif
        
  double tStartupStart = get_time_main();       
  double tStartModel   = 0;
  double tEndModel     = 0;

  bool mpiRenderMode = false;
  
  

  int nPlummer  = -1;
  int nSphere   = -1;
  int nCube     = -1;
  int nMilkyWay = -1;
  int nMWfork   =  4;
  int galSeed   =  0;
  std::string taskVar;

#ifdef PERIODIC
  // Phase 4 (PLAN.md): periodic minimum-image tree walk + TreePM short-range/long-range split.
  // boxSize has no other source in this port (Bonsai itself has no notion of a fixed simulation
  // box, only a bounding box derived from the particle distribution) -- required whenever
  // PERIODIC is compiled in. Rcut/Asmth derived from it using Gadget-2's own compile-time
  // constants (ASMTH=1.25, RCUT=4.5, allvars.h:83,87), not independently tunable, matching
  // Gadget-2's own pm_init_periodic (GADGET2_NOTES.md).
  float boxSize = -1.0f;
#endif

#if defined(PMGRID) && !defined(PERIODIC)
  // Isolated (non-periodic) TreePM counterpart to boxSize above (LOG.md §34) -- physical size of
  // the doubled FFT grid the isolated PM solver + tree-side short-range table both need. Required
  // whenever PMGRID is compiled in without PERIODIC, matching how boxSize is required for
  // PERIODIC. Particles must lie within [0, meshSize/2) in each axis (the "occupied first half",
  // pm.h's PMIsolatedSolver doc comment) for correct isolated-boundary behavior.
  float meshSize = -1.0f;
#endif

  // Phase 4 (PLAN.md): standalone CIC mass-assignment correctness check, same idiom as the
  // GADGET_HIP_DUMP_ACC(_WARM) diagnostics used for Phase 2/3 validation -- a one-shot check that
  // exits immediately after running, not a real simulation mode. Deposits the just-generated IC's
  // masses onto a periodic gridSize^3 grid and compares the grid's total mass against the sum of
  // particle masses; any discrepancy beyond float roundoff means the CIC kernel is dropping or
  // double-counting mass.
  int pmTestCicGrid = 0;
  float pmTestCicBoxSize = 10.0f;
#ifdef PMGRID
  // Same one-shot diagnostic idiom, testing the rocFFT plumbing in two independent stages:
  // --pm-test-fft is a pure forward+inverse round trip on an arbitrary seeded grid (no Green's
  // function, no particles) -- isolates plan-creation/stride/normalization bugs from the actual
  // physics. --pm-test-poisson seeds a single analytic cosine density mode directly (bypassing
  // CIC entirely) and checks the full forward-FFT -> Green's-function -> inverse-FFT pipeline
  // against the closed-form expected potential for that mode -- isolates the Green's-function
  // formula from any CIC-deconvolution or particle-deposition concerns.
  int pmTestFftGrid = 0;
  int pmTestPoissonGrid = 0;
  float pmTestPoissonBoxSize = 10.0f;
  // --pm-test-force reuses the same seeded-cosine-mode density and FFT+Green's-function pipeline
  // as --pm-test-poisson, then exercises the finite-difference force-extraction kernel (exact
  // discrete closed form for a stencil applied to a discrete cosine, not an approximation -- see
  // the test's own comment) and the CIC force-interpolation gather at grid-aligned test points.
  int pmTestForceGrid = 0;
  float pmTestForceBoxSize = 10.0f;
  // --pm-test-sinewave: the first end-to-end test of the WIRED pipeline (pm_compute_forces_periodic
  // -- real CIC deposit with deconvolution on, not synthetic grid seeding). Displaces a uniform
  // particle lattice sinusoidally along x -- exactly the Zel'dovich approximation's linear
  // displacement ansatz -- and checks the resulting force against linear theory. Deliberately a
  // static single-step check (no time integration yet); the intended next step after this is the
  // actual Zel'dovich pancake test (time evolution compared against the ZA solution until shell
  // crossing), which this groundwork feeds into.
  int pmTestSineGrid = 0;
  float pmTestSineBoxSize = 10.0f;
  float pmTestSineAmplitude = 0.05f; // dimensionless, in grid-cell units (Sg in the derivation)
  // --pm-test-isolated-force: standalone correctness check for the non-periodic (isolated/
  // vacuum-boundary) PM solver (PLAN.md Phase 4's deferred item) -- independent of --pm-test-
  // treepm-force above, which is PERIODIC-only. Ground truth here is plain Newtonian G*m/r^2, no
  // periodic images to sum (much simpler than the periodic case's Ewald reference), since an
  // isolated system has none. Works in ANY PMGRID build, regardless of PERIODIC -- the isolated
  // solver is orthogonal to the tree's own periodic/non-periodic wrap choice (GADGET2_NOTES.md).
  int pmTestIsolatedSamples = 0;
  float pmTestIsolatedMeshSize = 10.0f;
#ifdef PERIODIC
  // --pm-test-treepm-force: the combined tree(short-range)+PM(long-range) correctness check
  // flagged as Phase 4's real remaining risk in PLAN.md/LOG.md §27 -- every individual PM/tree
  // piece was validated in isolation against internally-derived expectations, but never against
  // an INDEPENDENT reference. Uses the program's own already-established --boxsize/PMGRID (the
  // exact Asmth/Rcut/table a real run would use, set up earlier in main() via
  // pm_periodic_setup_gravity_kernel), not a separate test-specific grid/box. The argument is the
  // number of log-spaced separations to sample, not a grid size (unlike the other --pm-test-*
  // flags).
  int pmTestTreePMSamples = 0;
#endif
#if !defined(PERIODIC)
  // --pm-test-isolated-treepm-force: the isolated-boundary counterpart to --pm-test-treepm-force
  // above (LOG.md §34) -- uses the program's own already-established --meshsize/PMGRID (the exact
  // Asmth/Rcut/table a real non-periodic-TreePM run would use, set up earlier in main() via
  // pm_isolated_setup_gravity_kernel), not a separate test-specific mesh size. Ground truth is
  // plain Newtonian G*m/r^2 (no periodic images), simpler than periodic's Ewald reference.
  int pmTestIsolatedTreePMSamples = 0;
#endif
#endif
  // Phase 5 ticket 07 (PLAN.md): the zoom-aware counterpart to --pm-test-isolated-treepm-force
  // above -- confirms tree (using Rcut[1]/Asmth[1] for a high-res-typed target) + coarse PM + fine
  // (zoom) PM sum to the correct Newtonian G*m/r^2, i.e. the short/long handoff works cleanly at
  // Rcut[1] specifically, not just Rcut[0]. Declared UNCONDITIONALLY (matching
  // zoomKernelTestSamples's own precedent above) so the "nothing to simulate" gate further down
  // can reference it without extra #ifdef nesting; only meaningful/settable via CLI in a
  // GADGET_HIP_HIGHRES + PMGRID + !PERIODIC build.
  int zoomTreePMTestSamples = 0;
//#define TITAN_G
//#define SLURM_G
#ifdef TITAN_G
  //Works for both Titan and Piz Daint
  taskVar = std::string("PMI_FORK_RANK");
#elif defined SLURM_G
  taskVar = std::string("SLURM_PROCID");
#endif
	/************** beg - command line arguments ********/
#if 1
	{
		AnyOption opt;

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}


 		ADDUSAGE("Bonsai command line usage:");
 		ADDUSAGE("                  ");
		ADDUSAGE(" -h  --help             Prints this help ");
		ADDUSAGE(" -i  --infile #         Input snapshot filename in Tipsy format");
		ADDUSAGE("     --param #          Gadget-2 .param file (PLAN.md Phase 5 ticket 01) -- parsed "
		         "and validated; overrides --theta/--erracc/--boxsize/--tend when given");
		ADDUSAGE("     --snapshot-test #  Phase 5 ticket 02: one-shot Gadget-2 snapshot format "
		         "round-trip self-test, SnapFormat # (1 or 2); exits after printing PASS/FAIL");
		ADDUSAGE("     --softening-test # Phase 5 ticket 03: one-shot Gadget-2 spline-softening "
		         "kernel self-test (value unused, just needs to be nonzero); exits after printing "
		         "PASS/FAIL per case");
		ADDUSAGE("     --timestep-test #  Phase 5 ticket 04: one-shot Gadget-2 timestep-criterion "
		         "self-test (value unused, just needs to be nonzero); exits after printing "
		         "PASS/FAIL per case");
		ADDUSAGE("     --driftfac-test #  Phase 5 ticket 05: one-shot drift/grav-kick table "
		         "device-vs-host lookup self-test (value unused, just needs to be nonzero); "
		         "exits after printing PASS/FAIL per case");
		ADDUSAGE("     --zoom-mask #      Phase 5 ticket 06: PLACEHIGHRESREGION zoom -- Gadget-2 "
		         "particle-type bitmask (bit t set => type t is high-res), hex or decimal; 0 "
		         "disables zoom [requires a GADGET_HIP_HIGHRES build]");
		ADDUSAGE("     --zoom-enlarge #   Phase 5 ticket 06: ENLARGEREGION multiplier for the "
		         "high-res region's auto-sized bounding box [" << zoomEnlarge << "]");
		ADDUSAGE("     --zoom-kernel-test # Phase 5 ticket 06: one-shot fine-grid differential "
		         "Green's-function-kernel self-test (value unused, just needs to be nonzero); "
		         "exits after printing PASS/FAIL per case [requires a GADGET_HIP_HIGHRES build]");
		ADDUSAGE(" -f  --bonsaifile #     Input snapshot filename in Bonsai format [muse be used with --usempiio]");
		ADDUSAGE("     --restart          Let each process restart from a snapshot as specified by 'infile'");
		ADDUSAGE("     --logfile #        Log filename [" << logFileName << "]");
		ADDUSAGE("     --dev #            Device ID [" << devID << "]");
		ADDUSAGE("     --renderdev #      Rendering Device ID [" << renderDevID << "]");
		ADDUSAGE(" -t  --dt #             time step [" << timeStep << "]");
		ADDUSAGE(" -T  --tend #           N-body end time [" << tEnd << "]");
		ADDUSAGE(" -I  --iend #           N-body end iteration [" << iterEnd << "]");
		ADDUSAGE(" -e  --eps #            softening (will be squared) [" << eps << "]");
		ADDUSAGE(" -o  --theta #          opening angle (theta) [" <<theta << "]");
		ADDUSAGE("     --erracc #         Springel MAC ErrTolForceAcc, only if built with _MAC_SPRINGEL_ [" << errTolForceAcc << "]");
		ADDUSAGE("     --snapname #       snapshot base name (N-body time is appended in 000000 format) [" << snapshotFile << "]");
		ADDUSAGE("     --snapiter #       snapshot iteration (N-body time) [" << snapshotIter << "]");
		ADDUSAGE("     --quickdump  #     how ofter to dump quick output (N-body time) [" << quickDump << "]");
		ADDUSAGE("     --quickratio #     which fraction of data to dump (fraction) [" << quickRatio << "]");
        ADDUSAGE("     --noquicksync      disable syncing for quick dumping ");
        ADDUSAGE("     --usempiio         use MPI-IO [disabled]");
		ADDUSAGE("     --rmdist #         Particle removal distance (-1 to disable) [" << remoDistance << "]");
		ADDUSAGE(" -r  --rebuild #        rebuild tree every # steps [" << rebuild_tree_rate << "]");
		ADDUSAGE("     --restart-file #   restart-state path; written periodically and at end of run");
		ADDUSAGE("     --resume           resume from --restart-file instead of the IC's state");
		ADDUSAGE("     --debug            per-step developer probes (device pointers, buffer\n                        sizes, group-max traces); also GADGET_HIP_DEBUG=1");
		ADDUSAGE("     --reducebodies #   cut down bodies dataset by # factor ");
#ifdef USE_DUST
        ADDUSAGE("     --reducedust #     cut down dust dataset by # factor ");
#endif
#if ENABLE_LOG
    ADDUSAGE("     --log              enable logging ");
    ADDUSAGE("     --prepend-rank     prepend the MPI rank in front of the log-lines ");
#endif
    ADDUSAGE("     --direct           enable N^2 direct gravitation [" << (direct ? "on" : "off") << "]");
#ifdef PERIODIC
    ADDUSAGE("     --boxsize #        periodic box size (required, only if built with PERIODIC)");
#endif
#if defined(PMGRID) && !defined(PERIODIC)
    ADDUSAGE("     --meshsize #       isolated-TreePM doubled-grid physical size (required for this PMGRID-without-PERIODIC build)");
#endif
    ADDUSAGE("     --pm-test-cic #    Phase 4: one-shot CIC mass-assignment check on the generated IC, grid size # (e.g. 64); exits after printing the result");
    ADDUSAGE("     --pm-test-boxsize #  periodic box size for --pm-test-cic [" << pmTestCicBoxSize << "]");
#ifdef PMGRID
    ADDUSAGE("     --pm-test-fft #    Phase 4: one-shot rocFFT forward+inverse round-trip check, grid size #; exits after printing the result");
    ADDUSAGE("     --pm-test-poisson #  Phase 4: one-shot Green's-function Poisson-solve check against a seeded analytic mode, grid size #; exits after printing the result");
    ADDUSAGE("     --pm-test-poisson-boxsize #  periodic box size for --pm-test-poisson [" << pmTestPoissonBoxSize << "]");
    ADDUSAGE("     --pm-test-force #  Phase 4: one-shot finite-difference force + CIC interpolation check, grid size #; exits after printing the result");
    ADDUSAGE("     --pm-test-force-boxsize #  periodic box size for --pm-test-force [" << pmTestForceBoxSize << "]");
    ADDUSAGE("     --pm-test-sinewave #  Phase 4: end-to-end wired-pipeline check on a sinusoidally-displaced particle lattice, grid size #; exits after printing the result");
    ADDUSAGE("     --pm-test-sinewave-boxsize #  periodic box size for --pm-test-sinewave [" << pmTestSineBoxSize << "]");
    ADDUSAGE("     --pm-test-sinewave-amplitude #  dimensionless displacement amplitude (grid-cell units) for --pm-test-sinewave [" << pmTestSineAmplitude << "]");
    ADDUSAGE("     --pm-test-isolated-force #  Phase 4: non-periodic (isolated) PM force vs. exact Newtonian G*m/r^2, # log-spaced separations; exits after printing the result");
    ADDUSAGE("     --pm-test-isolated-meshsize #  physical size of the doubled FFT grid for --pm-test-isolated-force [" << pmTestIsolatedMeshSize << "]");
#ifdef PERIODIC
    ADDUSAGE("     --pm-test-treepm-force #  Phase 4: combined tree+PM force vs. an Ewald-summation reference, # log-spaced separations; exits after printing the result");
#endif
#if defined(PMGRID) && !defined(PERIODIC)
    ADDUSAGE("     --pm-test-isolated-treepm-force #  Phase 4: combined tree+PM isolated force vs. exact Newtonian G*m/r^2 (uses --meshsize), # log-spaced separations; exits after printing the result");
#ifdef GADGET_HIP_HIGHRES
    ADDUSAGE("     --zoom-treepm-force-test #  Phase 5 ticket 07: combined tree(Rcut[1]/Asmth[1]) + coarse-PM + fine(zoom)-PM force vs. exact Newtonian G*m/r^2, # log-spaced separations; exits after printing the result");
#endif
#endif
#endif
#ifdef USE_OPENGL
		ADDUSAGE("     --fullscreen #     set fullscreen mode string");
    ADDUSAGE("     --displayfps       enable on-screen FPS display");
		ADDUSAGE("     --Tglow  #         enable glow @ # Myr [" << TstartGlow << "]");
		ADDUSAGE("     --dTglow  #        reach full brightness in @ # Myr [" << dTstartGlow << "]");
		ADDUSAGE("     --stereo           enable stereo rendering");
#endif
#ifdef GALACTICS
		ADDUSAGE("     --milkyway #       use Milky Way model with # particles per proc");
		ADDUSAGE("     --mwfork   #       fork Milky Way generator into # processes [" << nMWfork << "]");
		ADDUSAGE("     --seed     #       seed to use for the Milky Way  [" << galSeed  << "]");
    ADDUSAGE("     --taskvar  #       variable name to obtain task id [for randoms seed] before MPI_Init. \n");
#endif
    ADDUSAGE("     --plummer  #       use Plummer model with # particles per proc");
		ADDUSAGE("     --sphere   #       use spherical model with # particles per proc");
		ADDUSAGE("     --cube     #       use cube model with # particles per proc");
    ADDUSAGE("     --diskmode         use diskmode to read same input file all MPI taks and randomly shuffle its positions");
    ADDUSAGE("     --mpirendermode    use MPI to communicate with the renderer. Must only be used with bonsai_driver. [disabled]");
		ADDUSAGE(" ");


		opt.setFlag( "help" ,   'h');
		opt.setFlag( "diskmode");
		opt.setFlag( "mpirendermode");
		opt.setOption( "infile",  'i');
		opt.setOption( "param");
		opt.setOption( "snapshot-test");
		opt.setOption( "softening-test");
		opt.setOption( "timestep-test");
		opt.setOption( "driftfac-test");
		opt.setOption( "zoom-mask");
		opt.setOption( "zoom-enlarge");
		opt.setOption( "zoom-kernel-test");
		opt.setOption( "bonsaifile",  'f');
		opt.setFlag  ( "restart");
		opt.setOption( "dt",      't' );
		opt.setOption( "tend",    'T' );
		opt.setOption( "iend",    'I' );
		opt.setOption( "eps",     'e' );
		opt.setOption( "theta",   'o' );
		opt.setOption( "erracc" );
		opt.setOption( "rebuild", 'r' );
		opt.setOption( "restart-file" );
			// NOT "restart": Bonsai already owns that flag (line 513, `restartSim`) for resuming from
			// its own dump format via fileIO->readFile. Overloading it would make one flag mean two
			// different resume mechanisms, and with no --param both would fire on the same run.
			opt.setFlag  ( "resume" );
		opt.setFlag  ( "debug" );
    opt.setOption( "plummer");
#ifdef GALACTICS
    opt.setOption( "milkyway");
    opt.setOption( "mwfork");
    opt.setOption( "taskvar");
    opt.setOption( "seed");
#endif
    opt.setOption( "sphere");
    opt.setOption( "cube");
    opt.setOption( "dev" );
    opt.setOption( "renderdev" );
    opt.setOption( "logfile" );
    opt.setOption( "snapname");
    opt.setOption( "snapiter");
    opt.setOption( "quickdump");
    opt.setOption( "quickratio");
    opt.setFlag  ( "usempiio");
    opt.setFlag  ( "noquicksync");
    opt.setOption( "rmdist");
    opt.setOption( "valueadd");
    opt.setOption( "reducebodies");

#if ENABLE_LOG
    opt.setFlag("log");
    opt.setFlag("prepend-rank");
#endif
    opt.setFlag("direct");
#ifdef PERIODIC
    opt.setOption("boxsize");
#endif
#if defined(PMGRID) && !defined(PERIODIC)
    opt.setOption("meshsize");
#endif
    opt.setOption("pm-test-cic");
    opt.setOption("pm-test-boxsize");
#ifdef PMGRID
    opt.setOption("pm-test-fft");
    opt.setOption("pm-test-poisson");
    opt.setOption("pm-test-poisson-boxsize");
    opt.setOption("pm-test-force");
    opt.setOption("pm-test-force-boxsize");
    opt.setOption("pm-test-sinewave");
    opt.setOption("pm-test-sinewave-boxsize");
    opt.setOption("pm-test-sinewave-amplitude");
    opt.setOption("pm-test-isolated-force");
    opt.setOption("pm-test-isolated-meshsize");
#ifdef PERIODIC
    opt.setOption("pm-test-treepm-force");
#endif
#if !defined(PERIODIC)
    opt.setOption("pm-test-isolated-treepm-force");
#ifdef GADGET_HIP_HIGHRES
    opt.setOption("zoom-treepm-force-test");
#endif
#endif
#endif
#ifdef USE_OPENGL
    opt.setOption( "fullscreen");
    opt.setOption( "Tglow");
    opt.setOption( "dTglow");
    opt.setFlag("displayfps");
    opt.setFlag("stereo");
#endif

    opt.processCommandArgs( argc, argv );


    if( ! opt.hasOptions() ||  opt.getFlag( "help" ) || opt.getFlag( 'h' ) )
    {
      /* print usage if no options or requested help */
      opt.printUsage();
      ::exit(0);
    }

    if (opt.getFlag("direct"))          direct        = true;
    if (opt.getFlag("restart"))         restartSim    = true;
    if (opt.getFlag("displayfps"))      displayFPS    = true;
    if (opt.getFlag("diskmode"))        diskmode      = true;
    if (opt.getFlag("mpirendermode"))   mpiRenderMode = true;
    if(opt.getFlag("stereo"))           stereo        = true;

#if ENABLE_LOG
    if (opt.getFlag("log"))           ENABLE_RUNTIME_LOG = true;
    if (opt.getFlag("prepend-rank"))  PREPEND_RANK       = true;
#endif    
    char *optarg = NULL;
    if ((optarg = opt.getValue("infile")))       fileName           = string(optarg);
    if ((optarg = opt.getValue("param")))        paramFile          = string(optarg);
    if ((optarg = opt.getValue("snapshot-test"))) snapshotTestFormat = atoi(optarg);
    if ((optarg = opt.getValue("softening-test"))) softeningTestSamples = atoi(optarg);
    if ((optarg = opt.getValue("timestep-test"))) timestepTestSamples = atoi(optarg);
    if ((optarg = opt.getValue("driftfac-test"))) driftfacTestSamples = atoi(optarg);
    if ((optarg = opt.getValue("zoom-mask")))    zoomMask    = (unsigned int) strtoul(optarg, NULL, 0); // base 0: accepts "0x..." or decimal
    if ((optarg = opt.getValue("zoom-enlarge"))) zoomEnlarge = (float) atof(optarg);
    if ((optarg = opt.getValue("zoom-kernel-test"))) zoomKernelTestSamples = atoi(optarg);
    if ((optarg = opt.getValue("bonsaifile")))   bonsaiFileName     = std::string(optarg);
    if ((optarg = opt.getValue("plummer")))      nPlummer           = atoi(optarg);
    if ((optarg = opt.getValue("milkyway")))     nMilkyWay          = atoi(optarg);
    if ((optarg = opt.getValue("mwfork")))       nMWfork            = atoi(optarg);
    if ((optarg = opt.getValue("seed")))         galSeed            = atoi(optarg);
    if ((optarg = opt.getValue("taskvar")))      taskVar            = std::string(optarg);
    if ((optarg = opt.getValue("sphere")))       nSphere            = atoi(optarg);
    if ((optarg = opt.getValue("cube")))         nCube              = atoi(optarg);
#ifdef PERIODIC
    if ((optarg = opt.getValue("boxsize")))         boxSize          = (float) atof(optarg);
#endif
#if defined(PMGRID) && !defined(PERIODIC)
    if ((optarg = opt.getValue("meshsize")))        meshSize         = (float) atof(optarg);
#endif
    if ((optarg = opt.getValue("pm-test-cic")))     pmTestCicGrid    = atoi(optarg);
    if ((optarg = opt.getValue("pm-test-boxsize"))) pmTestCicBoxSize = (float) atof(optarg);
#ifdef PMGRID
    if ((optarg = opt.getValue("pm-test-fft")))             pmTestFftGrid         = atoi(optarg);
    if ((optarg = opt.getValue("pm-test-poisson")))         pmTestPoissonGrid     = atoi(optarg);
    if ((optarg = opt.getValue("pm-test-poisson-boxsize"))) pmTestPoissonBoxSize  = (float) atof(optarg);
    if ((optarg = opt.getValue("pm-test-force")))           pmTestForceGrid       = atoi(optarg);
    if ((optarg = opt.getValue("pm-test-force-boxsize")))   pmTestForceBoxSize    = (float) atof(optarg);
    if ((optarg = opt.getValue("pm-test-sinewave")))           pmTestSineGrid      = atoi(optarg);
    if ((optarg = opt.getValue("pm-test-sinewave-boxsize")))   pmTestSineBoxSize   = (float) atof(optarg);
    if ((optarg = opt.getValue("pm-test-sinewave-amplitude"))) pmTestSineAmplitude = (float) atof(optarg);
    if ((optarg = opt.getValue("pm-test-isolated-force")))    pmTestIsolatedSamples  = atoi(optarg);
    if ((optarg = opt.getValue("pm-test-isolated-meshsize"))) pmTestIsolatedMeshSize = (float) atof(optarg);
#ifdef PERIODIC
    if ((optarg = opt.getValue("pm-test-treepm-force"))) pmTestTreePMSamples = atoi(optarg);
#endif
#if !defined(PERIODIC)
    if ((optarg = opt.getValue("pm-test-isolated-treepm-force"))) pmTestIsolatedTreePMSamples = atoi(optarg);
#ifdef GADGET_HIP_HIGHRES
    if ((optarg = opt.getValue("zoom-treepm-force-test"))) zoomTreePMTestSamples = atoi(optarg);
#endif
#endif
#endif
    if ((optarg = opt.getValue("logfile")))      logFileName        = string(optarg);
    if ((optarg = opt.getValue("dev")))          devID              = atoi  (optarg);
    renderDevID = devID;
    if ((optarg = opt.getValue("renderdev")))    renderDevID        = atoi  (optarg);
    if ((optarg = opt.getValue("dt")))           timeStep           = (float) atof  (optarg);
    if ((optarg = opt.getValue("tend")))         tEnd               = (float) atof  (optarg);
    if ((optarg = opt.getValue("iend")))         iterEnd            = atoi  (optarg);
    if ((optarg = opt.getValue("eps")))          eps                = (float) atof  (optarg);
    if ((optarg = opt.getValue("theta")))        theta              = (float) atof  (optarg);
    if ((optarg = opt.getValue("erracc")))       errTolForceAcc     = (float) atof  (optarg);
    if ((optarg = opt.getValue("snapname")))     snapshotFile       = string(optarg);
    if ((optarg = opt.getValue("snapiter")))     snapshotIter       = (float) atof  (optarg);
    if ((optarg = opt.getValue("quickdump")))    quickDump          = (float) atof  (optarg);
    if ((optarg = opt.getValue("quickratio")))   quickRatio         = (float) atof  (optarg);
    if (opt.getValue("usempiio")) useMPIIO = true;
    if (opt.getValue("noquicksync")) quickSync = false;
    if ((optarg = opt.getValue("rmdist")))       remoDistance       = (float) atof  (optarg);
    if ((optarg = opt.getValue("rebuild")))      rebuild_tree_rate  = atoi  (optarg);
    if ((optarg = opt.getValue("restart-file"))) restartFilePath    = std::string(optarg);
    if (opt.getFlag("resume"))                   resumeFromRestart  = true;
    if (opt.getFlag("debug") || getenv("GADGET_HIP_DEBUG")) gadget_hip_debug_log = true;
    if ((optarg = opt.getValue("reducebodies"))) reduce_bodies_factor = atoi  (optarg);
    if ((optarg = opt.getValue("reducedust")))	 reduce_dust_factor = atoi  (optarg);
#if USE_OPENGL
    if ((optarg = opt.getValue("fullscreen")))	 fullScreenMode     = string(optarg);
    if ((optarg = opt.getValue("Tglow")))	 TstartGlow  = (float)atof(optarg);
    if ((optarg = opt.getValue("dTglow")))	 dTstartGlow  = (float)atof(optarg);
    dTstartGlow = std::max(dTstartGlow, 1.0f);
#endif
    // Phase 5 ticket 02 (PLAN.md): --snapshot-test is a pure host-side round trip, no MPI/CUDA
    // context or IC needed at all -- runs and exits before any of that gets set up.
    if (snapshotTestFormat != 0)
    {
#ifdef GADGET_HIP_LONGIDS
      const bool builtWithLongIds = true;
#else
      const bool builtWithLongIds = false;
#endif
      std::string report;
      bool ok = gadget_snapshot_selftest("/tmp/gadget_hip_snapshot_selftest.dat",
                                          snapshotTestFormat, builtWithLongIds, report);
      fprintf(stderr, "%s", report.c_str());
      ::exit(ok ? 0 : 1);
    }

    // Phase 5 ticket 01 (PLAN.md): --param is additive, parsed here (after every CLI value above
    // is already extracted, before anything downstream consumes theta/errTolForceAcc/boxSize/
    // tEnd) so it can both feed the PERIODIC boxSize check right below and override those four
    // values as the single source of truth for a --param run -- not a second copy layered on top
    // of whatever --theta/--erracc/--boxsize/--tend also set.
    if (!paramFile.empty())
    {
#ifdef PERIODIC
      const bool builtWithPeriodic = true;
#else
      const bool builtWithPeriodic = false;
#endif
      std::vector<std::string> paramErrors;
      if (!gadget_params_parse(paramFile, builtWithPeriodic, gadgetParams, paramErrors))
      {
        fprintf(stderr, "Error(s) parsing --param file '%s':\n", paramFile.c_str());
        for (auto &e : paramErrors) fprintf(stderr, "  %s\n", e.c_str());
        ::exit(1);
      }
      haveGadgetParams = true;

      fprintf(stderr, "[PARAM] Parsed '%s': G=%.9g Hubble=%.9g UnitTime_in_s=%.9g "
                       "ComovingIntegrationOn=%d Omega0=%g OmegaLambda=%g BoxSize=%g\n",
              paramFile.c_str(), gadgetParams.G, gadgetParams.Hubble,
              gadgetParams.UnitTime_in_s, gadgetParams.ComovingIntegrationOn,
              gadgetParams.Omega0, gadgetParams.OmegaLambda, gadgetParams.BoxSize);

      // Only the handful of physics parameters with a live consumer already in this port are
      // applied here. Snapshot I/O (ticket 02), real spline softening (ticket 03), the real
      // timestep criterion (ticket 04), and comoving integration (ticket 05) are what will make
      // the rest of GadgetParams actually drive a run -- until then, a --param file with no
      // --infile/--plummer/etc IC source can only be validated, not simulated (see the check
      // right below).
      // C-B-03: `TypeOfOpeningCriterion` was parsed (gadget_params.cpp:85) and declared
      // (gadget_params.h:86) and READ NOWHERE. Gadget-2 treats it as the switch between two
      // genuinely different multipole acceptance criteria: `1` sets `All.ErrTolTheta = 0`
      // (gravtree.c:321-322), which permanently selects the RELATIVE criterion driven by
      // `ErrTolForceAcc`, and `ErrTolTheta` is thereafter ignored. The port instead always used
      // `theta` below, i.e. the geometric criterion, silently substituting a different force
      // accuracy for the one the parameter file asked for.
      //
      // This was not hypothetical: the 512^3 run of record and the CPU reference it is validated
      // against BOTH specify `TypeOfOpeningCriterion 1` with `ErrTolForceAcc 0.002`, while the
      // production build has `MAC_SPRINGEL` OFF (CMakeLists.txt:52). So the reference ran the
      // relative criterion and the port ran geometric theta=0.5 -- the accuracy comparison was
      // between two different criteria.
      //
      // Refuse rather than substitute, on the same argument accepted for C-D-14: a configuration
      // that cannot produce the requested physics must stop the run, not emit a warning into a
      // batch log.
      if (gadgetParams.TypeOfOpeningCriterion == 1)
      {
#ifndef _MAC_SPRINGEL_
        fprintf(stderr,
          "\nFATAL (C-B-03): the parameter file requests TypeOfOpeningCriterion = 1 (Gadget-2's\n"
          "relative / Springel acceptance criterion, driven by ErrTolForceAcc = %g), but this\n"
          "binary was built WITHOUT the relative MAC (CMake option MAC_SPRINGEL is OFF), so it can\n"
          "only evaluate the geometric criterion with ErrTolTheta = %g.\n"
          "These are different force-accuracy criteria and results are NOT comparable.\n"
          "Either rebuild with -DMAC_SPRINGEL=ON, or set TypeOfOpeningCriterion 0 to ask for the\n"
          "geometric criterion explicitly.\n\n",
          gadgetParams.ErrTolForceAcc, gadgetParams.ErrTolTheta);
        ::exit(1);
#endif
      }
      else if (gadgetParams.TypeOfOpeningCriterion != 0)
      {
        fprintf(stderr, "\nFATAL (C-B-03): TypeOfOpeningCriterion = %d is not a value Gadget-2\n"
                        "defines (0 = geometric, 1 = relative).\n\n",
                gadgetParams.TypeOfOpeningCriterion);
        ::exit(1);
      }

      // T28: a build WITHOUT UNEQUALSOFTENINGS has no per-node softening summary and no forced
      // descent, so it can only be correct when every populated particle type shares one softening.
      // Refuse otherwise rather than silently apply the target's softening to nodes that contain
      // coarser members -- the same discipline as C-B-03's MAC guard above. (Gadget-2 makes this a
      // compile-time option too; it simply does not compile the machinery when it is off.)
#ifndef UNEQUALSOFTENINGS
      {
        const double sft[6] = { gadgetParams.SofteningGas,   gadgetParams.SofteningHalo,
                                gadgetParams.SofteningDisk,  gadgetParams.SofteningBulge,
                                gadgetParams.SofteningStars, gadgetParams.SofteningBndry };
        double distinct[6]; int nDistinct = 0;
        for (int t = 0; t < 6; t++)
        {
          if (!(sft[t] > 0.0)) continue;          // an unpopulated/zero type imposes nothing
          bool seen = false;
          for (int k = 0; k < nDistinct; k++) if (distinct[k] == sft[t]) { seen = true; break; }
          if (!seen) distinct[nDistinct++] = sft[t];
        }
        if (nDistinct > 1)
        {
          fprintf(stderr,
            "\nFATAL (T28): this parameter file specifies %d DIFFERENT non-zero softenings\n"
            "(Gas %g, Halo %g, Disk %g, Bulge %g, Stars %g, Bndry %g), but this binary was built\n"
            "WITHOUT mixed-softening support (CMake option UNEQUAL_SOFTENINGS is OFF).\n"
            "Without it a node containing coarser members is softened with the TARGET's length and\n"
            "is never force-opened, so forces near a species boundary are wrong.\n"
            "Rebuild with -DUNEQUAL_SOFTENINGS=ON, or give every populated type the same softening.\n\n",
            nDistinct, sft[0], sft[1], sft[2], sft[3], sft[4], sft[5]);
          ::exit(1);
        }
      }
#endif

      theta          = (float) gadgetParams.ErrTolTheta;
      errTolForceAcc = (float) gadgetParams.ErrTolForceAcc;
#ifdef PERIODIC
      if (gadgetParams.BoxSize > 0.0)
        boxSize = (float) gadgetParams.BoxSize;
#endif
      if (gadgetParams.TimeMax > 0.0)
        tEnd = (float) gadgetParams.TimeMax;
    }

    // Phase 5 ticket 02 (PLAN.md): a --param file's own InitCondFile now counts as a real IC
    // source (loaded via gadget_snapshot_read further below), not just something to validate and
    // discard -- ticket 01's "nothing to simulate yet" dry-run behavior now only applies when
    // --param was given with no InitCondFile-backed load actually possible.
    haveGadgetIC = haveGadgetParams && !gadgetParams.InitCondFile.empty();

    if (bonsaiFileName.empty() && fileName.empty() && nPlummer == -1 && nSphere == -1 &&
        nMilkyWay == -1 && nCube == -1 && !haveGadgetIC && softeningTestSamples == 0 &&
        timestepTestSamples == 0 && driftfacTestSamples == 0 && zoomKernelTestSamples == 0 &&
        zoomTreePMTestSamples == 0)
    {
      if (haveGadgetParams)
      {
        fprintf(stderr, "[PARAM] '%s' parsed and validated successfully, but its InitCondFile "
                         "is empty -- nothing to simulate.\n", paramFile.c_str());
        ::exit(0);
      }
      opt.printUsage();
      ::exit(0);
    }
#ifdef PERIODIC
    if (boxSize <= 0.0f)
    {
      fprintf(stderr, "Error: this binary was built with PERIODIC -- --boxsize # is required.\n");
      opt.printUsage();
      ::exit(1);
    }
#endif
#if defined(PMGRID) && !defined(PERIODIC)
    if (meshSize <= 0.0f)
    {
      fprintf(stderr, "Error: this binary was built with GADGET_HIP_PMGRID (isolated TreePM, no "
                       "PERIODIC) -- --meshsize # is required.\n");
      opt.printUsage();
      ::exit(1);
    }
#endif
    if (!bonsaiFileName.empty() && !useMPIIO)
    {
      opt.printUsage();
      ::exit(0);
    }

#undef ADDUSAGE
  }
#endif



  /********** init galaxy before MPI initialization to prevent problems with forking **********/
  const char * argVal = getenv(taskVar.c_str());
  if (argVal == NULL)
  {
    fprintf(stderr, " Unknown ENV_VARIABLE: %s  -- Falling to basic forking method after MPI_Init, unsafe!\n", taskVar.c_str());
    taskVar = std::string();
  }
  if (nMilkyWay >= 0 && !taskVar.empty())
  {
    assert(argVal != NULL);
    const int procId = atoi(argVal);
    //    fprintf(stderr, " taskVar= %s , value= %d\n", taskVar.c_str(), procId);
    #ifdef GALACTICS
        tStartModel = get_time_main();
        //Use 32768*7 for nProcs to create independent seeds for all processes we use
        //do not scale until we know the number of processors
        generateGalacticsModel(procId, 32768*7, galSeed, nMilkyWay, nMWfork,
                               false, bodyPositions, bodyVelocities,
                               bodyIDs);
        tEndModel   = get_time_main();
    #else
        assert(0);
    #endif
  }

  /*********************************/

  /************** end - command line arguments ********/

  /* Overrule settings for the device */
  //  const char * tempRankStr = getenv("OMPI_COMM_WORLD_RANK");
  //  devID = renderDevID = atoi(tempRankStr);
  //  fprintf(stderr,"Overruled ids: %d ", devID);
  /* End overrule */


#ifdef USE_OPENGL
  // create OpenGL context first, and register for interop
  initGL(argc, argv, fullScreenMode.c_str(), stereo);
//  cudaGLSetGLDevice(devID); //TODO should this not be renderDev?
#endif

  initTimers();

#ifdef WIN32
  int pid = _getpid();
#else
  int pid = (int)getpid();
#endif

#if 0
  //Used for profiler, note this has to be done before initing to
  //octree otherwise it has no effect...Therefore use pid instead of mpi procId
  char *gpu_prof_log;
  gpu_prof_log=getenv("CUDA_PROFILE_LOG");
  if(gpu_prof_log){
    char tmp[50];
    sprintf(tmp,"process_%d_%s",pid,gpu_prof_log);
    #ifdef WIN32
                SetEnvironmentVariable("CUDA_PROFILE_LOG", tmp);
    #else
                setenv("CUDA_PROFILE_LOG",tmp,1);
        LOGF(stderr, "TESTING log on proc: %d val: %s \n", pid, tmp);
    #endif
  }
#endif

  int mpiInitialized =  0;
  int procId         = -1;
  int nProcs         = -1;
  
#ifdef USE_MPI  
      MPI_Initialized(&mpiInitialized);
      MPI_Comm mpiCommWorld = MPI_COMM_WORLD;
      if (!mpiInitialized)
      {
        #ifdef _MPIMT
            int provided;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
            assert(MPI_THREAD_MULTIPLE == provided);
        #else
            MPI_Init(&argc, &argv);
        #endif
        shrMemPID = 0;
      }
      else
      {
        //MPI environment initialized by the driver program
        mpiCommWorld = comm;
      }

      MPI_Comm_size(mpiCommWorld, &nProcs);
      MPI_Comm_rank(mpiCommWorld, &procId);
#else
    MPI_Comm mpiCommWorld = 0;
    procId                = 0;
    nProcs                = 1;
#endif
#if ENABLE_LOG
  PREPEND_RANK_PROCID = procId;
  PREPEND_RANK_NPROCS = nProcs;
#endif

    if (mpiRenderMode) assert(mpiInitialized); //The external renderer requires a working MPI environment

    // C-A-17 / C-B-17: this port is single-GPU. The MPI code paths it inherited from Bonsai are
    // still REACHABLE -- makeLET() is called under `if(nProcs > 1)` (gpu_iterate.cpp), and the PM
    // block there warns "Long-range gravity will be MISSING from this multi-process run" and then
    // CONTINUES. A run that cannot produce correct physics must stop, not emit a warning into a
    // batch log that nobody reads; this is the same argument accepted for C-D-14's non-advancing
    // step and C-B-03's unhonoured MAC.
    //
    // Refused here, once, at startup rather than at each of the eight `nProcs > 1` sites: a single
    // gate cannot be half-applied, and it fires before any output file is written.
    if (nProcs > 1)
    {
      fprintf(stderr,
        "\nFATAL: this build is SINGLE-GPU and was launched on %d MPI ranks.\n"
        "The port has no MPI domain decomposition: the PM solver has no slab decomposition (so\n"
        "long-range gravity would be silently MISSING), and the locally-essential-tree exchange is\n"
        "inherited-but-unvalidated Bonsai code. Results from a multi-rank run would be wrong, not\n"
        "merely slow.\n"
        "Run on a single rank (no mpirun, or `mpirun -np 1`).\n\n", nProcs);
      fflush(stderr);
#ifdef USE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
#endif
      return 1;
    }


    if(nProcs > 1)
    {
      logFileName.append("-");
      char buff[16];
      sprintf(buff,"%d-%d", nProcs, procId);
      logFileName.append(buff);
    }

    //Use a string stream buffer, only write data at the end of the run
    std::stringstream logStream;
    ostream &logFile = logStream;

    my_dev::context cudaContext;

    if(nProcs > 1) devID = procId % getNumberOfCUDADevices();

    cudaContext.create(logFile, false); //Do logging to file and enable timing (false = enabled)
    cudaContext.createQueue(devID);

    char logPretext[64];
    sprintf(logPretext, "PROC-%05d ", procId);
    cudaContext.setLogPreamble(logPretext);

    // Phase 5 ticket 03 (PLAN.md): --softening-test, same one-shot-diagnostic idiom as the
    // --pm-test-* flags -- needs only a live HIP context (just created above), no IC/tree at all.
    // Host-side reference formula written independently from PHASE5_ZOOM_COMOVING_SPEC.md Sec 4.1
    // (not copied from the kernel) so this is a genuine independent check, matching this
    // project's own established testing practice for the PM finite-difference stencil etc.
    if (softeningTestSamples > 0)
    {
      auto expectedSpline = [](float r, float h, float mass, float &fac, float &pot)
      {
        if (r >= h)
        {
          fac = mass / (r * r * r);
          pot = -mass / r;
          return;
        }
        const float u = r / h;
        float wp;
        if (u < 0.5f)
        {
          fac = mass / (h*h*h) * (10.666666667f + u*u*(32.0f*u - 38.4f));
          wp  = -2.8f + u*u*(5.333333333f + u*u*(6.4f*u - 9.6f));
        }
        else
        {
          fac = mass / (h*h*h) * (21.333333333f - 48.0f*u + 38.4f*u*u -
                                   10.666666667f*u*u*u - 0.066666667f/(u*u*u));
          wp  = -3.2f + 0.066666667f/u +
                u*u*(10.666666667f + u*(-16.0f + u*(9.6f - 2.133333333f*u)));
        }
        pot = mass / h * wp;
      };

      // Test cases spanning: pure Newtonian (r>=h), both spline branches (u<0.5, u>=0.5), r=0
      // (no NaN), and same-type (h_i==h_j) vs. different-type (h_i!=h_j, exercising max()).
      struct Case { float r, h_i, h_j, mass; const char *label; };
      const std::vector<Case> cases = {
        { 5.00f, 1.0f, 1.0f, 2.0f, "r>>h (Newtonian), same type" },
        { 0.80f, 1.0f, 1.0f, 2.0f, "h/2<=r<h, same type" },
        { 0.30f, 1.0f, 1.0f, 2.0f, "r<h/2, same type" },
        { 0.00f, 1.0f, 1.0f, 2.0f, "r=0, same type" },
        { 0.90f, 0.5f, 1.5f, 3.0f, "different types, r inside larger h only (tests max())" },
        { 1.20f, 0.5f, 1.5f, 3.0f, "different types, r inside smaller h too" },
      };
      const int n = (int) cases.size();

      std::vector<float3> h_sep(n);
      std::vector<float>  h_mass(n), h_hi(n), h_hj(n);
      for (int i = 0; i < n; i++)
      {
        h_sep[i]  = make_float3(cases[i].r, 0.0f, 0.0f);
        h_mass[i] = cases[i].mass;
        h_hi[i]   = cases[i].h_i;
        h_hj[i]   = cases[i].h_j;
      }

      float3 *d_sep = nullptr; float *d_mass = nullptr, *d_hi = nullptr, *d_hj = nullptr;
      float4 *d_out = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_sep,  n * sizeof(float3)));
      CU_SAFE_CALL(hipMalloc((void**)&d_mass, n * sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_hi,   n * sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_hj,   n * sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_out,  n * sizeof(float4)));
      CU_SAFE_CALL(hipMemcpy(d_sep,  h_sep.data(),  n * sizeof(float3), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_mass, h_mass.data(), n * sizeof(float),  hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_hi,   h_hi.data(),   n * sizeof(float),  hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_hj,   h_hj.data(),   n * sizeof(float),  hipMemcpyHostToDevice));

      softening_test_pair(d_sep, d_mass, d_hi, d_hj, n, d_out, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());

      std::vector<float4> h_out(n);
      CU_SAFE_CALL(hipMemcpy(h_out.data(), d_out, n * sizeof(float4), hipMemcpyDeviceToHost));
      hipFree(d_sep); hipFree(d_mass); hipFree(d_hi); hipFree(d_hj); hipFree(d_out);

      bool allOk = true;
      for (int i = 0; i < n; i++)
      {
        const float h = std::max(cases[i].h_i, cases[i].h_j);
        float fac, pot;
        expectedSpline(cases[i].r, h, cases[i].mass, fac, pot);
        // Kernel force convention: acc.xyz += fac*dr, dr=(r,0,0) here, so acc.x == fac*r
        // (0 exactly at r=0, matching the expected fac*0). acc.w == pot directly: add_acc()
        // computes mrinv such that `acc.w -= mrinv` lands on the potential as defined here (see
        // add_acc()'s own derivation comment for both branches).
        const float expectAccX = fac * cases[i].r;
        const float gotAccX    = h_out[i].x;
        const float gotPot     = h_out[i].w;
        const float tolAcc = std::max(1e-5f, std::fabs(expectAccX) * 1e-4f);
        const float tolPot = std::max(1e-5f, std::fabs(pot) * 1e-4f);
        const bool ok = std::fabs(gotAccX - expectAccX) < tolAcc &&
                        std::fabs(gotPot - pot) < tolPot;
        allOk &= ok;
        fprintf(stderr, "[SOFTENING_TEST] %-45s r=%.3f h=%.3f  accX: got=%.6g want=%.6g  "
                         "pot: got=%.6g want=%.6g  %s\n",
                cases[i].label, cases[i].r, h, gotAccX, expectAccX, gotPot, pot,
                ok ? "PASS" : "FAIL");
      }
      ::exit(allOk ? 0 : 1);
    }

    // Phase 5 ticket 04 (PLAN.md): --timestep-test, direct-launch unit test for the real
    // compute_dt() kernel (timestep_test_compute_dt(), same idiom as --softening-test). Host-side
    // reference written independently from PHASE5_ZOOM_COMOVING_SPEC.md Sec 3, not copied from
    // the kernel -- reproduces the exact same clamp-then-power-of-two-snap sequence.
    if (timestepTestSamples > 0)
    {
      auto expectedDt = [](float errTolIntAccuracy, float atime, float fac1, float hubble_a,
                            float maxSizeTimestep, float minSizeTimestep, float dtDisplacement,
                            float4 acc, float forceSoftening) -> float
      {
        float ac = fac1 * std::sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z);
        if (ac == 0.0f) ac = 1.0e-30f;
        const float softeningTable = forceSoftening / 2.8f;
        float dt = std::sqrt(2.0f * errTolIntAccuracy * atime * softeningTable / ac);
        dt *= hubble_a;
        if (dt >= maxSizeTimestep) dt = maxSizeTimestep;
        if (dt >= dtDisplacement)  dt = dtDisplacement;
        if (dt <  minSizeTimestep) dt = minSizeTimestep;
        float q = maxSizeTimestep;
        while (q > dt) q *= 0.5f;
        return q;
      };

      struct Case {
        float errTol, atime, fac1, hubble_a, maxDt, minDt, dtDisp;
        float4 acc; float soft; const char *label;
      };
      const std::vector<Case> cases = {
        { 0.02f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0001f, 1.0e30f,
          make_float4(2.0f,0,0,0), 1.0f, "baseline, non-comoving-equivalent" },
        { 0.02f, 0.5f, 4.0f, 2.0f, 1.0f, 0.0001f, 1.0e30f,
          make_float4(0.5f,0,0,0), 1.0f, "comoving scaling (atime/fac1/hubble_a all != 1)" },
        { 0.02f, 1.0f, 1.0f, 1.0f, 0.05f, 0.0001f, 1.0e30f,
          make_float4(0.01f,0,0,0), 1.0f, "MaxSizeTimestep clamp (tiny accel)" },
        { 0.02f, 1.0f, 1.0f, 1.0f, 1.0f, 0.02f, 1.0e30f,
          make_float4(500.0f,0,0,0), 1.0f, "MinSizeTimestep clamp (huge accel)" },
        { 0.02f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0001f, 0.01f,
          make_float4(2.0f,0,0,0), 1.0f, "dt_displacement clamp (tighter than raw dt)" },
        { 0.02f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0001f, 1.0e30f,
          make_float4(3.0f,4.0f,0,0), 2.8f, "different softening/accel, 3-4-5 triangle magnitude" },
      };
      const int n = (int) cases.size();

      std::vector<float4> h_acc(n);
      std::vector<float>  h_soft(n), h_dtOut(n);
      for (int i = 0; i < n; i++) { h_acc[i] = cases[i].acc; h_soft[i] = cases[i].soft; }

      // Each case gets its own single-particle launch since errTol/atime/etc are per-launch
      // scalars in this kernel (matching the real per-step call site, which uses one set of
      // cosmology scalars for the whole active-particle set that step) -- not a limitation of the
      // kernel itself, just this test's simplest harness shape.
      bool allOk = true;
      for (int i = 0; i < n; i++)
      {
        const Case &c = cases[i];
        float got = 0.0f;
        timestep_test_compute_dt(1, 0.0f, c.errTol, c.atime, c.fac1, c.hubble_a, c.maxDt, c.minDt,
                                  c.dtDisp, &h_acc[i], &h_soft[i], &got, 0);
        float want = expectedDt(c.errTol, c.atime, c.fac1, c.hubble_a, c.maxDt, c.minDt, c.dtDisp,
                                 c.acc, c.soft);
        const float tol = std::max(1e-6f, std::fabs(want) * 1e-4f);
        const bool ok = std::fabs(got - want) < tol;
        allOk &= ok;
        fprintf(stderr, "[TIMESTEP_TEST] %-55s got=%.6g want=%.6g %s\n",
                c.label, got, want, ok ? "PASS" : "FAIL");
      }
      ::exit(allOk ? 0 : 1);
    }

    // Phase 5 ticket 05 (PLAN.md): --driftfac-test confirms the DEVICE table lookup
    // (driftFactorD() in timestep.cu) matches the HOST one (gadget_driftfac.cpp) it's a separate
    // re-implementation of -- the host implementation itself is validated against an independent
    // SciPy integration elsewhere (see LOG.md), so this test's job is purely "device == host",
    // not re-deriving correctness from scratch.
    if (driftfacTestSamples > 0)
    {
      GadgetDriftTables tables;
      gadget_init_drift_tables(/*timeBegin=*/0.02, /*timeMax=*/1.0, /*omega0=*/0.3,
                                /*omegaLambda=*/0.7, /*hubble=*/0.1, tables);

      struct Case { float t0, t1; int wantKick; const char *label; };
      const std::vector<Case> cases = {
        { 0.02f, 0.10f, 0, "drift, early times" },
        { 0.10f, 0.50f, 0, "drift, mid range" },
        { 0.50f, 1.00f, 0, "drift, late times" },
        { 0.02f, 0.10f, 1, "gravkick, early times" },
        { 0.10f, 0.50f, 1, "gravkick, mid range" },
        { 0.50f, 1.00f, 1, "gravkick, late times" },
        { 0.30f, 0.31f, 0, "drift, narrow interval" },
      };
      const int n = (int) cases.size();
      std::vector<float> h_t0(n), h_t1(n), h_out(n);
      std::vector<int>   h_want(n);
      for (int i = 0; i < n; i++) { h_t0[i]=cases[i].t0; h_t1[i]=cases[i].t1; h_want[i]=cases[i].wantKick; }

      driftfac_test_lookup(tables.driftTable, tables.gravKickTable, tables.logTimeBegin,
                            tables.logTimeMax, h_t0.data(), h_t1.data(), h_want.data(),
                            n, h_out.data());

      bool allOk = true;
      for (int i = 0; i < n; i++)
      {
        double want = cases[i].wantKick
            ? gadget_get_gravkick_factor(tables, cases[i].t0, cases[i].t1)
            : gadget_get_drift_factor(tables, cases[i].t0, cases[i].t1);
        const double tol = std::max(1e-6, std::fabs(want) * 1e-5);
        const bool ok = std::fabs((double) h_out[i] - want) < tol;
        allOk &= ok;
        fprintf(stderr, "[DRIFTFAC_TEST] %-30s device=%.9g host=%.9g %s\n",
                cases[i].label, h_out[i], want, ok ? "PASS" : "FAIL");
      }
      ::exit(allOk ? 0 : 1);
    }

#ifdef GADGET_HIP_HIGHRES
    // Phase 5 ticket 06 (PLAN.md): --zoom-kernel-test confirms pm_build_finegrid_kernel()
    // (CUDAkernels/pm_zoom.cu) matches the hand-derived differential Green's function
    // erfc(u*asmthRatio) - erfc(u) (including the analytic r=0 self-term), by directly launching
    // the kernel on a small grid and comparing selected cells against a host-side double-precision
    // std::erfc reference -- same "direct-launch a real device kernel, compare to an independent
    // host computation" idiom as --driftfac-test above, just without needing a dedicated
    // *_test.h wrapper since pm_build_finegrid_kernel() is already a plain scalar-argument HIP
    // host function (not routed through the legacy my_dev::kernel .set_args() wrapper).
    if (zoomKernelTestSamples > 0)
    {
      const int   gridSize   = 8;
      const int   gridPitch  = gridSize; // no FFT real-padding needed -- just checking raw values
      const float asmthRatio = 0.5f;     // Asmth[1]/Asmth[0], an arbitrary but nontrivial ratio
      const size_t gridElems = (size_t) gridSize * gridSize * gridPitch;

      float *d_kernel = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_kernel, gridElems * sizeof(float)));
      pm_build_finegrid_kernel(d_kernel, gridSize, gridPitch, asmthRatio, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());

      std::vector<float> h_kernel(gridElems);
      CU_SAFE_CALL(hipMemcpy(h_kernel.data(), d_kernel, gridElems * sizeof(float), hipMemcpyDeviceToHost));
      hipFree(d_kernel);

      auto hostExpected = [&](int i, int j, int k) -> double
      {
        double x = (double) i / gridSize, y = (double) j / gridSize, z = (double) k / gridSize;
        if (x >= 0.5) x -= 1.0;
        if (y >= 0.5) y -= 1.0;
        if (z >= 0.5) z -= 1.0;
        const double r = std::sqrt(x*x + y*y + z*z);
        const double asmthGrid = 1.25 / gridSize;
        const double u = 0.5 * r / asmthGrid;
        if (r > 0.0)
        {
          const double fac = std::erfc(u * (double) asmthRatio) - std::erfc(u);
          return -fac / r;
        }
        const double fac0 = 1.0 - (double) asmthRatio;
        return -fac0 / (std::sqrt(M_PI) * asmthGrid);
      };

      struct Case { int i, j, k; const char *label; };
      const std::vector<Case> cases = {
        { 0, 0, 0, "self-term, r=0" },
        { 1, 0, 0, "on-axis, r=1 cell" },
        { 2, 1, 0, "near, mixed axes" },
        { 4, 0, 0, "wrap boundary (i=gridSize/2 -> -0.5)" },
        { 3, 3, 3, "off-axis, larger r" },
        { 7, 7, 7, "wrap boundary, all axes" },
      };
      bool allOk = true;
      for (const auto &c : cases)
      {
        const size_t idx = ((size_t) c.i * gridSize + c.j) * gridPitch + c.k;
        const double want = hostExpected(c.i, c.j, c.k);
        const double got  = (double) h_kernel[idx];
        const double tol  = std::max(1e-5, std::fabs(want) * 1e-4);
        const bool ok = std::fabs(got - want) < tol;
        allOk &= ok;
        fprintf(stderr, "[ZOOM_KERNEL_TEST] %-40s device=%.9g host=%.9g %s\n",
                c.label, got, want, ok ? "PASS" : "FAIL");
      }
      ::exit(allOk ? 0 : 1);
    }
#endif




    //Create the octree class and set the properties
    octree *tree = new octree(  mpiCommWorld,
                                &cudaContext,
                                argv, devID, theta, eps,
                                snapshotFile, snapshotIter,
                                quickDump, quickRatio, quickSync,
                                useMPIIO,mpiRenderMode,
                                timeStep,
                                tEnd, iterEnd,
                                rebuild_tree_rate, direct, shrMemPID);

    tree->setErrTolForceAcc(errTolForceAcc);

    // Phase 5 tickets 03/04 (PLAN.md): wire the whole parsed --param config in -- softening (03)
    // and timestep (04) both read what they need from it (octree.h/gpu_iterate.cpp). Without
    // --param, recomputeSoftening()/recomputeTimestepGlobals() fall back to this port's own
    // pre-ticket flat-eps/fixed-timeStep behavior.
    if (haveGadgetParams)
      tree->setGadgetParams(gadgetParams);

    // Phase 5 ticket 05 (PLAN.md): always called (allocates the -- possibly all-zero, inert --
    // device tables predict()/correct() unconditionally reference); only actually builds real
    // drift/kick tables when haveGadgetParams && ComovingIntegrationOn.
    tree->initComovingTables();

#ifdef GADGET_HIP_HIGHRES
    // Phase 5 ticket 06 (PLAN.md): PLACEHIGHRESREGION zoom -- always called (mask=0 is the inert
    // "zoom disabled" default), same idiom as setGadgetParams/initComovingTables above. The actual
    // region/solver aren't built here -- that happens lazily on the first tree rebuild inside
    // iterate_once() (octree::recomputeZoomRegion(), gpu_iterate.cpp), since it needs the real
    // particle positions/types, not just CLI config.
    tree->setZoomConfig(zoomMask, zoomEnlarge);
#endif



  double tStartup = tree->get_time();


  if (procId == 0)
  {
    //Note can't use LOGF here since MPI isn't initialized yet
    cerr << "[INIT]\tUsed settings: \n";
    cerr << "[INIT]\tInput  filename "      << fileName                                                      << endl;
    cerr << "[INIT]\tBonsai filename "      << bonsaiFileName                                                << endl;
    cerr << "[INIT]\tLog filename "         << logFileName                                                   << endl;
    cerr << "[INIT]\tTheta: \t\t"           << theta                << "\t\teps: \t\t"      << eps           << endl;
#ifdef _MAC_SPRINGEL_
    cerr << "[INIT]\tMAC: \t\tSpringel(2005)\terrTolForceAcc: " << errTolForceAcc << endl;
#else
    cerr << "[INIT]\tMAC: \t\timproved-BH (theta)" << endl;
#endif
    cerr << "[INIT]\tTimestep: \t"          << timeStep             << "\t\ttEnd: \t\t"     << tEnd          << endl;
    cerr << "[INIT]\titerEnd: \t"           << iterEnd                                                       << endl;
    cerr << "[INIT]\tUse MPI-IO: \t"        << (useMPIIO ? "YES" : "NO")                                     << endl;
    cerr << "[INIT]\tsnapshotFile: \t"      << snapshotFile          << "\tsnapshotIter: \t" << snapshotIter << endl;
    if (useMPIIO)
    {
      cerr << "[INIT]\t  quickDump: \t"      << quickDump << "\t\tquickRatio: \t" << quickRatio << endl;
    }
    cerr << "[INIT]\tInput file: \t"        << fileName     << "\t\tdevID: \t\t"        << devID << endl;
    cerr << "[INIT]\tRemove dist: \t"   << remoDistance << endl;
    cerr << "[INIT]\tRebuild tree every " << rebuild_tree_rate << " timestep\n";


    if( reduce_bodies_factor > 1 ) cerr << "[INIT]\tReduce number of non-dust bodies by " << reduce_bodies_factor << " \n";
    if( reduce_dust_factor   > 1 ) cerr << "[INIT]\tReduce number of dust bodies by " << reduce_dust_factor << " \n";

#if ENABLE_LOG
    if (ENABLE_RUNTIME_LOG)
      cerr << "[INIT]\tRuntime logging is ENABLED \n";
    else
      cerr << "[INIT]\tRuntime logging is DISABLED \n";
#endif
    cerr << "[INIT]\tDirect gravitation is " << (direct ? "ENABLED" : "DISABLED") << endl;
#if USE_OPENGL
    cerr << "[INIT]\tTglow = " << TstartGlow << endl;
    cerr << "[INIT]\tdTglow = " << dTstartGlow << endl;
    cerr << "[INIT]\tstereo = " << stereo << endl;
#endif
#ifdef USE_MPI                
    cerr << "[INIT]\tCode is built WITH MPI Support \n";
#else
    cerr << "[INIT]\tCode is built WITHOUT MPI Support \n";
#endif
  }
  assert(quickRatio > 0 && quickRatio <= 1);

#ifdef USE_MPI

  //Used on Titan and Piz Daint
  #if 1
    omp_set_num_threads(16);
  #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      pthread_getaffinity_np(pthread_self()  , sizeof( cpu_set_t ), &cpuset );

      int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

      int i, set=-1;
      for (i = 0; i < CPU_SETSIZE; i++)
        if (CPU_ISSET(i, &cpuset))
          set = i;
      //    fprintf(stderr,"[Proc: %d ] Thread %d bound to: %d Total cores: %d\n",
      //        procId, tid,  set, num_cores);
    }
  #endif


  #if 0
    omp_set_num_threads(4);
    //default
    // int cpulist[] = {0,1,2,3,8,9,10,11};
    int cpulist[] = {0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15}; //HA-PACS
    //int cpulist[] = {0,1,2,3,4,5,6,7};
    //int cpulist[] = {0,2,4,6, 8,10,12,14};
    //int cpulist[] = {1,3,5,7, 9,11,13,15};
    //int cpulist[] = {1,9,5,11, 3,7,13,15};
    //int cpulist[] = {1,15,3,13, 2,4,6,8};
    //int cpulist[] = {1,1,1,1, 1,1,1,1};


  #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      //int core_id = procId*4+tid;
      int core_id = (procId%4)*4+tid;
      core_id     = cpulist[core_id];

      int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(core_id, &cpuset);
      pthread_t current_thread = pthread_self();
      int return_val = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

      CPU_ZERO(&cpuset);
      pthread_getaffinity_np(pthread_self()  , sizeof( cpu_set_t ), &cpuset );

      int i, set=-1;
      for (i = 0; i < CPU_SETSIZE; i++)
        if (CPU_ISSET(i, &cpuset))
          set = i;
      //printf("CPU2: CPU %d\n", i);

      fprintf(stderr,"Binding thread: %d of rank: %d to cpu: %d CHECK: %d Total cores: %d\n",
          tid, procId, core_id, set, num_cores);
    }
  #endif
#endif



  double tStartup2 = tree->get_time();  

  if (!bonsaiFileName.empty() && useMPIIO)
  {
#ifdef USE_MPI        
    //Read a BonsaiIO file
    const MPI_Comm &comm = mpiCommWorld;
    float tCurrent = 0;
    tree->lReadBonsaiFile(
        bodyPositions, 
        bodyVelocities,
        bodyIDs,
        tCurrent,
        bonsaiFileName,
        procId, nProcs, comm,
        restartSim,
        reduce_bodies_factor);
    tree->set_t_current(tCurrent);
    if (snapshotIter > 0) tree->set_nextSnapTime(tCurrent + snapshotIter);
#else
    fprintf(stderr,"Usage of these options requires to code to be built with MPI support!\n"); exit(0);
#endif      
  }
  else if (haveGadgetIC)
  {
    // Phase 5 ticket 02 (PLAN.md): load a real Gadget-2 snapshot IC via `--param`'s
    // `InitCondFile`/`ICFormat` -- this is the "real" IC-loading path this port has been missing;
    // Tipsy/Plummer/etc below remain the dev/diagnostic paths, unchanged.
    GadgetSnapshotHeader snapHeader;
    GadgetParticleData   snapData;
    std::string          snapErr;
#ifdef GADGET_HIP_LONGIDS
    const bool builtWithLongIds = true;
#else
    const bool builtWithLongIds = false;
#endif
    if (!gadget_snapshot_read(gadgetParams.InitCondFile, gadgetParams.ICFormat, builtWithLongIds,
                               gadgetParams.ComovingIntegrationOn != 0, snapHeader, snapData,
                               snapErr))
    {
      fprintf(stderr, "Error reading Gadget-2 IC '%s': %s\n",
              gadgetParams.InitCondFile.c_str(), snapErr.c_str());
      ::exit(1);
    }
    fprintf(stderr, "[PARAM] Loaded IC '%s': npart=(%d,%d,%d,%d,%d,%d) BoxSize=%g Omega0=%g "
                     "OmegaLambda=%g HubbleParam=%g Time=%g\n",
            gadgetParams.InitCondFile.c_str(), snapHeader.npart[0], snapHeader.npart[1],
            snapHeader.npart[2], snapHeader.npart[3], snapHeader.npart[4], snapHeader.npart[5],
            snapHeader.BoxSize, snapHeader.Omega0, snapHeader.OmegaLambda,
            snapHeader.HubbleParam, snapHeader.time);

    const size_t n = snapData.type.size();
    bodyPositions.resize(n);
    bodyVelocities.resize(n);
    bodyIDs.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
      bodyPositions[i]  = make_float4(snapData.pos[3 * i + 0], snapData.pos[3 * i + 1],
                                       snapData.pos[3 * i + 2], snapData.mass[i]);
      bodyVelocities[i] = make_float4(snapData.vel[3 * i + 0], snapData.vel[3 * i + 1],
                                       snapData.vel[3 * i + 2], 0.0f);
      bodyIDs[i]        = snapData.id[i];
    }
    // Phase 5 ticket 03 (PLAN.md): captured here, applied to tree->localTree.bodies_type once it
    // exists (right after allocateParticleMemory(), further below) -- per-type softening is the
    // first real consumer of snapData.type; tickets 06/07's zoom work will read it too.
    bodyTypes = snapData.type;

    // The IC header's epoch WINS over the parameter file's TimeBegin -- deliberately. But in a
    // COMOVING run a disagreement between the two is almost always a mistake with expensive
    // consequences: Gadget-2 silently starts from TimeBegin regardless of what the IC was
    // generated for, so an IC made for one initial redshift, run with a parameter file left at
    // another, produces a whole simulation with the wrong growth factor and the wrong velocity
    // normalisation -- and nothing in the output says so. Refuse instead.
    //
    // Deliberately NOT checked when ComovingIntegrationOn == 0: in a non-cosmological run `Time`
    // is an arbitrary clock origin and a deliberate mismatch is a legitimate thing to do.
    if (haveGadgetParams && gadgetParams.ComovingIntegrationOn && gadgetParams.TimeBegin > 0.0)
    {
      const double icTime = snapHeader.time;
      const double relDiff = fabs(icTime - gadgetParams.TimeBegin) / gadgetParams.TimeBegin;
      if (relDiff > 1e-6)
      {
        fprintf(stderr,
          "\nFATAL: the IC and the parameter file disagree about the starting epoch.\n"
          "  IC header time : a = %.10g   (z = %.6g)\n"
          "  TimeBegin      : a = %.10g   (z = %.6g)\n"
          "  relative difference %.3g\n\n"
          "In a comoving run this is almost always a mistake, and a costly one: the ICs carry a\n"
          "growth factor and a velocity normalisation fixed at the redshift they were generated\n"
          "for. Starting from a different epoch silently produces a wrong simulation -- nothing in\n"
          "the output reports it, and the error is usually noticed only after the run completes.\n"
          "Either set TimeBegin = %.10g to match the ICs, or regenerate the ICs for a = %.10g.\n"
          "(Not checked for ComovingIntegrationOn = 0, where the clock origin is arbitrary.)\n\n",
          icTime, 1.0/icTime - 1.0,
          gadgetParams.TimeBegin, 1.0/gadgetParams.TimeBegin - 1.0,
          relDiff, icTime, gadgetParams.TimeBegin);
        ::exit(1);
      }
    }

    tree->set_t_current((float) snapHeader.time);
  }
  else if ((nPlummer == -1 && nSphere == -1  && nCube == -1 && !diskmode && nMilkyWay == -1 && !haveGadgetIC) || restartSim)
  {
    float sTime = 0;
    tree->fileIO->readFile(mpiCommWorld, bodyPositions, bodyVelocities, bodyIDs, fileName,
                           procId, nProcs, sTime, reduce_bodies_factor, restartSim);
    tree->set_t_current((float) sTime);
    #if USE_MPI
        float tCurrent = tree->get_t_current();
        MPI_Bcast(&tCurrent, 1, MPI_FLOAT, 0,mpiCommWorld);
        tree->set_t_current(tCurrent);
    #endif
    if (snapshotIter > 0) tree->set_nextSnapTime(tree->get_t_current() + snapshotIter);
  }
  else if(nMilkyWay >= 0)
  {
    #ifdef GALACTICS
        if (taskVar.empty())
        {
          tStartModel   = get_time_main();

          generateGalacticsModel(procId, nProcs, galSeed, nMilkyWay, nMWfork,
                                 true, bodyPositions, bodyVelocities, bodyIDs);
          tEndModel   = get_time_main();
        }
        else
        {
          //Scale mass of previously generated model
          const int ntot = bodyPositions.size();
          for (int i= 0; i < ntot; i++)
            bodyPositions[i].w *= 1.0/(double)nProcs;
        }
    #else
          assert(0);
    #endif
  }
  else if(nPlummer >= 0)
  {
    generatePlummerModel(bodyPositions, bodyVelocities, bodyIDs, procId, nProcs, nPlummer);
  }
  else if (nSphere >= 0)
  {
    generateSphereModel(bodyPositions, bodyVelocities, bodyIDs, procId, nProcs, nSphere);
  }//else
  else if (nCube >= 0)
  {
    generateCubeModel(bodyPositions, bodyVelocities, bodyIDs, procId, nProcs, nCube);
  }//else
  else if (diskmode)
  {
    generateShuffledDiskModel(bodyPositions, bodyVelocities, bodyIDs, procId, nProcs, fileName);
  }
  else
    assert(0);

  tree->mpiSync();

#if defined(PMGRID) && !defined(PERIODIC)
  // Isolated TreePM's own required convention (pm.h's PMIsolatedSolver doc comment, LOG.md §34):
  // particles must lie within [0, meshSize/2) in each axis, not wherever the IC generator happens
  // to center them. Every built-in generator (Plummer/Cube/Sphere) centers its distribution at
  // the origin, so a uniform +meshSize/4 shift on every axis lands it in the middle of the
  // occupied first half with equal padding on both sides -- the simplest valid placement, not
  // something a real user-supplied IC (once Phase 5's snapshot I/O exists) should be assumed to
  // need verbatim.
  for (unsigned int i = 0; i < bodyPositions.size(); i++)
  {
    bodyPositions[i].x += meshSize * 0.25f;
    bodyPositions[i].y += meshSize * 0.25f;
    bodyPositions[i].z += meshSize * 0.25f;
  }
#endif

  //Sanity check
  double mass = 0;
  for(unsigned int i=0; i < bodyPositions.size(); i++) { mass += bodyPositions[i].w; }
  double totalMass = tree->SumOnRootRank(mass);

  tree->mpiSumParticleCount((int)bodyPositions.size());

  LOGF(stderr, "t_current = %g nLocal %d massLocal: %f Combined Mass: %f nGlobal: %llu \n",
                tree->get_t_current(), (int)bodyPositions.size(),
                mass, totalMass, tree->nTotalFreq_ull);
  fprintf(stderr,"Proc: %d Bootup times: Tree/MPI: %lg Threads/log: %lg IC-model: %lg \n",
                 procId, tStartup-tStartupStart, tStartup2-tStartup, tEndModel - tStartModel);
  tree->load_kernels();

  double t0 = tree->get_time();

  tree->localTree.setN((int)bodyPositions.size());
  tree->allocateParticleMemory(tree->localTree);

  // Restart wiring. The IC is still read first -- that is what establishes the particle COUNT and
  // the per-type table -- and the restart file then overwrites the state arrays. readRestartFile
  // refuses on a count mismatch, so resuming against the wrong IC is caught rather than silently
  // producing a scrambled run.
  if (!restartFilePath.empty())
  {
    tree->restartPath       = restartFilePath;
    tree->restartCpuInterval = (haveGadgetParams && gadgetParams.CpuTimeBetRestartFile > 0.0)
                                 ? gadgetParams.CpuTimeBetRestartFile : 0.0;
    // lastRestartWrite is armed lazily on the first iteration (iterate_once) -- get_time() is a
    // member of the tree's own timing mixin and is not reachable from here.
  }

  // Phase 5 ticket 03 (PLAN.md): allocateParticleMemory() just defaulted every particle to type 1
  // (Halo/DM); override with the real per-particle types when a Gadget-2 IC supplied them.
  if (!bodyTypes.empty())
  {
    tree->localTree.bodies_type = bodyTypes;
    // Bug fix (Phase 5 ticket 06, LOG.md): record the STABLE (id -> type) association from this
    // original, pre-sort load order -- sort_bodies()'s SFC sort never reorders bodies_type itself,
    // only bodies_ids, so recomputeSoftening() must rebuild bodies_type from bodies_ids through
    // this map every rebuild rather than trusting bodies_type's own (never-reordered) index.
    tree->setGadgetTypeMap(bodyIDs, bodyTypes);
  }

  //Load data onto the device
  for(uint i=0; i < bodyPositions.size(); i++)
  {
    // bodies_pos is bound to bodies_Ppos now (see gpu_iterate.cpp) -- one array, drifted in
    // place, as Gadget-2 does. This loop used to write the identical value to both, which is the
    // clearest statement that they were always the same thing.
    tree->localTree.bodies_Ppos[i] = bodyPositions[i];
    tree->localTree.bodies_vel[i]  = bodyVelocities[i];
    tree->localTree.bodies_Pvel[i] = bodyVelocities[i];
    tree->localTree.bodies_ids[i]  = bodyIDs[i];
    tree->localTree.bodies_time[i] = make_float2(tree->get_t_current(), tree->get_t_current());
  }

  tree->localTree.bodies_time.h2d();
  tree->localTree.bodies_vel. h2d();
  tree->localTree.bodies_Ppos.h2d();
  tree->localTree.bodies_Pvel.h2d();
  tree->localTree.bodies_ids. h2d();

  // Resume AFTER the IC upload: the IC established the particle count and the (id -> type) map,
  // and the restart file now overwrites the state arrays with the saved ones. readRestartFile
  // refuses on a count or setup mismatch rather than producing a scrambled continuation.
  if (resumeFromRestart)
  {
    if (restartFilePath.empty())
    {
      fprintf(stderr, "\nFATAL: --resume given without --restart-file.\n\n");
      return 1;
    }
    if (!tree->readRestartFile(restartFilePath.c_str())) return 1;
  }

#ifdef PERIODIC
  // Must run before any gravity kernel launch (including the --pm-test-* diagnostics below, in
  // case a future one exercises the tree path) -- device context is already up at this point
  // (tree construction + the h2d() calls above already required it). Rcut/Asmth derived from
  // boxSize using Gadget-2's own compile-time constants (ASMTH=1.25, RCUT=4.5, allvars.h:83,87),
  // not independently tunable -- GADGET2_NOTES.md.
  {
    const float asmth = 1.25f * boxSize / PMGRID;
    const float rcut  = 4.5f * asmth;
    fprintf(stderr, "[PERIODIC] boxSize=%g PMGRID=%d Asmth=%g Rcut=%g\n", boxSize, PMGRID, asmth, rcut);
    pm_periodic_setup_gravity_kernel(boxSize, rcut, asmth);
    pm_set_dump_context(boxSize, PMGRID);
  }
#endif

#if defined(PMGRID) && !defined(PERIODIC)
  // Isolated-TreePM counterpart to the PERIODIC setup above (LOG.md §34) -- Gadget-2 never runs
  // PM without a paired tree short-range correction (confirmed with the user directly), so this
  // closes that gap for the non-periodic case. Rcut/Asmth derived from meshSize/GRID (GRID =
  // 2*PMGRID, the doubled isolated-solver grid), matching pm_nonperiodic.c:125-126's own formula
  // shape exactly (same ASMTH=1.25/RCUT=4.5 compile-time constants as periodic).
  {
    const int   fftGrid = 2 * PMGRID;
    const float asmth   = 1.25f * meshSize / fftGrid;
    const float rcut    = 4.5f * asmth;
    fprintf(stderr, "[ISOLATED] meshSize=%g PMGRID=%d fftGrid=%d Asmth=%g Rcut=%g\n",
            meshSize, PMGRID, fftGrid, asmth, rcut);
    pm_isolated_setup_gravity_kernel(rcut, asmth);
    pm_set_dump_context(meshSize, PMGRID);
  }
#endif

  // Ticket T27 (gadget-audit map): softening-based bJ floor for split_node_grav_springel,
  // unconditional -- unlike the PM/PERIODIC setup above, this needs no box size or PM grid, so it
  // must run for every build (including the non-periodic, non-PMGRID isolated Plummer tests) not
  // just PERIODIC/PMGRID ones. Default coefficient is 0.0f (floor disabled, i.e. no behavior
  // change) unless overridden via GADGET_HIP_T27_SOFT_COEF -- see t27_set_soft_floor_coef's own
  // comment (dev_approximate_gravity_warp_new.cu) for the full env-var contract.
  t27_set_soft_floor_coef(0.0f);

  // T6 (contract C-D-01): drift anchor. Off by default pending the A/B in ticket T6.
  {
    const char *e = getenv("GADGET_HIP_T6_DRIFT_TP");
    t6_set_drift_anchor(e ? atoi(e) : 1);
  }

  // T9 (contracts C-A-01 / C-B-06 / C-B-10b): fixed-cell node size in each MAC. BOTH ON by default
  // as of the C-A-01 pass -- the comment here used to say "both off by default" and was already
  // stale for the geometric one.
  //
  // C-A-01: Gadget-2's node size is the fixed octree cube side `len` (forcetree.c:106, :190-206),
  // never fitted to the particles. The AABB alternative is always <= len, so it opens FEWER nodes
  // than the requested theta implies -- the port's effective theta was silently coarser than the
  // parameter file states. cellSizeInfo is exactly `len`, and the C-A-01 audit
  // (GADGET_HIP_CA01_AUDIT) confirmed it directly on real trees: zero containment violations and
  // zero alignment violations over every node, so the level field does carry a uniform fixed-cell
  // meaning -- which is what T26 doubted and never tested.
  //
  // The relative MAC's default was previously 0 because T26 saw its blowup return when it tried
  // this. That negative result did not transfer: T26 predated the T4/C-B-02 G-convention fix, which
  // had left the primary relative test nearly inert. Re-run now, the blowup does NOT reproduce
  // (257-step cosmological repro completes, max|de| 0.439549 vs 0.439553), the MAC's own force error
  // improves at every percentile against a near-exact reference (median 1.46x, p99 1.74x), and it
  // costs +1.67% wall. See tickets/T33-ca01-node-size.md.
  {
    const char *g = getenv("GADGET_HIP_T9_CELLSIZE_GEO");
    const char *b = getenv("GADGET_HIP_T9_CELLSIZE_BJ");
    t9_set_cellsize_geo(g ? atoi(g) : 1);
    t9_set_cellsize_bj (b ? atoi(b) : 1);
    const char *ns = getenv("GADGET_HIP_T9_NO_S");
    t9_set_no_s(ns ? atoi(ns) : 1);
    const char *pc = getenv("GADGET_HIP_T11_PROX_CENTER");
    t11_set_prox_center(pc ? atoi(pc) : 1);
    // C-A-01 part (b): the proximity box's centre. T11 moved it from the centre of mass to the
    // AABB midpoint; this moves it the rest of the way, to the FIXED octree-cell centre, which is
    // the point Gadget-2's `nop->center` actually is. Same defect and same cause as the size term
    // T33 fixed -- a particle-fitted quantity standing in for a geometric one.
    const char *cc = getenv("GADGET_HIP_CA01_CELL_CENTER");
    ca01_set_cell_center(cc ? atoi(cc) : 1);
    const char *ccs = getenv("GADGET_HIP_CA01_CC_SKIP");
    ca01_set_cc_skip(ccs ? atoi(ccs) : 0);
    const char *k2 = getenv("GADGET_HIP_CD02_KICK_TEND");
    const char *k3 = getenv("GADGET_HIP_CD03_STRICT_ACTIVE");
    cd02_set_kick_tend(k2 ? atoi(k2) : 1);
    cd03_set_strict_active(k3 ? atoi(k3) : 1);
    // C-D-03b is a correctness fix (index-space mismatch, same class as the old T12/T23/T27
    // bugs). Default ON: verified bit-identical on the synchronous path and it is what makes the
    // individual-timestep path advance at all. GADGET_HIP_CD03B_FIX_INDEX=0 restores the old bug.
    const char *k3b = getenv("GADGET_HIP_CD03B_FIX_INDEX");
    cd03b_set_fix_index(k3b ? atoi(k3b) : 1);
    const char *ee = getenv("GADGET_HIP_ENERGY_EXTRAP");
    energy_set_extrap(ee ? atoi(ee) : 1);
    const char *cl = getenv("GADGET_HIP_CD04_LADDER");
    cd04_set_timeline_ladder(cl ? atoi(cl) : 1);
  }


  // Phase 4 (PLAN.md): one-shot CIC mass-assignment correctness check, same diagnostic idiom as
  // GADGET_HIP_DUMP_ACC(_WARM) -- exits immediately after printing the result, not a real
  // simulation step. gridPitch = 2*(gridSize/2+1) matches Gadget-2's own PMGRID2 padding
  // convention for an in-place real-to-complex FFT (GADGET2_NOTES.md); no FFT is actually run
  // here yet, just the CIC deposit, so this checks the kernel in isolation before it's wired into
  // a full PM step.
  if (pmTestCicGrid > 0)
  {
    const int gridSize  = pmTestCicGrid;
    const int gridPitch = 2 * (gridSize / 2 + 1);
    const size_t gridElems = (size_t)gridSize * gridSize * gridPitch;

    float *d_density = nullptr;
    CU_SAFE_CALL(hipMalloc((void**)&d_density, gridElems * sizeof(float)));
    CU_SAFE_CALL(hipMemset(d_density, 0, gridElems * sizeof(float)));

    pm_cic_assign_mass(tree->localTree.bodies_Ppos.raw_p(), (int)bodyPositions.size(), d_density,
                        gridSize, gridPitch, pmTestCicBoxSize, 0);
    CU_SAFE_CALL(hipDeviceSynchronize());

    std::vector<float> h_density(gridElems);
    CU_SAFE_CALL(hipMemcpy(h_density.data(), d_density, gridElems * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(d_density);

    double gridMass = 0;
    for (size_t i = 0; i < gridElems; i++)
      gridMass += h_density[i];

    const double relErr = fabs(gridMass - totalMass) / totalMass;
    fprintf(stderr, "[PM_TEST_CIC] gridSize=%d boxSize=%g nBodies=%zu totalMass=%.9g gridMass=%.9g relErr=%.3e\n",
            gridSize, pmTestCicBoxSize, bodyPositions.size(), totalMass, gridMass, relErr);
    ::exit(relErr < 1e-5 ? 0 : 1);
  }

#ifdef PMGRID
  // Phase 4 (PLAN.md): rocFFT forward+inverse round-trip check. No Green's function, no
  // particles -- pure plan-creation/stride/normalization correctness, isolated from the physics.
  // Seeds a unique, non-symmetric value per cell (not just noise) so a transposition or stride
  // bug between axes shows up as a clear per-cell mismatch rather than being masked by symmetry.
  if (pmTestFftGrid > 0)
  {
    const int gridSize  = pmTestFftGrid;
    const int gridPitch = 2 * (gridSize / 2 + 1);
    const size_t gridElems = (size_t)gridSize * gridSize * gridPitch;

    std::vector<float> h_grid(gridElems, 0.0f);
    for (int x = 0; x < gridSize; x++)
      for (int y = 0; y < gridSize; y++)
        for (int z = 0; z < gridSize; z++)
          h_grid[(size_t)(x * gridSize + y) * gridPitch + z] =
              sinf(0.37f * x + 1.0f) * cosf(0.53f * y + 2.0f) * sinf(0.71f * z + 3.0f);

    float *d_grid = nullptr;
    CU_SAFE_CALL(hipMalloc((void**)&d_grid, gridElems * sizeof(float)));
    CU_SAFE_CALL(hipMemcpy(d_grid, h_grid.data(), gridElems * sizeof(float), hipMemcpyHostToDevice));

    pm_fft_init();
    PMFFTPlans plans = pm_fft_create_plans(gridSize);
    pm_fft_forward(plans, d_grid, 0);
    pm_fft_inverse(plans, d_grid, 0);
    CU_SAFE_CALL(hipDeviceSynchronize());

    std::vector<float> h_result(gridElems);
    CU_SAFE_CALL(hipMemcpy(h_result.data(), d_grid, gridElems * sizeof(float), hipMemcpyDeviceToHost));
    pm_fft_destroy_plans(plans);
    hipFree(d_grid);

    double maxAbsErr = 0, maxVal = 0;
    for (int x = 0; x < gridSize; x++)
      for (int y = 0; y < gridSize; y++)
        for (int z = 0; z < gridSize; z++)
        {
          const size_t idx = (size_t)(x * gridSize + y) * gridPitch + z;
          const double err = fabs((double)h_result[idx] - (double)h_grid[idx]);
          maxAbsErr = std::max(maxAbsErr, err);
          maxVal = std::max(maxVal, (double)fabs(h_grid[idx]));
        }
    const double relErr = maxAbsErr / maxVal;
    fprintf(stderr, "[PM_TEST_FFT] gridSize=%d maxAbsErr=%.3e maxVal=%.3e relErr=%.3e\n",
            gridSize, maxAbsErr, maxVal, relErr);
    ::exit(relErr < 1e-4 ? 0 : 1);
  }

  // Phase 4 (PLAN.md): Green's-function Poisson-solve check. Seeds a single analytic cosine
  // density mode directly (bypassing CIC entirely -- applyCicDeconvolution=false), runs the full
  // forward-FFT -> Green's-function -> inverse-FFT pipeline, and compares against the closed-form
  // expected potential for that mode: rho = A*cos(2*pi*m*x/gridSize) has an FFT supported only at
  // k=(m,0,0) and its mirror, so after the Green's-function multiply the inverse transform must
  // give back exactly smth(k=m) * A * cos(2*pi*m*x/gridSize) -- smth computed here with the exact
  // same formula pm_greens_multiply_periodic uses (GADGET2_NOTES.md), so this checks the pipeline
  // wiring/k-indexing, not a re-derivation of the physics.
  if (pmTestPoissonGrid > 0)
  {
    const int gridSize  = pmTestPoissonGrid;
    const int gridPitch = 2 * (gridSize / 2 + 1);
    const size_t gridElems = (size_t)gridSize * gridSize * gridPitch;
    const float boxSize = pmTestPoissonBoxSize;
    const float A = 0.01f;
    const int m = 1; // lowest nonzero mode along x, ky=kz=0

    std::vector<float> h_rho(gridElems, 0.0f);
    for (int x = 0; x < gridSize; x++)
    {
      const float rho = A * cosf(2.0f * (float)M_PI * m * x / gridSize);
      for (int y = 0; y < gridSize; y++)
        for (int z = 0; z < gridSize; z++)
          h_rho[(size_t)(x * gridSize + y) * gridPitch + z] = rho;
    }

    float *d_grid = nullptr;
    CU_SAFE_CALL(hipMalloc((void**)&d_grid, gridElems * sizeof(float)));
    CU_SAFE_CALL(hipMemcpy(d_grid, h_rho.data(), gridElems * sizeof(float), hipMemcpyHostToDevice));

    pm_fft_init();
    PMFFTPlans plans = pm_fft_create_plans(gridSize);
    pm_fft_forward(plans, d_grid, 0);
    pm_greens_multiply_periodic(d_grid, gridSize, boxSize, /*applyCicDeconvolution=*/false, 0);
    pm_fft_inverse(plans, d_grid, 0);
    CU_SAFE_CALL(hipDeviceSynchronize());

    std::vector<float> h_result(gridElems);
    CU_SAFE_CALL(hipMemcpy(h_result.data(), d_grid, gridElems * sizeof(float), hipMemcpyDeviceToHost));
    pm_fft_destroy_plans(plans);
    hipFree(d_grid);

    const float asmth  = 1.25f * boxSize / gridSize;
    const float asmth2 = (2.0f * (float)M_PI * asmth / boxSize) * (2.0f * (float)M_PI * asmth / boxSize);
    const float k2 = (float)(m * m);
    const float smth = -expf(-k2 * asmth2) / k2;

    double maxAbsErr = 0, maxVal = 0;
    for (int x = 0; x < gridSize; x++)
    {
      const float expected = smth * A * cosf(2.0f * (float)M_PI * m * x / gridSize);
      for (int y = 0; y < gridSize; y++)
        for (int z = 0; z < gridSize; z++)
        {
          const size_t idx = (size_t)(x * gridSize + y) * gridPitch + z;
          const double err = fabs((double)h_result[idx] - (double)expected);
          maxAbsErr = std::max(maxAbsErr, err);
          maxVal = std::max(maxVal, (double)fabs(expected));
        }
    }
    const double relErr = maxAbsErr / maxVal;
    fprintf(stderr, "[PM_TEST_POISSON] gridSize=%d boxSize=%g smth(k=%d)=%.6g maxAbsErr=%.3e maxVal=%.3e relErr=%.3e\n",
            gridSize, boxSize, m, smth, maxAbsErr, maxVal, relErr);
    ::exit(relErr < 1e-3 ? 0 : 1);
  }

  // Phase 4 (PLAN.md): finite-difference force extraction + CIC force interpolation check.
  // Reuses --pm-test-poisson's seeded cosine density and validated FFT+Green's-function pipeline
  // to get a trusted "raw_solve" potential grid, then checks two more pieces against it:
  //  (1) the finite-difference stencil, axis=x, against the EXACT closed form for a 4-point
  //      stencil applied to a discretely-sampled cosine (not a continuum approximation -- see the
  //      derivation in the comment below), plus axes y/z, which must come out ~zero since the
  //      seeded density has no y/z dependence;
  //  (2) CIC force interpolation, by comparing the GPU kernel's output at a handful of grid-
  //      aligned and fractional-offset test points against a straightforward host-side trilinear
  //      re-implementation reading the same (now-trusted) force grid.
  if (pmTestForceGrid > 0)
  {
    const int gridSize  = pmTestForceGrid;
    const int gridPitch = 2 * (gridSize / 2 + 1);
    const size_t gridElems = (size_t)gridSize * gridSize * gridPitch;
    const float boxSize = pmTestForceBoxSize;
    const float A = 0.01f;
    const int m = 1;
    const float theta = 2.0f * (float)M_PI * m / gridSize;

    std::vector<float> h_rho(gridElems, 0.0f);
    for (int x = 0; x < gridSize; x++)
    {
      const float rho = A * cosf(theta * x);
      for (int y = 0; y < gridSize; y++)
        for (int z = 0; z < gridSize; z++)
          h_rho[(size_t)(x * gridSize + y) * gridPitch + z] = rho;
    }

    float *d_potential = nullptr, *d_force_x = nullptr, *d_force_scratch = nullptr;
    CU_SAFE_CALL(hipMalloc((void**)&d_potential, gridElems * sizeof(float)));
    CU_SAFE_CALL(hipMalloc((void**)&d_force_x, gridElems * sizeof(float)));
    CU_SAFE_CALL(hipMalloc((void**)&d_force_scratch, gridElems * sizeof(float)));
    CU_SAFE_CALL(hipMemcpy(d_potential, h_rho.data(), gridElems * sizeof(float), hipMemcpyHostToDevice));

    pm_fft_init();
    PMFFTPlans plans = pm_fft_create_plans(gridSize);
    pm_fft_forward(plans, d_potential, 0);
    pm_greens_multiply_periodic(d_potential, gridSize, boxSize, /*applyCicDeconvolution=*/false, 0);
    pm_fft_inverse(plans, d_potential, 0);

    pm_finite_diff_force(d_potential, d_force_x, gridSize, gridPitch, /*axis=*/0, boxSize, 0);
    pm_finite_diff_force(d_potential, d_force_scratch, gridSize, gridPitch, /*axis=*/1, boxSize, 0);
    CU_SAFE_CALL(hipDeviceSynchronize());
    std::vector<float> h_force_y(gridElems);
    CU_SAFE_CALL(hipMemcpy(h_force_y.data(), d_force_scratch, gridElems * sizeof(float), hipMemcpyDeviceToHost));

    pm_finite_diff_force(d_potential, d_force_scratch, gridSize, gridPitch, /*axis=*/2, boxSize, 0);
    CU_SAFE_CALL(hipDeviceSynchronize());
    std::vector<float> h_force_z(gridElems);
    CU_SAFE_CALL(hipMemcpy(h_force_z.data(), d_force_scratch, gridElems * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> h_force_x(gridElems);
    CU_SAFE_CALL(hipMemcpy(h_force_x.data(), d_force_x, gridElems * sizeof(float), hipMemcpyDeviceToHost));

    // Exact closed form: raw_solve(x) = B*cos(theta*x), B = smth(k=m)*A. The 4-point stencil
    // (4/3)(phi(x-1)-phi(x+1)) - (1/6)(phi(x-2)-phi(x+2)) applied to a discretely-sampled cosine
    // has an EXACT (not approximate) result via cos(theta(x-1))-cos(theta(x+1)) = 2 sin(theta*x)
    // sin(theta), and similarly for the +-2 terms with sin(2*theta):
    //   stencil(x) = 2*B*sin(theta*x) * [ (4/3)*sin(theta) - (1/6)*sin(2*theta) ]
    // force(x) = gridScale * stencil(x), gridScale = gridSize/(2*boxSize) (pm_finite_diff_force's
    // own normalization, matched here rather than re-derived independently).
    const float asmth  = 1.25f * boxSize / gridSize;
    const float asmth2 = (2.0f * (float)M_PI * asmth / boxSize) * (2.0f * (float)M_PI * asmth / boxSize);
    const float k2 = (float)(m * m);
    const float smth = -expf(-k2 * asmth2) / k2;
    const float B = smth * A;
    const float gridScale = gridSize / (2.0f * boxSize);
    const float stencilCoeff = (4.0f / 3.0f) * sinf(theta) - (1.0f / 6.0f) * sinf(2.0f * theta);

    double maxAbsErrX = 0, maxValX = 0;
    for (int x = 0; x < gridSize; x++)
    {
      const float expected = gridScale * 2.0f * B * sinf(theta * x) * stencilCoeff;
      for (int y = 0; y < gridSize; y++)
        for (int z = 0; z < gridSize; z++)
        {
          const size_t idx = (size_t)(x * gridSize + y) * gridPitch + z;
          const double err = fabs((double)h_force_x[idx] - (double)expected);
          maxAbsErrX = std::max(maxAbsErrX, err);
          maxValX = std::max(maxValX, (double)fabs(expected));
        }
    }
    const double relErrX = maxAbsErrX / maxValX;

    double maxAbsY = 0, maxAbsZ = 0;
    for (size_t i = 0; i < gridElems; i++)
    {
      maxAbsY = std::max(maxAbsY, (double)fabs(h_force_y[i]));
      maxAbsZ = std::max(maxAbsZ, (double)fabs(h_force_z[i]));
    }

    // CIC interpolation check: a handful of grid-aligned and fractional-offset test points,
    // compared against a plain host-side trilinear re-implementation reading h_force_x (already
    // validated above) -- exercises pm_cic_interpolate's addressing/weighting independently.
    const int nTest = 6;
    std::vector<float4> h_testPos(nTest);
    const float cellSize = boxSize / gridSize;
    h_testPos[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    h_testPos[1] = make_float4((gridSize / 4) * cellSize, 0.0f, 0.0f, 0.0f);
    h_testPos[2] = make_float4(((gridSize / 4) + 0.37f) * cellSize, 0.6f * cellSize, 0.0f, 0.0f);
    h_testPos[3] = make_float4(((gridSize / 2) + 0.5f) * cellSize, 1.2f * cellSize, 2.8f * cellSize, 0.0f);
    h_testPos[4] = make_float4((gridSize - 0.25f) * cellSize, 0.0f, 0.0f, 0.0f); // wraps past the box edge
    h_testPos[5] = make_float4(-0.6f * cellSize, -1.3f * cellSize, 0.0f, 0.0f);  // negative position, must still wrap

    float4 *d_testPos = nullptr;
    float *d_testOut = nullptr;
    CU_SAFE_CALL(hipMalloc((void**)&d_testPos, nTest * sizeof(float4)));
    CU_SAFE_CALL(hipMalloc((void**)&d_testOut, nTest * sizeof(float)));
    CU_SAFE_CALL(hipMemcpy(d_testPos, h_testPos.data(), nTest * sizeof(float4), hipMemcpyHostToDevice));

    pm_cic_interpolate(d_testPos, nTest, d_force_x, d_testOut, gridSize, gridPitch, boxSize, 0);
    CU_SAFE_CALL(hipDeviceSynchronize());

    std::vector<float> h_testOut(nTest);
    CU_SAFE_CALL(hipMemcpy(h_testOut.data(), d_testOut, nTest * sizeof(float), hipMemcpyDeviceToHost));

    double maxAbsErrCic = 0, maxValCic = 0;
    for (int t = 0; t < nTest; t++)
    {
      const float invCellSize = gridSize / boxSize;
      const float gx = h_testPos[t].x * invCellSize;
      const float gy = h_testPos[t].y * invCellSize;
      const float gz = h_testPos[t].z * invCellSize;
      const int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
      const float dx = gx - ix, dy = gy - iy, dz = gz - iz;
      const int ix0 = ((ix % gridSize) + gridSize) % gridSize;
      const int iy0 = ((iy % gridSize) + gridSize) % gridSize;
      const int iz0 = ((iz % gridSize) + gridSize) % gridSize;
      const int ix1 = (ix0 + 1) % gridSize, iy1 = (iy0 + 1) % gridSize, iz1 = (iz0 + 1) % gridSize;
      auto at = [&](int xx, int yy, int zz) { return h_force_x[(size_t)(xx * gridSize + yy) * gridPitch + zz]; };
      const float expected =
          at(ix0,iy0,iz0)*(1-dx)*(1-dy)*(1-dz) + at(ix0,iy0,iz1)*(1-dx)*(1-dy)*dz +
          at(ix0,iy1,iz0)*(1-dx)*dy*(1-dz)     + at(ix0,iy1,iz1)*(1-dx)*dy*dz +
          at(ix1,iy0,iz0)*dx*(1-dy)*(1-dz)     + at(ix1,iy0,iz1)*dx*(1-dy)*dz +
          at(ix1,iy1,iz0)*dx*dy*(1-dz)         + at(ix1,iy1,iz1)*dx*dy*dz;
      const double err = fabs((double)h_testOut[t] - (double)expected);
      maxAbsErrCic = std::max(maxAbsErrCic, err);
      maxValCic = std::max(maxValCic, (double)fabs(expected));
    }
    const double relErrCic = maxAbsErrCic / std::max(maxValCic, 1e-30);

    pm_fft_destroy_plans(plans);
    hipFree(d_potential); hipFree(d_force_x); hipFree(d_force_scratch);
    hipFree(d_testPos); hipFree(d_testOut);

    fprintf(stderr, "[PM_TEST_FORCE] gridSize=%d boxSize=%g axisX_relErr=%.3e axisY_maxAbs=%.3e "
                     "axisZ_maxAbs=%.3e cic_relErr=%.3e\n",
            gridSize, boxSize, relErrX, maxAbsY, maxAbsZ, relErrCic);
    const bool pass = relErrX < 1e-3 && maxAbsY < 1e-3 * maxValX && maxAbsZ < 1e-3 * maxValX && relErrCic < 1e-4;
    ::exit(pass ? 0 : 1);
  }

  // Phase 4 (PLAN.md): first end-to-end test of the WIRED pipeline (pm_compute_forces_periodic),
  // via real CIC deposit (deconvolution ON, unlike the synthetic-seeding tests above) on a
  // uniform particle lattice displaced sinusoidally along x -- exactly the Zel'dovich
  // approximation's linear displacement ansatz, x = x0 + Sg*sin(theta*x0). By 1D continuity
  // (mass conservation under the coordinate map x0->x), this produces a density perturbation
  // rho(x) ~= rho0*(1 - Sg*theta*cos(theta*x0)) to leading order in Sg -- i.e. a single cosine
  // mode, letting this reuse the exact closed-form force formula already validated by
  // --pm-test-force. Two independent checks, not one, to isolate "did the displacement produce
  // the predicted density" from "does the solver correctly turn a given density into the right
  // force": (1) the ACTUAL Fourier amplitude of the CIC-deposited mass grid is measured directly
  // (discrete cosine projection) and compared to the Sg*theta linear-theory prediction; (2) the
  // measured (not predicted) amplitude is then used to compute the expected force, compared
  // against the solver's real output. Unlike the fully-synthetic tests above, exact machine-
  // precision agreement is NOT expected here -- finite-N CIC discretization noise is real and
  // well-understood (matches the force-error levels Springel's own paper reports for tree/PM
  // codes), so this reports the measured errors rather than asserting a strict pass/fail.
  if (pmTestSineGrid > 0)
  {
    const int gridSize   = pmTestSineGrid;
    const int gridPitch  = 2 * (gridSize / 2 + 1);
    const float boxSize  = pmTestSineBoxSize;
    const float Sg       = pmTestSineAmplitude;
    const int m          = 1;
    const float theta    = 2.0f * (float)M_PI * m / gridSize;
    const float cellSize = boxSize / gridSize;

    const int Np = gridSize; // one particle per PM cell, unperturbed
    const size_t nBodies = (size_t)Np * Np * Np;
    const float massPerParticle = 1.0f / (float)nBodies; // total mass = 1

    std::vector<float4> h_pos(nBodies);
    size_t idx = 0;
    for (int ix = 0; ix < Np; ix++)
    {
      const float x0 = ix + 0.5f; // cell-centered, grid-index units
      const float xg = x0 + Sg * sinf(theta * x0);
      const float xPhys = xg * cellSize;
      for (int iy = 0; iy < Np; iy++)
      {
        const float yPhys = (iy + 0.5f) * cellSize;
        for (int iz = 0; iz < Np; iz++)
        {
          const float zPhys = (iz + 0.5f) * cellSize;
          h_pos[idx++] = make_float4(xPhys, yPhys, zPhys, massPerParticle);
        }
      }
    }

    float4 *d_pos = nullptr;
    CU_SAFE_CALL(hipMalloc((void**)&d_pos, nBodies * sizeof(float4)));
    CU_SAFE_CALL(hipMemcpy(d_pos, h_pos.data(), nBodies * sizeof(float4), hipMemcpyHostToDevice));

    // Check 1: measure the actual CIC-deposited mass grid's cosine amplitude.
    const size_t gridElems = (size_t)gridSize * gridSize * gridPitch;
    float *d_massGrid = nullptr;
    CU_SAFE_CALL(hipMalloc((void**)&d_massGrid, gridElems * sizeof(float)));
    CU_SAFE_CALL(hipMemset(d_massGrid, 0, gridElems * sizeof(float)));
    pm_cic_assign_mass(d_pos, (int)nBodies, d_massGrid, gridSize, gridPitch, boxSize, 0);
    CU_SAFE_CALL(hipDeviceSynchronize());
    std::vector<float> h_massGrid(gridElems);
    CU_SAFE_CALL(hipMemcpy(h_massGrid.data(), d_massGrid, gridElems * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(d_massGrid);

    double cosProjection = 0;
    for (int x = 0; x < gridSize; x++)
      for (int y = 0; y < gridSize; y++)
        for (int z = 0; z < gridSize; z++)
          cosProjection += (double)h_massGrid[(size_t)(x * gridSize + y) * gridPitch + z] * cos(theta * x);
    const double A_measured = (2.0 / ((double)gridSize * gridSize * gridSize)) * cosProjection;
    const double A_predicted = -(1.0 / ((double)gridSize * gridSize * gridSize)) * Sg * theta;
    const double A_relDiff = fabs(A_measured - A_predicted) / fabs(A_predicted);

    // Check 2: run the real wired pipeline, compare force_x against the closed-form formula
    // using the MEASURED amplitude above (decoupling this check from check 1's own accuracy).
    pm_fft_init();
    PMPeriodicSolver solver = pm_solver_create_periodic(gridSize, boxSize);
    float *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
    CU_SAFE_CALL(hipMalloc((void**)&d_fx, nBodies * sizeof(float)));
    CU_SAFE_CALL(hipMalloc((void**)&d_fy, nBodies * sizeof(float)));
    CU_SAFE_CALL(hipMalloc((void**)&d_fz, nBodies * sizeof(float)));
    pm_compute_forces_periodic(solver, d_pos, (int)nBodies, d_fx, d_fy, d_fz, 0);
    CU_SAFE_CALL(hipDeviceSynchronize());

    std::vector<float> h_fx(nBodies), h_fy(nBodies), h_fz(nBodies);
    CU_SAFE_CALL(hipMemcpy(h_fx.data(), d_fx, nBodies * sizeof(float), hipMemcpyDeviceToHost));
    CU_SAFE_CALL(hipMemcpy(h_fy.data(), d_fy, nBodies * sizeof(float), hipMemcpyDeviceToHost));
    CU_SAFE_CALL(hipMemcpy(h_fz.data(), d_fz, nBodies * sizeof(float), hipMemcpyDeviceToHost));

    const float asmth  = 1.25f * boxSize / gridSize;
    const float asmth2 = (2.0f * (float)M_PI * asmth / boxSize) * (2.0f * (float)M_PI * asmth / boxSize);
    const float k2 = (float)(m * m);
    const float smth = -expf(-k2 * asmth2) / k2;
    const float B = smth * (float)A_measured;
    const float gridScale = gridSize / (2.0f * boxSize);
    // gridSize^3 factor: matches the fix in pm_solve.cpp's pm_compute_forces_periodic
    // (LOG.md §27-28) -- this hand-derived "expected" formula independently re-implements the
    // same physicalScale the real solver applies, so it needs the identical correction, or this
    // check would silently go back to being blind to the exact bug that correction fixes.
    const float physicalScale = (float)gridSize * (float)gridSize * (float)gridSize /
                                 ((float)M_PI * boxSize); // G=1
    const float stencilCoeff = (4.0f / 3.0f) * sinf(theta) - (1.0f / 6.0f) * sinf(2.0f * theta);

    double sumAbsErr = 0, sumAbsExp = 0, maxRelErr = 0, maxAbsFy = 0, maxAbsFz = 0;
    for (size_t i = 0; i < nBodies; i++)
    {
      const float xg = h_pos[i].x / cellSize; // particle's own (displaced) grid-index position
      const float expected = physicalScale * gridScale * 2.0f * B * sinf(theta * xg) * stencilCoeff;
      const double err = fabs((double)h_fx[i] - (double)expected);
      sumAbsErr += err;
      sumAbsExp += fabs((double)expected);
      if (fabs((double)expected) > 1e-12)
        maxRelErr = std::max(maxRelErr, err / fabs((double)expected));
      maxAbsFy = std::max(maxAbsFy, (double)fabs(h_fy[i]));
      maxAbsFz = std::max(maxAbsFz, (double)fabs(h_fz[i]));
    }
    const double meanRelErr = sumAbsErr / sumAbsExp;

    pm_solver_destroy_periodic(solver);
    hipFree(d_pos); hipFree(d_fx); hipFree(d_fy); hipFree(d_fz);

    fprintf(stderr, "[PM_TEST_SINEWAVE] gridSize=%d boxSize=%g Sg=%g nBodies=%zu\n", gridSize, boxSize, Sg, nBodies);
    fprintf(stderr, "[PM_TEST_SINEWAVE]   density amplitude: measured=%.6e predicted=%.6e relDiff=%.3e\n",
            A_measured, A_predicted, A_relDiff);
    fprintf(stderr, "[PM_TEST_SINEWAVE]   force_x vs. linear theory (using measured amplitude): meanRelErr=%.3e maxRelErr=%.3e\n",
            meanRelErr, maxRelErr);
    fprintf(stderr, "[PM_TEST_SINEWAVE]   force_y/z leakage (should be small vs. force_x): maxAbsFy=%.3e maxAbsFz=%.3e\n",
            maxAbsFy, maxAbsFz);
    ::exit(0); // diagnostic report, not a strict pass/fail gate -- see comment above
  }
  // Phase 4 (PLAN.md): standalone correctness check for the non-periodic (isolated/vacuum-
  // boundary) PM solver -- independent of --pm-test-treepm-force below (PERIODIC-only). Ground
  // truth is plain Newtonian G*m/r^2: an isolated system has no periodic images to sum, so this
  // is a much simpler, exact reference than the periodic case's Ewald summation. Two-particle
  // setup (target mass=0, source mass=1) mirrors --pm-test-treepm-force's own trick of isolating
  // PM's field from its own self-force. Both particles are placed in the "occupied first half"
  // [0, meshSize/2) the isolated solver's own doc comment (pm.h) requires.
  if (pmTestIsolatedSamples > 0)
  {
    const int nSamples   = pmTestIsolatedSamples;
    const int gridSize   = PMGRID;
    const float meshSize = pmTestIsolatedMeshSize;
    const float cellSize = (meshSize) / (2.0f * gridSize); // physical cell size of the doubled grid

    // Log-spaced from a few cells (grid-resolution-limited near field) to a large fraction of the
    // occupied half-region (meshSize/2), staying well clear of the zero-padding boundary.
    const double rMin = 3.0 * cellSize;
    const double rMax = 0.4 * (meshSize / 2.0);

    pm_fft_init();
    PMIsolatedSolver solver = pm_solver_create_isolated(gridSize, meshSize);

    fprintf(stderr, "[PM_TEST_ISOLATED_FORCE] gridSize=%d meshSize=%g cellSize=%g nSamples=%d\n",
            gridSize, meshSize, cellSize, nSamples);
    fprintf(stderr, "[PM_TEST_ISOLATED_FORCE] %12s %14s %14s %10s\n",
            "sep", "pmFx", "newtonFx", "relErr");

    double sumRelErr = 0, maxRelErr = 0;
    int nCounted = 0;

    for (int s = 0; s < nSamples; s++)
    {
      const double logMin = log(rMin), logMax = log(rMax);
      const double sep = exp(logMin + (logMax - logMin) * s / (double)std::max(1, nSamples - 1));

      const float base = meshSize * 0.25f; // center of the occupied [0, meshSize/2) half
      float4 h_pos[2];
      h_pos[0] = make_float4(base, base, base, 0.0f);
      h_pos[1] = make_float4(base + (float)sep, base, base, 1.0f);

      float4 *d_pos = nullptr; float *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_pos, 2*sizeof(float4)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fx, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fy, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fz, 2*sizeof(float)));
      CU_SAFE_CALL(hipMemcpy(d_pos, h_pos, 2*sizeof(float4), hipMemcpyHostToDevice));
      pm_compute_forces_isolated(solver, d_pos, 2, d_fx, d_fy, d_fz, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());
      float h_fx0 = 0.0f;
      CU_SAFE_CALL(hipMemcpy(&h_fx0, d_fx, sizeof(float), hipMemcpyDeviceToHost));
      hipFree(d_pos); hipFree(d_fx); hipFree(d_fy); hipFree(d_fz);

      const double newtonFx = 1.0 / (sep * sep); // G=1, mass=1, matching this port's native units
      const double relErr = fabs((double)h_fx0 - newtonFx) / newtonFx;
      sumRelErr += relErr;
      maxRelErr = std::max(maxRelErr, relErr);
      nCounted++;

      fprintf(stderr, "[PM_TEST_ISOLATED_FORCE] %12.6g %14.6g %14.6g %10.3e\n",
              sep, (double)h_fx0, newtonFx, relErr);
    }

    pm_solver_destroy_isolated(solver);
    fprintf(stderr, "[PM_TEST_ISOLATED_FORCE] meanRelErr=%.3e maxRelErr=%.3e\n",
            sumRelErr / std::max(1, nCounted), maxRelErr);
    ::exit(0); // diagnostic report, not a strict pass/fail gate -- see comment above
  }
#ifdef PERIODIC
  // Phase 4 (PLAN.md) / LOG.md §27: the actual combined tree+PM correctness check -- every
  // individual PM piece and the periodic tree's own minimum-image/erfc suppression were validated
  // in isolation against internally-derived expectations, but never against an INDEPENDENT
  // reference until this test. Ground truth is an independent Ewald summation
  // (include/ewald_ref.h, a from-scratch, self-checked port of Gadget-2's own ewald_force()/
  // ewald_psi()), not the tree or PM code itself.
  //
  // Tree side: calls add_acc() directly (pm.h's pm_test_add_acc_pair) for a single
  // target-at-origin/source pair -- the exact same device function the real tree walk uses, but
  // with no tree build/domain-decomposition involved (sidesteps the known small-N domain-
  // decomposition crash noted elsewhere in this project). PM side: a real 2-particle
  // pm_compute_forces_periodic() call, target mass=0 so it only samples the source's field
  // (deliberately removes PM's own self-force artifact from this comparison, which is about the
  // tree/PM split, not PM's independently-already-characterized self-force noise).
  //
  // Reuses the program's own already-configured --boxsize/PMGRID (pm_periodic_setup_gravity_kernel
  // was already called with these above), not separate test-specific values, so this checks
  // exactly the Asmth/Rcut/table a real run would use.
  if (pmTestTreePMSamples > 0)
  {
    const int nSamples  = pmTestTreePMSamples;
    const int gridSize  = PMGRID;
    const float cellSize = boxSize / gridSize;

    // Log-spaced from well inside one PM cell (tree-dominated) to just under half the box
    // (PM-dominated), spanning the erfc short-range/long-range crossover (a few cells) in between.
    const double rMin = 0.05 * cellSize;
    const double rMax = 0.45 * boxSize;

    pm_fft_init();
    PMPeriodicSolver solver = pm_solver_create_periodic(gridSize, boxSize);

    fprintf(stderr, "[PM_TEST_TREEPM_FORCE] boxSize=%g PMGRID=%d cellSize=%g nSamples=%d\n",
            boxSize, gridSize, cellSize, nSamples);
    fprintf(stderr, "[PM_TEST_TREEPM_FORCE] %12s %14s %14s %14s %14s %10s\n",
            "sep", "treeFx", "pmFx", "totalFx", "ewaldFx", "relErr");

    double sumRelErr = 0, maxRelErr = 0;
    int nCounted = 0;

    for (int s = 0; s < nSamples; s++)
    {
      const double logMin = log(rMin), logMax = log(rMax);
      const double sep = exp(logMin + (logMax - logMin) * s / (double)std::max(1, nSamples - 1));

      // --- Tree short-range force: exact add_acc() call, no tree build. ---
      float3 h_sep  = make_float3((float)sep, 0.0f, 0.0f);
      float  h_mass = 1.0f;
      const int h_type = 0; // type=0 (Gas): irrelevant here, this test never enables the zoom mask
      float3 *d_sep = nullptr; float *d_srcMass = nullptr; int *d_type = nullptr; float4 *d_treeAcc = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_sep, sizeof(float3)));
      CU_SAFE_CALL(hipMalloc((void**)&d_srcMass, sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_type, sizeof(int)));
      CU_SAFE_CALL(hipMalloc((void**)&d_treeAcc, sizeof(float4)));
      CU_SAFE_CALL(hipMemcpy(d_sep, &h_sep, sizeof(float3), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_srcMass, &h_mass, sizeof(float), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_type, &h_type, sizeof(int), hipMemcpyHostToDevice));
      pm_test_add_acc_pair(d_sep, d_srcMass, d_type, 1, d_treeAcc, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());
      float4 h_treeAcc;
      CU_SAFE_CALL(hipMemcpy(&h_treeAcc, d_treeAcc, sizeof(float4), hipMemcpyDeviceToHost));
      hipFree(d_sep); hipFree(d_srcMass); hipFree(d_type); hipFree(d_treeAcc);

      // --- PM long-range force: 2-particle system, target mass=0 (see comment above). ---
      float4 h_pos[2];
      h_pos[0] = make_float4(boxSize*0.5f, boxSize*0.5f, boxSize*0.5f, 0.0f);
      h_pos[1] = make_float4(boxSize*0.5f + (float)sep, boxSize*0.5f, boxSize*0.5f, 1.0f);
      float4 *d_pos = nullptr; float *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_pos, 2*sizeof(float4)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fx, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fy, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fz, 2*sizeof(float)));
      CU_SAFE_CALL(hipMemcpy(d_pos, h_pos, 2*sizeof(float4), hipMemcpyHostToDevice));
      pm_compute_forces_periodic(solver, d_pos, 2, d_fx, d_fy, d_fz, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());
      float h_fx0 = 0.0f;
      CU_SAFE_CALL(hipMemcpy(&h_fx0, d_fx, sizeof(float), hipMemcpyDeviceToHost));
      hipFree(d_pos); hipFree(d_fx); hipFree(d_fy); hipFree(d_fz);

      const double totalFx = (double)h_treeAcc.x + (double)h_fx0;

      double d3[3] = { sep, 0.0, 0.0 };
      double refForce[3];
      ewald_ref::total_force(d3, (double)boxSize, refForce);

      const double relErr = fabs(totalFx - refForce[0]) / fabs(refForce[0]);
      sumRelErr += relErr;
      maxRelErr = std::max(maxRelErr, relErr);
      nCounted++;

      fprintf(stderr, "[PM_TEST_TREEPM_FORCE] %12.6g %14.6g %14.6g %14.6g %14.6g %10.3e\n",
              sep, (double)h_treeAcc.x, h_fx0, totalFx, refForce[0], relErr);
    }

    pm_solver_destroy_periodic(solver);
    fprintf(stderr, "[PM_TEST_TREEPM_FORCE] meanRelErr=%.3e maxRelErr=%.3e\n",
            sumRelErr / std::max(1, nCounted), maxRelErr);
    ::exit(0); // diagnostic report, not a strict pass/fail gate -- see comment above
  }
#endif
#if !defined(PERIODIC)
  // Phase 4 (PLAN.md) / LOG.md §34: the isolated-boundary counterpart to --pm-test-treepm-force
  // above -- confirms the tree side's newly-added (this session) erfc short-range suppression for
  // non-periodic TreePM actually combines correctly with the isolated PM solver's long-range
  // contribution. Ground truth is plain Newtonian G*m/r^2 -- an isolated system has no periodic
  // images to sum, simpler than the periodic case's Ewald reference. Same 2-particle
  // (target mass=0, source mass=1) trick as the periodic test, to isolate PM's field from its own
  // self-force. Reuses the program's own already-configured --meshsize/PMGRID
  // (pm_isolated_setup_gravity_kernel was already called with these above), not separate
  // test-specific values.
  if (pmTestIsolatedTreePMSamples > 0)
  {
    const int nSamples   = pmTestIsolatedTreePMSamples;
    const int gridSize   = PMGRID;
    const float cellSize = meshSize / (2.0f * gridSize); // physical cell size of the doubled grid

    // Log-spaced from a few cells (grid-resolution-limited near field) to a large fraction of the
    // occupied half-region (meshSize/2), same range convention as --pm-test-isolated-force.
    const double rMin = 3.0 * cellSize;
    const double rMax = 0.4 * (meshSize / 2.0);

    pm_fft_init();
    PMIsolatedSolver solver = pm_solver_create_isolated(gridSize, meshSize);

    fprintf(stderr, "[PM_TEST_ISOLATED_TREEPM_FORCE] meshSize=%g PMGRID=%d cellSize=%g nSamples=%d\n",
            meshSize, gridSize, cellSize, nSamples);
    fprintf(stderr, "[PM_TEST_ISOLATED_TREEPM_FORCE] %12s %14s %14s %14s %14s %10s\n",
            "sep", "treeFx", "pmFx", "totalFx", "newtonFx", "relErr");

    double sumRelErr = 0, maxRelErr = 0;
    int nCounted = 0;

    for (int s = 0; s < nSamples; s++)
    {
      const double logMin = log(rMin), logMax = log(rMax);
      const double sep = exp(logMin + (logMax - logMin) * s / (double)std::max(1, nSamples - 1));

      // --- Tree short-range force: exact add_acc() call, no tree build. ---
      float3 h_sep  = make_float3((float)sep, 0.0f, 0.0f);
      float  h_mass = 1.0f;
      const int h_type = 0; // type=0 (Gas): irrelevant here, this test never enables the zoom mask
      float3 *d_sep = nullptr; float *d_srcMass = nullptr; int *d_type = nullptr; float4 *d_treeAcc = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_sep, sizeof(float3)));
      CU_SAFE_CALL(hipMalloc((void**)&d_srcMass, sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_type, sizeof(int)));
      CU_SAFE_CALL(hipMalloc((void**)&d_treeAcc, sizeof(float4)));
      CU_SAFE_CALL(hipMemcpy(d_sep, &h_sep, sizeof(float3), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_srcMass, &h_mass, sizeof(float), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_type, &h_type, sizeof(int), hipMemcpyHostToDevice));
      pm_test_add_acc_pair(d_sep, d_srcMass, d_type, 1, d_treeAcc, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());
      float4 h_treeAcc;
      CU_SAFE_CALL(hipMemcpy(&h_treeAcc, d_treeAcc, sizeof(float4), hipMemcpyDeviceToHost));
      hipFree(d_sep); hipFree(d_srcMass); hipFree(d_type); hipFree(d_treeAcc);

      // --- PM long-range force: 2-particle system, target mass=0 (see comment above), both
      // placed in the occupied first half [0, meshSize/2), matching pm.h's PMIsolatedSolver
      // convention. ---
      const float base = meshSize * 0.25f; // center of the occupied half
      float4 h_pos[2];
      h_pos[0] = make_float4(base, base, base, 0.0f);
      h_pos[1] = make_float4(base + (float)sep, base, base, 1.0f);
      float4 *d_pos = nullptr; float *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_pos, 2*sizeof(float4)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fx, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fy, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fz, 2*sizeof(float)));
      CU_SAFE_CALL(hipMemcpy(d_pos, h_pos, 2*sizeof(float4), hipMemcpyHostToDevice));
      pm_compute_forces_isolated(solver, d_pos, 2, d_fx, d_fy, d_fz, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());
      float h_fx0 = 0.0f;
      CU_SAFE_CALL(hipMemcpy(&h_fx0, d_fx, sizeof(float), hipMemcpyDeviceToHost));
      hipFree(d_pos); hipFree(d_fx); hipFree(d_fy); hipFree(d_fz);

      const double totalFx = (double)h_treeAcc.x + (double)h_fx0;
      const double newtonFx = 1.0 / (sep * sep); // G=1, mass=1

      const double relErr = fabs(totalFx - newtonFx) / newtonFx;
      sumRelErr += relErr;
      maxRelErr = std::max(maxRelErr, relErr);
      nCounted++;

      fprintf(stderr, "[PM_TEST_ISOLATED_TREEPM_FORCE] %12.6g %14.6g %14.6g %14.6g %14.6g %10.3e\n",
              sep, (double)h_treeAcc.x, h_fx0, totalFx, newtonFx, relErr);
    }

    pm_solver_destroy_isolated(solver);
    fprintf(stderr, "[PM_TEST_ISOLATED_TREEPM_FORCE] meanRelErr=%.3e maxRelErr=%.3e\n",
            sumRelErr / std::max(1, nCounted), maxRelErr);
    ::exit(0); // diagnostic report, not a strict pass/fail gate -- see comment above
  }
#ifdef GADGET_HIP_HIGHRES
  // Phase 5 ticket 07 (PLAN.md): the zoom-aware counterpart to --pm-test-isolated-treepm-force
  // above -- ground truth is again plain Newtonian G*m/r^2, but the TARGET is given a high-res
  // Gadget-2 type (2) and a REAL fine/zoom grid (built directly, not via the particle-driven
  // pm_zoom_compute_region() -- this is a synthetic 2-particle test, there's no particle
  // distribution to derive a region from) is layered on top of the same coarse solver used above.
  // Confirms tree(Rcut[1]/Asmth[1]) + coarse-PM + fine-PM sum correctly -- i.e. the short/long
  // handoff works cleanly at Rcut[1] specifically (a genuinely different, finer cutoff than
  // Rcut[0]), not just re-testing Rcut[0]'s already-covered handoff from the test above.
  if (zoomTreePMTestSamples > 0)
  {
    const int nSamples = zoomTreePMTestSamples;
    const int gridSize = PMGRID;
    const float base   = meshSize * 0.25f; // SAME center the coarse test above uses

    // A genuinely "zoomed in" fine grid: a fraction of the coarse grid's own meshSize, not a copy
    // of it (ticket 06's own end-to-end test, LOG.md §41, found and fixed a bug where an earlier
    // formula could never produce a fine grid smaller than the coarse one).
    //
    // regionFrac=0.9 (a modest zoom, not an aggressive one) is a deliberate choice, not an
    // arbitrary one -- investigated directly (LOG.md, Ticket 07) via a parameter sweep from 0.6 to
    // 0.99: meanRelErr against exact Newtonian falls monotonically from ~18% (regionFrac=0.6) to
    // ~1.8% (0.99, matching the single-grid --pm-test-isolated-treepm-force baseline's own
    // ~1.6%/5.2% exactly). This is a REAL, understood, and BOUNDED numerical characteristic of
    // combining two INDEPENDENTLY CIC/FFT-discretized grids at different physical scales, not a
    // code defect: confirmed by hand-deriving the exact continuum formula
    // (tree_frac(u1)+coarse_frac(u0)+fine_frac(u0,u1) == 1 identically, verified numerically to
    // machine precision for every sample here) and finding it holds exactly, while the GPU's own
    // CIC/finite-difference realization of pmCoarseFx and pmFineFx each carry their own
    // near-field/mid-range discretization bias tied to their OWN grid's resolution. In the
    // single-grid case (the baseline test above), tree and PM share the SAME Asmth, so tree's
    // exact real-space contribution happens to compensate PM's grid bias in the region where PM is
    // least accurate; here, tree instead hands off at the FINER Asmth[1] scale while pmCoarseFx
    // still carries Asmth[0]-grid-resolution bias, so that same compensation is imperfect
    // specifically in the gap between Rcut[1] and Rcut[0] -- exactly where the fine grid's own
    // differential correction must supply the difference, each numerically imperfect grid's bias no
    // longer canceling as neatly as the single-grid case's does. A more aggressive zoom (smaller
    // regionFrac) makes this gap wider (relative to either grid's own cell size), and the residual
    // error correspondingly larger -- an aggressive zoom-ratio in a real particle-driven zoom sim
    // (Ticket 06's own pm_zoom_compute_region()) inherits this same characteristic, worth
    // remembering when interpreting any future zoom-sim energy/force diagnostics.
    const float  regionFrac = 0.9f;
    const float  regionMeshSize = meshSize * regionFrac;
    const double asmth0 = 1.25 * meshSize       / (2.0 * gridSize);
    const double asmth1 = 1.25 * regionMeshSize / (2.0 * gridSize);
    const double rcut1  = 4.5 * asmth1;
    const float  asmthRatio = (float)(asmth1 / asmth0);

    ZoomRegion region;
    region.corner[0] = region.corner[1] = region.corner[2] = base - 0.5f * regionMeshSize;
    region.totalMeshSize = regionMeshSize;
    region.meshSize       = 2.0f * regionMeshSize;
    region.asmth1         = (float) asmth1;
    region.rcut1          = (float) rcut1;
    region.valid          = true;

    const unsigned int zoomMask = (1u << 2); // type 2 (Disk) marked high-res for this test
    const int targetType = 2;

    // rMin must respect the COARSER of the two grids' own actual physical cell sizes (real cell
    // size = grid's own meshSize / fftGridSize; solver0's meshSize=meshSize, fftGridSize=2*gridSize
    // -- coarseCellSize = meshSize/(2*gridSize); solver1's meshSize=region.meshSize=2*regionMeshSize,
    // fftGridSize=2*gridSize too -- fineCellSize = 2*regionMeshSize/(2*gridSize) =
    // regionMeshSize/gridSize, NOT regionMeshSize/(2*gridSize) -- an earlier version of this test
    // computed fineCellSize as half its true value, silently under-flooring rMin whenever the true
    // fine cell size was the larger (binding) constraint, i.e. whenever regionFrac > 0.5). Using
    // the fine grid's own (smaller-at-low-ratio) cell size alone would let rMin fall below the
    // coarse grid's valid CIC resolution, aliasing pmCoarseFx (found via this test's own first run
    // before the invCellSize fix below: relErr ~50-80%). rMax stays a fraction of the fine grid's
    // own occupied half-region, same convention as the coarse test above.
    const float  coarseCellSize = meshSize       / (2.0f * gridSize);
    const float  fineCellSize   = regionMeshSize / (1.0f * gridSize);
    const double rMin = 3.0 * std::max(coarseCellSize, fineCellSize);
    const double rMax = 0.4 * (regionMeshSize / 2.0);

    pm_fft_init();
    PMIsolatedSolver solver0 = pm_solver_create_isolated(gridSize, meshSize);
    PMIsolatedSolver solver1 = pm_solver_create_finegrid(gridSize, region.meshSize, asmthRatio);
    pm_zoom_upload_rcut_asmth((float)rcut1, (float)asmth1, zoomMask);

    fprintf(stderr, "[ZOOM_TREEPM_FORCE_TEST] meshSize=%g regionMeshSize=%g PMGRID=%d "
                     "asmth0=%.6g asmth1=%.6g asmthRatio=%.6g nSamples=%d\n",
            meshSize, regionMeshSize, gridSize, asmth0, asmth1, asmthRatio, nSamples);
    fprintf(stderr, "[ZOOM_TREEPM_FORCE_TEST] %12s %14s %14s %14s %14s %14s %10s\n",
            "sep", "treeFx", "pmCoarseFx", "pmFineFx", "totalFx", "newtonFx", "relErr");

    double sumRelErr = 0, maxRelErr = 0;
    int nCounted = 0;

    for (int s = 0; s < nSamples; s++)
    {
      const double logMin = log(rMin), logMax = log(rMax);
      const double sep = exp(logMin + (logMax - logMin) * s / (double)std::max(1, nSamples - 1));

      // --- Tree short-range force: exact add_acc() call, target type=2 (high-res). ---
      float3 h_sep  = make_float3((float)sep, 0.0f, 0.0f);
      float  h_mass = 1.0f;
      float3 *d_sep = nullptr; float *d_srcMass = nullptr; int *d_type = nullptr; float4 *d_treeAcc = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_sep, sizeof(float3)));
      CU_SAFE_CALL(hipMalloc((void**)&d_srcMass, sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_type, sizeof(int)));
      CU_SAFE_CALL(hipMalloc((void**)&d_treeAcc, sizeof(float4)));
      CU_SAFE_CALL(hipMemcpy(d_sep, &h_sep, sizeof(float3), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_srcMass, &h_mass, sizeof(float), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_type, &targetType, sizeof(int), hipMemcpyHostToDevice));
      pm_test_add_acc_pair(d_sep, d_srcMass, d_type, 1, d_treeAcc, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());
      float4 h_treeAcc;
      CU_SAFE_CALL(hipMemcpy(&h_treeAcc, d_treeAcc, sizeof(float4), hipMemcpyDeviceToHost));
      hipFree(d_sep); hipFree(d_srcMass); hipFree(d_type); hipFree(d_treeAcc);

      // --- Coarse + fine PM: SAME 2-particle positions fed to both solvers (target mass=0,
      // source mass=1), both already inside the fine grid's own [corner, corner+totalMeshSize)
      // box by construction (base +/- rMax stays well inside it, rMax < 0.5*regionMeshSize). ---
      float4 h_pos[2];
      h_pos[0] = make_float4(base, base, base, 0.0f);
      h_pos[1] = make_float4(base + (float)sep, base, base, 1.0f);
      float4 *d_pos = nullptr;
      float *d_fx0 = nullptr, *d_fy0 = nullptr, *d_fz0 = nullptr;
      float *d_fx1 = nullptr, *d_fy1 = nullptr, *d_fz1 = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_pos, 2*sizeof(float4)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fx0, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fy0, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fz0, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fx1, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fy1, 2*sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_fz1, 2*sizeof(float)));
      CU_SAFE_CALL(hipMemcpy(d_pos, h_pos, 2*sizeof(float4), hipMemcpyHostToDevice));
      pm_compute_forces_isolated(solver0, d_pos, 2, d_fx0, d_fy0, d_fz0, 0);
      pm_compute_forces_finegrid(solver1, region, d_pos, 2, d_fx1, d_fy1, d_fz1, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());
      float h_fxCoarse = 0.0f, h_fxFine = 0.0f;
      CU_SAFE_CALL(hipMemcpy(&h_fxCoarse, d_fx0, sizeof(float), hipMemcpyDeviceToHost));
      CU_SAFE_CALL(hipMemcpy(&h_fxFine, d_fx1, sizeof(float), hipMemcpyDeviceToHost));
      hipFree(d_pos);
      hipFree(d_fx0); hipFree(d_fy0); hipFree(d_fz0);
      hipFree(d_fx1); hipFree(d_fy1); hipFree(d_fz1);

      const double totalFx  = (double)h_treeAcc.x + (double)h_fxCoarse + (double)h_fxFine;
      const double newtonFx = 1.0 / (sep * sep); // G=1, mass=1

      const double relErr = fabs(totalFx - newtonFx) / newtonFx;
      sumRelErr += relErr;
      maxRelErr = std::max(maxRelErr, relErr);
      nCounted++;

      fprintf(stderr, "[ZOOM_TREEPM_FORCE_TEST] %12.6g %14.6g %14.6g %14.6g %14.6g %14.6g %10.3e\n",
              sep, (double)h_treeAcc.x, h_fxCoarse, h_fxFine, totalFx, newtonFx, relErr);
    }

    // Ticket 07's own explicit acceptance criterion: "a low-res-typed particle's tree walk is
    // unaffected (still uses Rcut[0]/Asmth[0]) even when physically located inside the high-res
    // region." zoomMask above only marks type 2; probe the SAME separations with a type-1 (NOT in
    // the mask) target on this SAME build/run (g_pm_zoom_mask is already uploaded and nonzero,
    // proving this isn't just "mask happens to be 0") and confirm its tree force matches the
    // asmth0-based shortrange table exactly -- i.e. `add_acc()`'s `(1u<<type) & g_pm_zoom_mask`
    // test genuinely differentiates by type, not just by whether zoom is engaged at all.
    fprintf(stderr, "[ZOOM_TREEPM_FORCE_TEST] --- unmasked-type (type=1) probe: same build, same "
                     "uploaded Rcut[1]/Asmth[1]/mask, target NOT in the mask ---\n");
    bool unmaskedOk = true;
    for (int s = 0; s < 3; s++)
    {
      // A few near-field separations, kept well inside the shortrange table's own valid range
      // (u0 = 0.5*sep/asmth0 < 3, i.e. sep < 6*asmth0) so "want" itself isn't near the table's own
      // zero-cutoff floor, where a tiny absolute (correctly near-zero) difference would blow up a
      // relative-error check for no real reason.
      const double sep = rMin * (1.0 + 0.15 * s);

      float3 h_sep = make_float3((float)sep, 0.0f, 0.0f);
      float  h_mass = 1.0f;
      const int unmaskedType = 1; // NOT in zoomMask=(1u<<2)
      float3 *d_sep = nullptr; float *d_srcMass = nullptr; int *d_type = nullptr; float4 *d_acc = nullptr;
      CU_SAFE_CALL(hipMalloc((void**)&d_sep, sizeof(float3)));
      CU_SAFE_CALL(hipMalloc((void**)&d_srcMass, sizeof(float)));
      CU_SAFE_CALL(hipMalloc((void**)&d_type, sizeof(int)));
      CU_SAFE_CALL(hipMalloc((void**)&d_acc, sizeof(float4)));
      CU_SAFE_CALL(hipMemcpy(d_sep, &h_sep, sizeof(float3), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_srcMass, &h_mass, sizeof(float), hipMemcpyHostToDevice));
      CU_SAFE_CALL(hipMemcpy(d_type, &unmaskedType, sizeof(int), hipMemcpyHostToDevice));
      pm_test_add_acc_pair(d_sep, d_srcMass, d_type, 1, d_acc, 0);
      CU_SAFE_CALL(hipDeviceSynchronize());
      float4 h_acc;
      CU_SAFE_CALL(hipMemcpy(&h_acc, d_acc, sizeof(float4), hipMemcpyDeviceToHost));
      hipFree(d_sep); hipFree(d_srcMass); hipFree(d_type); hipFree(d_acc);

      // Independent host reference: Gadget-2's own shortrange FORCE table formula
      // (erfc(u)+2u/sqrt(pi)*exp(-u^2)) evaluated at asmth0 -- NOT asmth1 -- since type 1 must
      // never select the fine grid's pair.
      const double u0 = 0.5 * sep / asmth0;
      const double wantFrac = erfc(u0) + 2.0 * u0 / sqrt(M_PI) * exp(-u0 * u0);
      const double wantFx = (1.0 / (sep * sep)) * wantFrac;
      const double gotFx = (double) h_acc.x;
      const double relErr = fabs(gotFx - wantFx) / fabs(wantFx);
      // GADGET_HIP_NTAB=1000 discretizes u in [0,3) into 1000 bins (see
      // pm_upload_shortrange_tables's own doc comment) -- ~0.1-1% relative granularity is the
      // table's own expected/accepted precision floor, not a bug (same tolerance convention
      // established by earlier tickets' own device-vs-host table lookup tests).
      const bool ok = relErr < 1e-2;
      unmaskedOk = unmaskedOk && ok;
      fprintf(stderr, "[ZOOM_TREEPM_FORCE_TEST]   sep=%.6g type=1 treeFx=%.6g want(asmth0)=%.6g "
                       "relErr=%.3e %s\n", sep, gotFx, wantFx, relErr, ok ? "PASS" : "FAIL");
    }
    fprintf(stderr, "[ZOOM_TREEPM_FORCE_TEST] unmasked-type probe: %s\n",
            unmaskedOk ? "PASS (type 1 correctly used Asmth[0]/Rcut[0], unaffected by the zoom mask)"
                       : "FAIL");

    pm_solver_destroy_isolated(solver0);
    pm_solver_destroy_isolated(solver1);
    fprintf(stderr, "[ZOOM_TREEPM_FORCE_TEST] meanRelErr=%.3e maxRelErr=%.3e\n",
            sumRelErr / std::max(1, nCounted), maxRelErr);
    ::exit(0); // diagnostic report, not a strict pass/fail gate -- see comment above
  }
#endif
#endif
#endif

  #ifdef USE_MPI
    omp_set_num_threads(4); //Startup the OMP threads to be used during LET phase
  #endif


  //Start the integration
#ifdef USE_OPENGL
  octree::IterationData idata;
  initAppRenderer(argc, argv, tree, idata, displayFPS, stereo);
  LOG("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);
#else
  tree->mpiSync();
  if (procId==0) fprintf(stderr, " Start iterating\n");


  bool simulationFinished = false;
  ioSharedData.writingFinished       = true;

  //Uncomment this to have a monitor thread that can kill the 
  //program when no activity is detected
  //std::thread watcher(watchThread, tree);
  //watcher.detach();



  /* w/o MPI-IO use async fwrite, so use 2 threads otherwise, use 1 threads
   */
#pragma omp parallel num_threads(1+ (!useMPIIO))
  {
    const int tid = omp_get_thread_num();
    if (tid == 0)
    {
      //Catch exceptions to add some extra print info
      try
      {
        tree->iterate();
      }
      catch(const std::exception &exc)
      {
        std::cerr << "Process: "  << procId << "\t" << exc.what() <<std::endl;
        if(nProcs > 1) ::abort();
      }
      catch(...)
      {
        std::cerr << "Unknown exception on process: " << procId << std::endl;
        if(nProcs > 1) ::abort();
      }
      simulationFinished = true;
    }
    else
    {
      assert(!useMPIIO);
      /* IO */
      sleep(1);
      while(!simulationFinished)
      {
        if(ioSharedData.writingFinished == false)
        {
          const float t_current = ioSharedData.t_current;

          bool distributed = true;
          string fileName; fileName.resize(256);
          sprintf(&fileName[0], "%s_%010.4f-%d", snapshotFile.c_str(), t_current, procId);


          if(nProcs <= 16)
          {
//              distributed = false;
//              sprintf(&fileName[0], "%s_%010.4f", snapshotFile.c_str(), t_current);
          }

          tree->fileIO->writeFile(ioSharedData.Pos, ioSharedData.Vel,
                                  ioSharedData.IDs, ioSharedData.nBodies,
                                  fileName.c_str(), t_current,
                                  procId, nProcs, mpiCommWorld, distributed) ;

          ioSharedData.free();
          assert(ioSharedData.writingFinished == false);
          ioSharedData.writingFinished = true;
        }
        else
        {
          usleep(100);
        }
      }
    }
  }

  if (useMPIIO) tree->terminateIO();

  LOG("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);


  std::stringstream sstemp;
  sstemp << "Finished total took: " << tree->get_time()-t0 << std::endl;
  std::string stemp = sstemp.str();
  tree->writeLogData(stemp);
  tree->writeLogToFile();//Final write in case anything is left in the buffers

  if(tree->procId == 0)
  {
    LOGF(stderr, "TOTAL:   Time spent between the start of 'iterate' and the final time-step (very first step is not accounted)\n");
    LOGF(stderr, "Grav:    Time spent to compute gravity, including communication (wall-clock time)\n");
    LOGF(stderr, "GPUgrav: Time spent ON the GPU to compute local and LET gravity\n");
    LOGF(stderr, "LET Com: Time spent in exchanging and building LET data\n");
    LOGF(stderr, "Build:   Time spent in constructing the tree (incl sorting, making groups, etc.)\n");
    LOGF(stderr, "Domain:  Time spent in computing new domain decomposition and exchanging particles between nodes.\n");
    LOGF(stderr, "Wait:    Time spent in waiting on other processes after the gravity part.\n");
  }


  delete tree;
  tree = NULL;

#endif

  displayTimers();

#ifdef USE_MPI
  //Finalize MPI if we initialized it ourselves, otherwise the driver will do it.
  if (!mpiInitialized) MPI_Finalize();
#endif
  return 0;
}

int main(int argc, char *argv[])
{
#ifdef USE_MPI
  return bonsai_main(argc, argv, MPI_COMM_WORLD, 0);
#else
  // Matches the !USE_MPI branch inside bonsai_main, which already sets mpiCommWorld = 0.
  return bonsai_main(argc, argv, 0, 0);
#endif
}
