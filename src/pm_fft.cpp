#include <cstdio>
#include <cstdlib>
#include <rocfft/rocfft.h>
#include "pm.h"

static void pm_fft_check(rocfft_status status, const char *what)
{
  if (status != rocfft_status_success)
  {
    fprintf(stderr, "rocFFT error in %s: status %d\n", what, (int)status);
    exit(EXIT_FAILURE);
  }
}

void pm_fft_init()
{
  pm_fft_check(rocfft_setup(), "rocfft_setup");
}

void pm_fft_shutdown()
{
  pm_fft_check(rocfft_cleanup(), "rocfft_cleanup");
}

// Builds one direction (forward or inverse) of the plan pair. Both directions share the same
// explicit strides, computed from gridSize/gridPitch rather than trusting rocFFT's unspecified
// default layout for an in-place real<->hermitian transform -- see pm.h's PMFFTPlans comment and
// LOG.md for why this was made explicit rather than assumed.
static rocfft_plan pm_fft_create_one_plan(int gridSize, int gridPitch, bool forward, double scaleFactor)
{
  const size_t lengths[3] = { (size_t)gridSize, (size_t)gridSize, (size_t)gridSize };
  // lengths[0] is rocFFT's "innermost" (fastest, contiguous) dimension -- for our cubic grid the
  // three values are identical so this doesn't affect the lengths array itself, but it does mean
  // the strides below must be given in the same fastest-to-slowest order.

  const size_t complexPitch = gridPitch / 2; // gridPitch real floats == complexPitch float2s

  // Real-side strides (element = 1 float), matching pm_cic_assign_mass's own indexing
  // ((x*gridSize + y)*gridPitch + z): z (fastest) stride 1, y stride gridPitch, x stride
  // gridSize*gridPitch.
  const size_t realStrides[3]    = { 1, (size_t)gridPitch, (size_t)gridSize * gridPitch };
  // Complex-side strides (element = 1 float2), same physical layout reinterpreted as complex:
  // z (fastest) stride 1, y stride complexPitch, x stride gridSize*complexPitch.
  const size_t complexStrides[3] = { 1, complexPitch, (size_t)gridSize * complexPitch };

  const size_t realDistance    = (size_t)gridSize * gridSize * gridPitch;
  const size_t complexDistance = (size_t)gridSize * gridSize * complexPitch;

  rocfft_plan_description desc;
  pm_fft_check(rocfft_plan_description_create(&desc), "rocfft_plan_description_create");

  if (forward)
    pm_fft_check(rocfft_plan_description_set_data_layout(
                     desc, rocfft_array_type_real, rocfft_array_type_hermitian_interleaved,
                     nullptr, nullptr, 3, realStrides, realDistance, 3, complexStrides, complexDistance),
                 "rocfft_plan_description_set_data_layout (forward)");
  else
    pm_fft_check(rocfft_plan_description_set_data_layout(
                     desc, rocfft_array_type_hermitian_interleaved, rocfft_array_type_real,
                     nullptr, nullptr, 3, complexStrides, complexDistance, 3, realStrides, realDistance),
                 "rocfft_plan_description_set_data_layout (inverse)");

  pm_fft_check(rocfft_plan_description_set_scale_factor(desc, scaleFactor),
               "rocfft_plan_description_set_scale_factor");

  rocfft_plan plan;
  pm_fft_check(rocfft_plan_create(&plan, rocfft_placement_inplace,
                                   forward ? rocfft_transform_type_real_forward
                                           : rocfft_transform_type_real_inverse,
                                   rocfft_precision_single, 3, lengths, 1, desc),
               "rocfft_plan_create");

  rocfft_plan_description_destroy(desc); // safe to free once passed to rocfft_plan_create
  return plan;
}

PMFFTPlans pm_fft_create_plans(int gridSize)
{
  const int gridPitch = 2 * (gridSize / 2 + 1);

  PMFFTPlans plans;
  plans.gridSize = gridSize;
  plans.forwardPlan = pm_fft_create_one_plan(gridSize, gridPitch, /*forward=*/true, /*scale=*/1.0);
  plans.inversePlan = pm_fft_create_one_plan(gridSize, gridPitch, /*forward=*/false,
                                              /*scale=*/1.0 / ((double)gridSize * gridSize * gridSize));

  rocfft_execution_info info;
  pm_fft_check(rocfft_execution_info_create(&info), "rocfft_execution_info_create");
  plans.execInfo = info;

  return plans;
}

void pm_fft_destroy_plans(PMFFTPlans &plans)
{
  if (plans.execInfo)
    rocfft_execution_info_destroy((rocfft_execution_info)plans.execInfo);
  if (plans.forwardPlan)
    rocfft_plan_destroy((rocfft_plan)plans.forwardPlan);
  if (plans.inversePlan)
    rocfft_plan_destroy((rocfft_plan)plans.inversePlan);
  plans.execInfo = plans.forwardPlan = plans.inversePlan = nullptr;
}

static void pm_fft_execute(rocfft_plan plan, rocfft_execution_info info, float *d_grid, hipStream_t stream)
{
  pm_fft_check(rocfft_execution_info_set_stream(info, (void*)stream),
               "rocfft_execution_info_set_stream");
  void *buf[1] = { (void*)d_grid };
  pm_fft_check(rocfft_execute(plan, buf, nullptr, info), "rocfft_execute");
}

void pm_fft_forward(PMFFTPlans &plans, float *d_grid, hipStream_t stream)
{
  pm_fft_execute((rocfft_plan)plans.forwardPlan, (rocfft_execution_info)plans.execInfo, d_grid, stream);
}

void pm_fft_inverse(PMFFTPlans &plans, float *d_grid, hipStream_t stream)
{
  pm_fft_execute((rocfft_plan)plans.inversePlan, (rocfft_execution_info)plans.execInfo, d_grid, stream);
}
