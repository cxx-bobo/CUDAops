#include <spmv.cuh>
#include <cassert>

#define FULL_WARP_MASK 0xffffffff

__global__ void csr_spmv_scalar_kernel (
    const uint64_t n_rows,
    const uint64_t *col_ids,
    const uint64_t *row_ptr,
    const float *data,
    const float *x,
    float *y)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows)
    {
        const uint64_t row_start = row_ptr[row];
        const uint64_t row_end = row_ptr[row + 1];
        float sum = 0;
        for (unsigned int element = row_start; element< row_end; element++){
            sum += data[element] * x[col_ids[element]];
        }
        y[row] = sum;
    }
}


__device__ float warp_reduce (float val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
     val += __shfl_down_sync (FULL_WARP_MASK,val,offset);
  return val;
}

__global__ void csr_spmv_vector_kernel (
  unsigned int n_rows,
  const unsigned int *col_ids,
  const unsigned int *row_ptr,
  const float *data,
  const float *x,
  float *y)
{
  const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int warp_id = thread_id / 32;
  const unsigned int lane = thread_id % 32;
  const unsigned int row = warp_id; ///< One warp per row
  float sum =0;
  if (row < n_rows)
 {
   const unsigned int row_start = row_ptr[row];
   const unsigned int row_end = row_ptr[row + 1];
   for (unsigned int element = row_start + lane; element < row_end; element += 32)
       sum += data[element] * x[col_ids[element]];
  }
 sum = warp_reduce (sum);
 if (lane == 0 && row < n_rows)
    y[row] = sum;
}