#include <spmv.cuh>
#include <cassert>
#include <iostream>

#define FULL_WARP_MASK 0xffffffff

//csr_spmv_scalar kernel
__global__ void csr_spmv_scalar_kernel (
    const uint64_t n_rows,
    const uint64_t *col_ids,
    const uint64_t *row_ptr,
    const float *data,
    const float *x,
    float *y)
{
    uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows)
    {
        const uint64_t data_start = row_ptr[row];
        const uint64_t data_end = row_ptr[row + 1];
        float sum = 0;
        for (uint64_t element = data_start; element< data_end; element++){
            sum += data[element] * x[col_ids[element]];
        }
        y[row] = sum;
    }
}



//使用warp-level primitives进行并行规约
__device__ float warp_reduce (float val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
     val += __shfl_down_sync (FULL_WARP_MASK,val,offset);
  return val;
}
//csr_spmv_vector kernel
__global__ void csr_spmv_vector_kernel (
  const uint64_t num_rows,
  const uint64_t *col_ids,
  const uint64_t *row_ptr,
  const float *data,
  const float *x,
  float *y)
{
  const uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t warp_id = thread_id / 32;
  const uint64_t lane = thread_id % 32; //当前thread在一个warp里的索引
  const uint64_t row = warp_id; /// One warp per row
  float sum =0;
  if (row < num_rows)
 {
   const uint64_t data_start = row_ptr[row];
   const uint64_t data_end = row_ptr[row + 1];
   for (uint64_t element = data_start + lane; element < data_end; element += 32)
       sum += data[element] * x[col_ids[element]];
  }
 sum = warp_reduce (sum);
 if (lane == 0 && row < num_rows)
    y[row] = sum;
}



//计算≤n的最大2的幂次
__device__ unsigned int prev_power_of_2 (unsigned int n)
{
  while (n & n - 1)
    n = n & n - 1;
  return n;
}
//csr_spmv_adaptive kernel
__global__ void csr_spmv_adaptive_kernel (
    const uint64_t n_rows,
    const uint64_t *col_ids,
    const uint64_t *row_ptr,
    const uint64_t *row_blocks,
    const float *data,
    const float *x,
    float *y)
{
  // printf("I'm in adaptive-kernel \n");
  const unsigned int block_row_begin = row_blocks[blockIdx.x];  //当前block中的起始行索引
  const unsigned int block_row_end = row_blocks[blockIdx.x + 1];  //当前block中的结束行索引
  const unsigned int nnz = row_ptr[block_row_end] - row_ptr[block_row_begin]; //当前block中的非零元个数

  //--声明共享内存数组cache，大小为NNZ_PER_WG
  __shared__ float cache[NNZ_PER_WG];

  //if：若一个block中存的非零元超过1行，则执行CSR_Stream
  if (block_row_end - block_row_begin > 1)
  {
    /// CSR-Stream case
    // printf("I'm in CSR-Stream case \n");
    const unsigned int block_data_begin = row_ptr[block_row_begin]; //当前block中非零元素的起始索引
    const unsigned int thread_data = block_data_begin + threadIdx.x;  //当前thread要处理的非零元索引

    
    // if (threadIdx.x < nnz)
    //每个thread处理一个非零元，并将结果保存在共享内存中
    cache[threadIdx.x] = data[thread_data] * x[col_ids[thread_data]];
    __syncthreads ();

    //计算用来进行规约的线程数量
    const unsigned int threads_for_reduction = prev_power_of_2 (blockDim.x / (block_row_end - block_row_begin));

    //对当前block进行并行规约处理
    //if:若进行规约的线程数量>1,则各行先进行一次粗规约(threads_for_reduction个线程)；
    //   然后再各行进行细规约(threads_for_reduction个线程)；
    //   实际threads_for_reduction = 该block中的行数
    if (threads_for_reduction > 1)
      {
        /// Reduce all non zeroes of row by multiple thread
        //计算当前thread在block中的索引
        const unsigned int thread_in_block = threadIdx.x % threads_for_reduction;
        //计算当前thread处理的当前行索引
        const unsigned int local_row = block_row_begin + threadIdx.x / threads_for_reduction;
        float dot = 0.0;

        if (local_row < block_row_end)
          {
            const unsigned int local_first_element = row_ptr[local_row] - row_ptr[block_row_begin];
            const unsigned int local_last_element = row_ptr[local_row + 1] - row_ptr[block_row_begin];

            for (unsigned int local_element = local_first_element + thread_in_block;
                 local_element < local_last_element;
                 local_element += threads_for_reduction)
              {
                dot += cache[local_element];
              }
          }
        __syncthreads ();
        cache[threadIdx.x] = dot;
        __syncthreads ();

        /// Now each row has threads_for_reduction values in cache
        for (int j = threads_for_reduction / 2; j > 0; j /= 2)
          {
            /// Reduce for each row
            //__syncthreads ();

            const bool use_result = thread_in_block < j && threadIdx.x + j < NNZ_PER_WG;

            if (use_result)
              dot += cache[threadIdx.x + j];
            __syncthreads ();

            if (use_result)
              cache[threadIdx.x] = dot;
          }

        if (thread_in_block == 0 && local_row < block_row_end)
          y[local_row] = dot;
      }
    //如果进行规约的线程数=1，用1个thread对1行进行规约。
    else
      {
        /// Reduce all non zeroes of row by single thread
        //计算线程处理的当前行索引
        unsigned int local_row = block_row_begin + threadIdx.x;
        while (local_row < block_row_end)
          {
            float dot = 0.0;

            for (unsigned int j = row_ptr[local_row] - block_data_begin;
                 j < row_ptr[local_row + 1] - block_data_begin;
                 j++)
              {
                dot += cache[j];
              }

            y[local_row] = dot;
            //一个block有NNZ_PER_WG个thread，故+NNZ_PER_WG
            local_row += NNZ_PER_WG;
          }
      }
  }
  else
  {
    const unsigned int row = block_row_begin;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;

    float dot = 0;

    if (nnz <= 64 || NNZ_PER_WG <= 32)
    {
      /// CSR-Vector case
      // printf("I'm in CSR-Vector case \n");
      if (row < n_rows)
      {
        const unsigned int row_start = row_ptr[row];
        const unsigned int row_end = row_ptr[row + 1];

        for (unsigned int element = row_start + lane; element < row_end; element += 32)
          dot += data[element] * x[col_ids[element]];
      }

      dot = warp_reduce (dot);

      if (lane == 0 && warp_id == 0 && row < n_rows)
      {
        y[row] = dot;
      }
    }
    else
    {
      /// CSR-VectorL case
      // printf("I'm in CSR-VectorL case \n");
      if (row < n_rows)
      {
        const unsigned int row_start = row_ptr[row];
        const unsigned int row_end = row_ptr[row + 1];

        for (unsigned int element = row_start + threadIdx.x; element < row_end; element += blockDim.x)
          dot += data[element] * x[col_ids[element]];
      }

      dot = warp_reduce (dot);

      if (lane == 0)
        cache[warp_id] = dot;
      __syncthreads ();

      if (warp_id == 0)
      {
        dot = 0.0;

        for (unsigned int element = lane; element < blockDim.x / 32; element += 32)
          dot += cache[element];

        dot = warp_reduce (dot);

        if (lane == 0 && row < n_rows)
        {
          y[row] = dot;
        }
      }
    }
  }
}

