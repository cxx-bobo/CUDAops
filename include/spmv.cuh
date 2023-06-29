#ifndef _SPMV_H_
#define _SPMV_H_
#define NNZ_PER_WG 256u  ///u表示无符号整数

#include<stdint.h>

__global__ void csr_spmv_scalar_kernel (
    const uint64_t n_rows,
    const uint64_t *col_ids,
    const uint64_t *row_ptr,
    const float *data,
    const float *x,
    float *y
);

__global__ void csr_spmv_vector_kernel (
  const uint64_t n_rows,
  const uint64_t *col_ids,
  const uint64_t *row_ptr,
  const float *data,
  const float *x,
  float *y
);

__global__ void csr_spmv_adaptive_kernel(
  const uint64_t n_rows,
  const uint64_t *col_dis,
  const uint64_t *row_ptr,
  const uint64_t *row_blocks,
  const float *data,
  const float *x,
  float *y
);

#endif