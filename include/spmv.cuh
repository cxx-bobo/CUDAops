#ifndef _SPMV_H_
#define _SPMV_H_

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
#endif