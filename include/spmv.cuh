#ifndef _SPMV_H_
#define _SPMV_H_

#include<stdint.h>

__global__ void csr_spmv_scalar_kernel (
    const int n_rows,
    const int *col_ids,
    const int *row_ptr,const
    const float *data,
    const float *x,
    float *y
);

__global__ void csr_spmv_vector_kernel (
  unsigned int n_rows,
  const unsigned int *col_ids,
  const unsigned int *row_ptr,
  const float *data,
  const float *x,
  float *y
);
#endif