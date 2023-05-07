#ifndef _MATRIX_MUL_H_
#define _MATRIX_MUL_H_

#include<stdint.h>

__global__ void basicMatrixMul(
    const int *a, 
    const int *b, 
    const int *vector_i,
    int *c, 
    int N
);

__global__ void tiledMatrixMul(
    const int *matrix_W, 
    const int *matrix_H, 
    const int *vector_b,
    const int tile_size,
    int *c,
    const int N
);

#endif