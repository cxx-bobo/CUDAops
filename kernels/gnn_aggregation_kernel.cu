#include <GNNaggregation.cuh>
#include <cassert>
#include <iostream>

__global__ void basicMatrixMul(
    const int *a, 
    const int *b, 
    const int *vector_i,
    int *c, 
    int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * N + col] += a[row * N + k] * b[k * N + col];  
  }
  c[row * N + col] += vector_i[row];
}


__global__ void tiledMatrixMul(
    const int *matrix_W, 
    const int *matrix_H, 
    const int *vector_b,
    const int tile_size,
    int *c,
    const int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  extern __shared__ int tile[];
  int* tile_W = tile;
  int* tile_H = tile+tile_size*tile_size;
  int tmp = 0;
  printf("I'm in kernel");
  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    
    int index_W = row * N + i + threadIdx.x;
    if(index_W < N*N){
      // Load in elements for this tile
      tile_W[threadIdx.y * blockDim.x + threadIdx.x] = matrix_W[index_W];
    }else{
      tile_W[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    }

    int index_H = i * N + threadIdx.y * N + col;
    if(index_H < N*N){
      // Load in elements for this tile
      tile_H[threadIdx.y * blockDim.x + threadIdx.x] = matrix_H[index_H];
    }else{
      tile_H[threadIdx.y * blockDim.x + threadIdx.x] = 0;
    }
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp += tile_W[threadIdx.y * blockDim.x + j] * tile_H[j * blockDim.x + threadIdx.x];
    }
    __syncthreads();
    }

  // Write back results
  if(row < N && col < N){
    c[row * N + col] = vector_b[row] +tmp;
  }
}