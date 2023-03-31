// This program computes a simple version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include <nvToolsExt.h>


using std::generate;
using std::vector;

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

// Check result on the CPU
void verify_result(
    vector<int> &a, 
    vector<int> &b, 
    vector<int> &vector_i,
    vector<int> &c, 
    int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }
      tmp += vector_i[i];
      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
  // Matrix size of 1024 x 1024;
  int N = 1 << 10;  //N=2^10=1024

  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);
  size_t vector_size = N * sizeof(int);

  // Host vectors 
  nvtxRangePush("allocate host memory for three matrices and one vector");
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);
  nvtxRangePop();
  nvtxRangePush("allocate host memory for one vector");
  vector<int> h_i(N);
  nvtxRangePop();

  // Initialize matrices
  nvtxRangePush("initialize two source matrices with random numbers");
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
  nvtxRangePop();
  nvtxRangePush("initialize one vector with random numbers");
  generate(h_i.begin(), h_i.end(), []() { return rand() % 100; });
  nvtxRangePop();

  // Allocate device memory
  nvtxRangePush("allocate device memory for three matrices");
  int *d_a, *d_b, *d_i, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  nvtxRangePop();
  nvtxRangePush("allocate device memory for one vector");
  cudaMalloc(&d_i, vector_size);
  nvtxRangePop();

  // Copy data to the device
  nvtxRangePush("copy matrices from host to device memory");
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
  nvtxRangePop();
  nvtxRangePush("copy vector from host to device memory");
  cudaMemcpy(d_i, h_i.data(), vector_size, cudaMemcpyHostToDevice);
  nvtxRangePop();

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  std::cout << "Launch Kernel: " << THREADS << " threads per block, " << BLOCKS << " blocks in the grid" << std::endl;
  nvtxRangePush("start kernel");
  basicMatrixMul<<<blocks, threads>>>(d_a, d_b, d_i, d_c, N);
  nvtxRangePop();

  // Copy back to the host
  nvtxRangePush("copy matrices from device to host memory");
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
  nvtxRangePop();

  // Check result
  nvtxRangePush("copy matrix from device to host memory");
  verify_result(h_a, h_b, h_i, h_c, N);
  nvtxRangePop();

  // Free memory on device
  nvtxRangePush("free device memory");
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_i);
  cudaFree(d_c);
  nvtxRangePop();

  std::cout << "matrix COMPLETED SUCCESSFULLY\n";

  return 0;
}
