#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <GNNaggregation.cuh>

#include <nvToolsExt.h>


// Check result on the CPU
void verify_result(
    std::vector<int> &a, 
    std::vector<int> &b, 
    std::vector<int> &vector_i,
    std::vector<int> &c, 
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
  constexpr int N = 1 << 10;  //N=2^10=1024

  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);
  size_t vector_size = N * sizeof(int);

  // Host vectors 
  nvtxRangePush("allocate host memory for three matrices and one vector");
  std::vector<int> h_a(N * N);
  std::vector<int> h_b(N * N);
  std::vector<int> h_c(N * N);
  std::vector<int> h_i(N);
  nvtxRangePop();

  // Initialize matrices
  nvtxRangePush("initialize two source matrices and one vetcor with random numbers");
  std::generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  std::generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
  std::generate(h_i.begin(), h_i.end(), []() { return rand() % 100; });
  nvtxRangePop();

  // Allocate device memory
  nvtxRangePush("allocate device memory for three matrices and one vector");
  int *d_a, *d_b, *d_i, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMalloc(&d_i, vector_size);
  nvtxRangePop();

  // Copy data to the device
  nvtxRangePush("copy matrices/vector from host to device memory");
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
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
  nvtxRangePush("verify result");
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
