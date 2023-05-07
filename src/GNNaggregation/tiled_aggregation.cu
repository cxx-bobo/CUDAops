#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <GNNaggregation.cuh>

#include <nvToolsExt.h>
#include <cstdlib>


// Check result on the CPU
void verify_result(
  std::vector<int> &matrix_W, 
  std::vector<int> &matrix_H, 
  std::vector<int> &vector_b,
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
        tmp += matrix_W[i * N + k] * matrix_H[k * N + j];
      }
      tmp += vector_b[i];
      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {

  // Matrix size 
  constexpr int N = 3; //N=32*k（k=1,2,3...）
  std::cout << "N=" << N << std::endl;

  // Size (in bytes) of matrix
  size_t matrix_size = N * N * sizeof(int);
  size_t vector_size = N * sizeof(int);

  // Host vectors
  nvtxRangePush("allocate host memory for three matrices and one vector");
  std::vector<int> h_matrix_W(N * N);
  std::vector<int> h_matrix_H(N * N);
  std::vector<int> h_matrix_HK(N * N);
  std::vector<int> h_vector_b(N);
  nvtxRangePop();

  // Initialize matrices
  nvtxRangePush("initialize two source matrices and one vectorwith random numbers");
  std::generate(h_matrix_W.begin(), h_matrix_W.end(), []() { return rand() % 100; });
  std::generate(h_matrix_H.begin(), h_matrix_H.end(), []() { return rand() % 100; });
  std::generate(h_vector_b.begin(), h_vector_b.end(), []() { return rand() % 100; });
  // std::cout << "vector_b = " << h_vector_b;
  nvtxRangePop();

  // Allocate device memory
  nvtxRangePush("allocate device memory for three matrices and one vector");
  int *d_matrix_W, *d_matrix_H, *d_vector_b, *d_matrix_HK;
  cudaMalloc(&d_matrix_W, matrix_size);
  cudaMalloc(&d_matrix_H, matrix_size);
  cudaMalloc(&d_matrix_HK, matrix_size);
  cudaMalloc(&d_vector_b, vector_size);
  nvtxRangePop();

  // Copy data to the device
  nvtxRangePush("copy matrices and vector from host to device memory");
  cudaMemcpy(d_matrix_W, h_matrix_W.data(), matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrix_H, h_matrix_H.data(), matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_b, h_vector_b.data(), vector_size, cudaMemcpyHostToDevice);
  nvtxRangePop();

  // Threads per CTA dimension
  int THREADS_PER_BLOCK = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS_NUM = N % THREADS_PER_BLOCK  == 0 ? N / THREADS_PER_BLOCK  : N / THREADS_PER_BLOCK  + 1;

  // Use dim3 structs for block  and grid dimensions
  dim3 block(THREADS_PER_BLOCK , THREADS_PER_BLOCK);
  dim3 grip(BLOCKS_NUM, BLOCKS_NUM);

  // obtain shared memory size for each thread block(tile_A+tile_B,所以乘2)
  int shared_memory_size = 2*THREADS_PER_BLOCK *THREADS_PER_BLOCK *sizeof(int);

  // Launch kernel
  std::cout << block.x <<std::endl;
  std::cout << "Launch Kernel: " << THREADS_PER_BLOCK  << " threads per block(one dimension), " << BLOCKS_NUM << " blocks in the grid(one dimension)" << std::endl;
  nvtxRangePush("start kernel");
  tiledMatrixMul<<<grip, block, shared_memory_size>>>(d_matrix_W, d_matrix_H, d_vector_b, THREADS_PER_BLOCK, d_matrix_HK, N);
  nvtxRangePop();

  // Copy back to the host
  nvtxRangePush("copy matrix from device to host memory");
  cudaMemcpy(h_matrix_HK.data(), d_matrix_HK, matrix_size, cudaMemcpyDeviceToHost);
  nvtxRangePop();

  // Check result
  nvtxRangePush("verify result");
  verify_result(h_matrix_W, h_matrix_H, h_vector_b, h_matrix_HK, N);
  nvtxRangePop();

  std::cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  nvtxRangePush("free device memory");
  cudaFree(d_matrix_W);
  cudaFree(d_matrix_H);
  cudaFree(d_matrix_HK);
  cudaFree(d_vector_b);
  nvtxRangePop();

  return 0;
}
