#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <spmv.cuh>

#include <nvToolsExt.h>

void generateSparseMatrix(
    const int numRows, 
    const int numCols, 
    double density,
    std::vector<float>& values, 
    std::vector<int>& colIndices, 
    std::vector<int>& rowOffsets
);

int main() {
  // initial constants
  constexpr int numRows = 10; 
  constexpr int sizeRow = 6;
  const double density = 0.3;
  
  size_t size_x = numRows * sizeof(float);
  size_t size_y = sizeRow * sizeof(float);
  
  // Host vectors 
  nvtxRangePush("allocate host memory for vectors");
  std::vector<float> values; 
  std::vector<int> col_idx; 
  std::vector<int> row_ptr;
  std::vector<float> vector_x(size_x);
  std::vector<float> result_y(size_y);
  nvtxRangePop();

  // Initialize csr and vector_x
  nvtxRangePush("initialize source csr with random numbers");
  generateSparseMatrix(numRows, sizeRow, density, values, col_idx, row_ptr);
  std::generate(vector_x.begin(), vector_x.end(), []() { return rand() % 100; });
  nvtxRangePop();

  // Allocate device memory
  nvtxRangePush("allocate device memory for three matrices and one vector");
  float *d_values, *d_x, *d_y;
  int *d_col_idx, *d_row_ptr;
  cudaMalloc(&d_values, sizeof(float)*values.size());
  cudaMalloc(&d_col_idx, sizeof(int)*col_idx.size());
  cudaMalloc(&d_row_ptr, sizeof(int)*row_ptr.size());
  cudaMalloc(&d_x, size_x);
  cudaMalloc(&d_y, size_y);
  nvtxRangePop();

  // Copy data to the device
  nvtxRangePush("copy data from host to device memory");
  cudaMemcpy(d_values, values.data(), sizeof(float)*values.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, col_idx.data(), sizeof(int)*col_idx.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(int)*row_ptr.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, vector_x.data(), size_x, cudaMemcpyHostToDevice);
  nvtxRangePop();

  // Threads per CTA dimension
  int threads_per_CTAdim = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int blocks_per_GRIDdim = ( numRows + threads_per_CTAdim -1 ) / threads_per_CTAdim;

  // Use dim3 structs for block  and grid dimensions
  dim3 BLOCK(threads_per_CTAdim, threads_per_CTAdim);
  dim3 GRID(blocks_per_GRIDdim, blocks_per_GRIDdim);

  // Launch kernel
  std::cout << "Launch Kernel: " << threads_per_CTAdim << " threads per block, " << blocks_per_GRIDdim << " blocks in the grid" << std::endl;
  nvtxRangePush("start kernel");
  csr_spmv_scalar_kernel<<<GRID, BLOCK>>>(numRows, d_col_idx, d_row_ptr, d_values, d_x, d_y);
  nvtxRangePop();

  // Copy back to the host
  nvtxRangePush("copy matrices from device to host memory");
  cudaMemcpy(result_y.data(), d_y, size_y, cudaMemcpyDeviceToHost);
  nvtxRangePop();

  // // Check result
  // nvtxRangePush("copy matrix from device to host memory");
  // verify_result(h_a, h_b, h_i, h_c, N);
  // nvtxRangePop();

  // Free memory on device
  nvtxRangePush("free device memory");
  cudaFree(d_values);
  cudaFree(d_col_idx);
  cudaFree(d_row_ptr);
  cudaFree(d_x);
  cudaFree(d_y);
  nvtxRangePop();

  std::cout << "matrix COMPLETED SUCCESSFULLY\n";

  return 0;
}
