#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <spmv.cuh>

#include <nvToolsExt.h>

void generateSparseMatrix(
    const uint64_t numRows, 
    const uint64_t numCols, 
    double density,
    std::vector<float> &values, 
    std::vector<uint64_t> &colIndices, 
    std::vector<uint64_t> &rowOffsets
);

void verifySpMVresult(
    const uint64_t numRows, 
    const uint64_t numCols, 
    std::vector<float> &values, 
    std::vector<uint64_t> &col_idx, 
    std::vector<uint64_t> &row_ptr,
    std::vector<float> &x,
    std::vector<float> &y
);

int main() {
  // initial constants
  constexpr uint64_t numRows = 1<<4; 
  constexpr uint64_t sizeRow = 1<<3;
  const double density = 0.1;
  
  size_t size_x = sizeRow * sizeof(float);
  size_t size_y = numRows * sizeof(float);
  
  // Host vectors 
  std::vector<float> values; 
  std::vector<uint64_t> col_idx; 
  std::vector<uint64_t> row_ptr;
  std::vector<float> h_x;
  h_x.reserve(sizeRow);
  std::vector<float> h_y;
  h_y.reserve(numRows);
  
  // Initialize csr and vector_x
  nvtxRangePush("initialize csr with random numbers");
  generateSparseMatrix(numRows, sizeRow, density, values, col_idx, row_ptr);
  for (int i=0; i<sizeRow; i++){
    h_x.push_back(static_cast<float>(rand() % 100));
  };
  nvtxRangePop();

  // Allocate device memory
  nvtxRangePush("allocate device memory");
  float *d_values, *d_x, *d_y;
  uint64_t *d_col_idx, *d_row_ptr;
  cudaMalloc(&d_values, sizeof(float)*values.size());
  cudaMalloc(&d_col_idx, sizeof(uint64_t)*col_idx.size());
  cudaMalloc(&d_row_ptr, sizeof(uint64_t)*row_ptr.size());
  cudaMalloc(&d_x, size_x);
  cudaMalloc(&d_y, size_y);
  nvtxRangePop();

  // Copy data to the device
  nvtxRangePush("copy data from host to device memory");
  cudaMemcpy(d_values, values.data(), sizeof(float)*values.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, col_idx.data(), sizeof(uint64_t)*col_idx.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(uint64_t)*row_ptr.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x.data(), size_x, cudaMemcpyHostToDevice);
  nvtxRangePop();

  // Threads per CTA dimension
  int threads_per_CTAdim = 128;

  // Blocks per grid dimension 
  int blocks_per_GRIDdim = ( numRows + threads_per_CTAdim -1 ) / threads_per_CTAdim;

  // Launch kernel
  std::cout << "Launch Kernel: " << threads_per_CTAdim << " threads per block, " << blocks_per_GRIDdim << " blocks in the grid" << std::endl;
  nvtxRangePush("Launch kernel");
  csr_spmv_scalar_kernel<<<blocks_per_GRIDdim, threads_per_CTAdim>>>(numRows, d_col_idx, d_row_ptr, d_values, d_x, d_y);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
  } 
  nvtxRangePop();

  // Copy back to the host
  nvtxRangePush("copy data from device to host memory");
  cudaMemcpy(h_y.data(), d_y, size_y, cudaMemcpyDeviceToHost);
  nvtxRangePop();

  // Check result
  nvtxRangePush("veryfy result");
  verifySpMVresult(numRows, sizeRow, values, col_idx, row_ptr, h_x, h_y);
  nvtxRangePop();

  // Free memory on device
  nvtxRangePush("free device memory");
  cudaFree(d_values);
  cudaFree(d_col_idx);
  cudaFree(d_row_ptr);
  cudaFree(d_x);
  cudaFree(d_y);
  nvtxRangePop();

  std::cout << "csr_spmv_scalar COMPLETED SUCCESSFULLY\n";

  return 0;
}
