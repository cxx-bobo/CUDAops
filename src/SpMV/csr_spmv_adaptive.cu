#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <spmv.cuh>
#include <memory>
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

unsigned int rowAssignment(
  const uint64_t numRows, 
  std::vector<uint64_t> &row_ptr,
  std::vector<uint64_t> &row_assign,
  const uint64_t nnz_per_block
);

int main() {
  // Initial constants
  constexpr uint64_t numRows = 1<<10; 
  constexpr uint64_t sizeRow = 1<<9;
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
  std::vector<uint64_t> row_assign; //表示行划分情况
  uint64_t num_blocks;  //一共需要的row_block数量
  
  // Initialize csr and vector_x
  nvtxRangePush("initialize csr and vector with random numbers");
  generateSparseMatrix(numRows, sizeRow, density, values, col_idx, row_ptr);
  for (int i=0; i<sizeRow; i++){
    h_x.push_back(static_cast<float>(rand() % 100));
  };
  nvtxRangePop();

  //Get the number of row_blocks
  std::cout <<"before turn into rowAssignment "<<std::endl;
  num_blocks = rowAssignment(numRows, row_ptr, row_assign, NNZ_PER_WG);
  std::cout <<"\nnum_blocks = "<<num_blocks<<std::endl;
  std::cout <<"row_assign.size() = "<<row_assign.size()<<std::endl;
  // return 0;

  // Allocate device memory
  nvtxRangePush("allocate device memory");
  float *d_values, *d_x, *d_y;
  uint64_t *d_col_idx, *d_row_ptr, *d_row_assign;
  cudaMalloc(&d_values, sizeof(float)*values.size());
  cudaMalloc(&d_col_idx, sizeof(uint64_t)*col_idx.size());
  cudaMalloc(&d_row_ptr, sizeof(uint64_t)*row_ptr.size());
  cudaMalloc(&d_x, size_x);
  cudaMalloc(&d_y, size_y);
  cudaMalloc(&d_row_assign, sizeof(uint64_t)*row_assign.size());
  nvtxRangePop();

  // Copy data to the device
  nvtxRangePush("copy data from host to device memory");
  cudaMemcpy(d_values, values.data(), sizeof(float)*values.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, col_idx.data(), sizeof(uint64_t)*col_idx.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr, row_ptr.data(), sizeof(uint64_t)*row_ptr.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x.data(), size_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_assign, row_assign.data(),sizeof(uint64_t)*row_assign.size(),cudaMemcpyHostToDevice);
  nvtxRangePop();

  // Threads per CTA dimension
  int threads_per_CTAdim = NNZ_PER_WG;

  // Blocks per grid dimension 
  int blocks_per_GRIDdim = num_blocks;

  // Launch kernel
  std::cout << "Launch Kernel: " << threads_per_CTAdim << " threads per block, " << blocks_per_GRIDdim << " blocks in the grid" << std::endl;
  nvtxRangePush("Launch kernel");
  csr_spmv_adaptive_kernel<<<blocks_per_GRIDdim, threads_per_CTAdim>>>(numRows, d_col_idx, d_row_ptr, d_row_assign, d_values, d_x, d_y);
  
  //会阻塞当前的 CPU 线程，直到前面的所有 CUDA kernel 执行完成。
  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess){
    printf("kernel launch failed with error \"%s\".\n",
    cudaGetErrorString(cudaerr));
    exit(-1);
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

  std::cout << "\ncsr_spmv_adaptive completed successfully !\n";

  return 0;
}
