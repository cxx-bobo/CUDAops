/*!
 * \file basic_aggregate,cu
 * \brief basic version of gnn aggregate operator
 * \author Chenxi
 * \date March 29, 2023
*/

#include <iostream>
#include <vector>
#include <cassert>
#include <nvToolsExt.h>
#include <algorithm>

__global__ void basicAggregate(
    const int *matrix_W,
    const int *matrix_H,
    const int *vector_b,
    int *matrix_HK,
    int N){
    //check kernel shape
    assert(blockDim.x == blockDim.y);
    assert(gridDim.x == gridDim.y);

    //Compute each thread's global row and colum index
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    //Initialize destination element
    int dest_index = global_row * N +global_col;
    matrix_HK[dest_index] = 0;

    //Compute W * H(k)
    for (int k = 0; k < N; k++){
        //Accumulate partial results for a single element
        matrix_HK[dest_index] += matrix_W[global_row * N + k] * matrix_H[k * N + global_col];
    }
    //Compute W * H(k) + d
    //matrix_HK[dest_index] += vector_b[global_row];
}

//Check result on cpu
void verify_result(
    std::vector<int> &matrix_W,
    std::vector<int> &matrix_H,
    std::vector<int> &vector_b,
    std::vector<int> &matrix_HK,
    int N){
    //For every row
    for(int i = 0; i < N; i++){
        //For every column
        for(int j = 0; j < N; j++){
            int tmp = 0;
            //Accumulate partial results for a single element
            for(int k = 0; k < N; k++){
                tmp += matrix_W[i * N + k] * matrix_H[k * N + j];
            }
            //tmp += vector_b[i];
            //Check
            assert(tmp == matrix_HK[i * N + j]);
        }
    }
}

int main(){
    //Matrix size of 1024 x 1024
    constexpr int N = 1 << 10;
    constexpr int matrix_size = N*N*sizeof(int);
    constexpr int vector_size = N*sizeof(int);

    //Create matrices in host memory
    std::vector<int> h_matrix_W(N*N);
    std::vector<int> h_matrix_H(N*N);
    std::vector<int> h_vector_b(N);
    std::vector<int> h_matrix_Hk(N*N);

    //Initialize matrices
    std::generate(h_matrix_W.begin(), h_matrix_W.end(), []() { return rand() % 100; });
    std::generate(h_matrix_H.begin(), h_matrix_H.end(), []() { return rand() % 100; });
    std::generate(h_vector_b.begin(), h_vector_b.end(), []() { return rand() % 100; });

    //Allocate device memory
    int *d_matrix_W, *d_matrix_H, *d_vector_b, *d_matrix_HK;
    cudaMalloc(&d_matrix_W, matrix_size);
    cudaMalloc(&d_matrix_H, matrix_size);
    cudaMalloc(&d_vector_b, vector_size);
    cudaMalloc(&d_matrix_HK, matrix_size);

    //Copy data from host memory to device memory
    cudaMemcpy(d_matrix_W, h_matrix_W.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_H, h_matrix_H.data(), matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, h_vector_b.data(), vector_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_matrix_HK, h_matrix_Hk.data(), matrix_size, cudaMemcpyHostToDevice);

    //Initialize kernel configuration
    //Number of threads per block (one dimension)
    int NUM_THREADS_PER_BLOCK = 32;
    
    //Number of blocks per grid (onde dimension)
    int NUM_BLOCKS_PER_GRID = N % NUM_THREADS_PER_BLOCK ?
                              N / NUM_THREADS_PER_BLOCK :
                              N / NUM_THREADS_PER_BLOCK + 1;
    
    //Use dim3 structure of block and grid dimensions
    dim3 threads(NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK);
    dim3 blocks(NUM_BLOCKS_PER_GRID, NUM_BLOCKS_PER_GRID);

    //Launch kernel
    //std::cout << "Launch Kernel: " << threads << " threads per block, " << blocks << " blocks in the grid" << std::endl;
    basicAggregate<<<blocks, threads>>>(d_matrix_W, d_matrix_H, d_vector_b, d_matrix_HK, N);

    //cudaDeviceSynchronize();

    //Copy result back to host memory
    cudaMemcpy(h_matrix_Hk.data(), d_matrix_HK, matrix_size, cudaMemcpyDeviceToHost);

    //Verify result
    verify_result(h_matrix_W, h_matrix_H, h_vector_b, h_matrix_Hk, N);

    //Free memory on device
    cudaFree(d_matrix_W);
    cudaFree(d_matrix_H);
    cudaFree(d_vector_b);
    cudaFree(d_matrix_HK);

    std::cout<< "Get correct basic_aggregate result" << std::endl;

    return 0;


}