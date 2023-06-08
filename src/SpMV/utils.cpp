#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

// 生成稀疏矩阵
void generateSparseMatrix(
    const uint64_t numRows, 
    const uint64_t numCols, 
    double density,
    std::vector<float> &values, 
    std::vector<uint64_t> &colIndices, 
    std::vector<uint64_t> &rowOffsets) 
{
    // 初始化 rowOffsets 向量
    rowOffsets.push_back(0);

    // 逐行生成矩阵元素
    for (int i = 0; i < numRows; i++) {
        int numNonZerosInRow = 0;

        // 逐列生成非零元素
        for (int j = 0; j < numCols; j++) {
            double randValue = static_cast<double>(rand()) / RAND_MAX;
            if (randValue < density) {
                values.push_back(rand() % 100);
                colIndices.push_back(j);
                numNonZerosInRow++;
            }
        }

        // update rowOffsets 向量
        rowOffsets.push_back(rowOffsets.back() + numNonZerosInRow);
    }
    // //print csr
    // std::cout << "row_ptr: ";
    // for (int i = 0; i < rowOffsets.size(); i++) {
    //     std::cout << rowOffsets[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "col_idx: ";
    // for (int i = 0; i < colIndices.size(); i++) {
    //     std::cout << colIndices[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "values: ";
    // for (int i = 0; i < values.size(); i++) {
    //     std::cout << values[i] << " ";
    // }
    // std::cout << std::endl;
}

//检查结果是否正确
void verifySpMVresult(
    const uint64_t numRows, 
    const uint64_t numCols, 
    std::vector<float> &values, 
    std::vector<uint64_t> &col_idx, 
    std::vector<uint64_t> &row_ptr,
    std::vector<float> &x,
    std::vector<float> &y
){
    // For every row...
    for (uint64_t i = 0; i < numRows; i++) {
        const uint64_t row_start =row_ptr[i]; 
        const uint64_t row_end =row_ptr[i+1]; 
        float sum = 0;
        // For every column...
        for (uint64_t j = row_start; j < row_end; j++) {    
            sum += values[j] * x[col_idx[j]];
        }
        assert(fabs(y[i] - sum) <= 0.000001);
    }
    // //打印CPU和GPU计算结果 检查
    // std::vector<float> sum;
    // for (uint64_t i = 0; i < numRows; i++) {
    //     const uint64_t row_start =row_ptr[i]; 
    //     const uint64_t row_end =row_ptr[i+1]; 
    //     float tmp = 0;
    //     // For every column...
    //     for (uint64_t j = row_start; j < row_end; j++) {    
    //         tmp += values[j] * x[col_idx[j]];
    //     }
    //     sum.push_back(tmp);
    // }
    // std::cout << "\nVECTOR x: " << std::endl;
    // for (float &val : x) {
    // std::cout << val << " ";
    // }
    // std::cout << "\nGPU RESULT: " << std::endl;
    // for (int i = 0; i < numRows; i++) {
    // std::cout << y[i] << " ";
    // }
    // std::cout << "\nCPU RESULT: " << std::endl;
    // for (float &val : sum) {
    // std::cout << val << " ";
    // }
}