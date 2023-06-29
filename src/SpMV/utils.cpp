#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
// #include <spmv.cuh>

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
    //print csr
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
    std::cout<< "wating for verify the results ..."<<std::endl;
    // for (uint64_t i = 0; i < numRows; i++) {
    //     const uint64_t row_start =row_ptr[i]; 
    //     const uint64_t row_end =row_ptr[i+1]; 
    //     float sum = 0;
    //     for (uint64_t j = row_start; j < row_end; j++) {    
    //         sum += values[j] * x[col_idx[j]];
    //     }
    //     assert(fabs(y[i] - sum) <= 0.000001);
    // }

    //打印结果向量y中出错的index
    std::vector<int> error_id;
    for (uint64_t i = 0; i < numRows; i++) {
        const uint64_t row_start =row_ptr[i]; 
        const uint64_t row_end =row_ptr[i+1]; 
        float sum = 0;
        for (uint64_t j = row_start; j < row_end; j++) {    
            sum += values[j] * x[col_idx[j]];
        }
        if(fabs(y[i] - sum) > 0.000001){
          error_id.push_back(i);
        }
    }
    std::cout << "\nerror_id: " << std::endl;
    for (int &val : error_id) {
    std::cout << val << " ";
    }

    //打印CPU和GPU计算结果 检查
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
    // std::cout<<"verify finished !"<<std::endl;
}

//计算row_blocks的数量
unsigned int rowAssignment(
  const uint64_t numRows, 
  std::vector<uint64_t> &row_ptr,
  std::vector<uint64_t> &row_assign,
  const uint64_t nnz_per_block
){
  uint64_t last_row_id = 0;
  uint64_t nnz_sum = 0;
  uint64_t num_blocks = 0;
  row_assign.push_back(0);
  std::vector<uint64_t> nnz_detail;
  // std::cout<< "nnz_per_block = "<<nnz_per_block<<std::endl;
  #define UPDATE_ASSIGNMENT_META(row_id) \
    row_assign.push_back(row_id); \
    last_row_id = row_id; \
    nnz_detail.push_back(nnz_sum); \
    nnz_sum = 0; \
    num_blocks++;
  for (uint64_t i = 1; i <= numRows; i++)
  {
    nnz_sum += row_ptr[i] - row_ptr[i-1];
    if (nnz_sum == nnz_per_block)
    {
      // std::cout << "nnz_sum1 =  " <<nnz_sum<<std::endl;
      UPDATE_ASSIGNMENT_META(i);
    }else if (nnz_sum > nnz_per_block)
    {
      if (i-last_row_id > 1)
      {
        i--;
      }
      // std::cout << "nnz_sum2 =  " <<nnz_sum<<std::endl;
      UPDATE_ASSIGNMENT_META(i);
    }else if (i-last_row_id > nnz_per_block)  //在CSR_Stream中计算用来规约的线程数量时，有限制
    {
      // std::cout << "nnz_sum3 =  " <<nnz_sum<<std::endl;
      UPDATE_ASSIGNMENT_META(i);
    }else if (i == numRows)
    {
      // std::cout << "nnz_sum4 =  " <<nnz_sum<<std::endl;
      UPDATE_ASSIGNMENT_META(i);
    }
  }
  #undef UPDATE_ASSIGNMENT_META
  std::cout << "\nrow_assign: " << std::endl;
  for (int i = 0; i < row_assign.size(); i++) {
  std::cout << row_assign[i]<< " ";
  }
  std::cout << "\nnz_detail: " << std::endl;
  for (int i = 0; i < nnz_detail.size(); i++) {
  std::cout << nnz_detail[i]<< " ";
  }
  return num_blocks;
}