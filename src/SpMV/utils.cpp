#include <iostream>
#include <vector>
#include <cstdlib>


// 生成稀疏矩阵
void generateSparseMatrix(
    const int numRows, 
    const int numCols, 
    double density,
    std::vector<float>& values, 
    std::vector<int>& colIndices, 
    std::vector<int>& rowOffsets) 
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

        // 更新 rowOffsets 向量
        rowOffsets.push_back(rowOffsets.back() + numNonZerosInRow);
    }
    //print csr
    std::cout << "row_ptr: ";
    for (int i = 0; i < rowOffsets.size(); i++) {
        std::cout << rowOffsets[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "col_idx: ";
    for (int i = 0; i < colIndices.size(); i++) {
        std::cout << colIndices[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "values: ";
    for (int i = 0; i < values.size(); i++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    
}