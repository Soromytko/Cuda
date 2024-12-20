﻿#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <cassert>
#include "cuda.h"

void initRandom()
{
    std::srand( ( unsigned int )std::time( nullptr ) );
}

int getRand(int min, int max)
{
    return rand() % (max + 1 - min) + min;
}

const mat* generateMat(int dimX = MAT_DIM_X, int dimY = MAT_DIM_Y)
{
    mat *result = new mat[dimX * dimY];
    for (int i = 0; i < dimX * dimY; i++) {
      result[i] = getRand(0, 10);
    }
    return result;
}

const result_mat *countZeros(
    const mat *mat1,
    const mat *mat2,
    int dimX = MAT_DIM_X, int dimY = MAT_DIM_Y)
{
    result_mat *result = new result_mat[dimY];
    for (int y = 0; y < dimY; y++) {
        int zeroCount = 0;
        for (int x = 0; x < dimX; x++) {
            const int index = x + y * dimX;
            if (mat1[index] == 0 && mat1[index] == mat2[index]) {
                zeroCount++;
            }
        }
        result[y] = zeroCount;
    }
    return result;
}

bool isMatEqual(
    const result_mat *mat1,
    const result_mat *mat2,
    int count)
{
    for (int i = 0; i < count; i++) {
        if (mat1[i] != mat2[i]) {
            std::cout << "NOT " << i  << " " << mat1[i] << " " << mat2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void printMat(const mat* mat0, int dimX = MAT_DIM_X, int dimY = MAT_DIM_Y)
{
    for (int y = 0; y < dimY; y++) {
        for (int x = 0; x < dimX; x++) {
            const int index = MAT_INDEX(x, y, dimX);
            std::cout << (int)mat0[index] << " ";
        }
        std::cout << std::endl;
    }
}

void calcSimple(const mat *mat1, const mat *mat2, const result_mat *resultMat)
{
    std::cout << "GRID " <<  s_gridDim.x << " " << s_gridDim.y << std::endl;
    mat *dev_mat1, *dev_mat2;
    result_mat *dev_resultMat;

    CHECK(cudaMalloc(&dev_mat1, matSize));
    CHECK(cudaMalloc(&dev_mat2, matSize));
    CHECK(cudaMalloc(&dev_resultMat, MAT_DIM_Y * sizeof(result_mat)));
    CHECK(cudaMemcpy(dev_mat1, mat1, matSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_mat2, mat2, matSize, cudaMemcpyHostToDevice));

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA, 0);
    calc_zero_count_simple_part<<<ceil((float)MAT_DIM_Y / BLOCK_SIZE), BLOCK_SIZE>>>(dev_mat1, dev_mat2, dev_resultMat, MAT_DIM_X, MAT_DIM_Y);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cudaEventDestroy(startCUDA);
    cudaEventDestroy(stopCUDA);

    result_mat *cudaResultMat = new result_mat[MAT_DIM_Y];
    CHECK(cudaMemcpy(cudaResultMat, dev_resultMat, MAT_DIM_Y * sizeof(result_mat), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dev_mat1));
    CHECK(cudaFree(dev_mat2));
    CHECK(cudaFree(dev_resultMat));

    assert(isMatEqual(resultMat, cudaResultMat, MAT_DIM_Y));

    delete cudaResultMat;

    std::cout << "SIMPLE:" << std::endl;
    std::cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    std::cout << "CUDA memory throughput = " << matSize / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";
    std::cout << std::endl;
}

void calcWithSharedMem(const mat *mat1, const mat *mat2, const result_mat *resultMat)
{
    mat *dev_mat1, *dev_mat2;
    result_mat *dev_rawResultMat;

    const int resultCount = MAT_DIM_Y;

    CHECK(cudaMalloc(&dev_mat1, matSize));
    CHECK(cudaMalloc(&dev_mat2, matSize));
    CHECK(cudaMalloc(&dev_rawResultMat, resultCount * sizeof(result_mat)));
    CHECK(cudaMemcpy(dev_mat1, mat1, matSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_mat2, mat2, matSize, cudaMemcpyHostToDevice));

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA, 0);
    calc_zero_count_with_shared_mem<<<s_gridDim, s_threadDim>>>(dev_mat1, dev_mat2, dev_rawResultMat, MAT_DIM_X, MAT_DIM_Y);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cudaEventDestroy(startCUDA);
    cudaEventDestroy(stopCUDA);

    result_mat *cudaResultMat = new result_mat[resultCount];
    CHECK(cudaMemcpy(cudaResultMat, dev_rawResultMat, resultCount * sizeof(result_mat), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dev_mat1));
    CHECK(cudaFree(dev_mat2));
    CHECK(cudaFree(dev_rawResultMat));

    assert(isMatEqual(resultMat, cudaResultMat, MAT_DIM_Y));

    delete cudaResultMat;

    std::cout << "SHARED MEM:" << std::endl;
    std::cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    std::cout << "CUDA memory throughput = " << matSize / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";
    std::cout << std::endl;
}

void calcWithAtomic(const mat *mat1, const mat *mat2, const result_mat *resultMat)
{
    mat *dev_mat1, *dev_mat2;
    result_mat *dev_result;

    CHECK(cudaMalloc(&dev_mat1, matSize));
    CHECK(cudaMalloc(&dev_mat2, matSize));
    CHECK(cudaMalloc(&dev_result, MAT_DIM_Y * sizeof(result_mat)));
    CHECK(cudaMemcpy(dev_mat1, mat1, matSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_mat2, mat2, matSize, cudaMemcpyHostToDevice));

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA, 0);
    calc_zero_count_with_atomic<<<s_gridDim, s_threadDim>>>(dev_mat1, dev_mat2, dev_result, MAT_DIM_X, MAT_DIM_Y);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cudaEventDestroy(startCUDA);
    cudaEventDestroy(stopCUDA);

    // Use {} to initialize with zeros.
    result_mat *cudaResult = new result_mat[MAT_DIM_Y]{};
    CHECK(cudaMemcpy(cudaResult, dev_result, MAT_DIM_Y * sizeof(result_mat), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dev_mat1));
    CHECK(cudaFree(dev_mat2));
    CHECK(cudaFree(dev_result));

    assert(isMatEqual(resultMat, cudaResult, MAT_DIM_Y));

    delete cudaResult;

    std::cout << "ATOMIC:" << std::endl;
    std::cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    std::cout << "CUDA memory throughput = " << matSize / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";
    std::cout << std::endl;
}

int main(void)
{
    initRandom();

    const mat *mat1 = generateMat();
    const mat *mat2 = generateMat();
    const result_mat *resultMat = countZeros(mat1, mat2);

    calcSimple(mat1, mat2, resultMat);
    calcWithSharedMem(mat1, mat2, resultMat);
    calcWithAtomic(mat1, mat2, resultMat);

    delete mat1;
    delete mat2;
    delete resultMat;

    return 0;
}
