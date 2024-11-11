#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cassert>

#define BLOCK_SIZE 16
#define MAT_DIM_X 1000000
#define MAT_DIM_Y 128
#define THREAD_PER_BLOCK 512

using namespace std;

#define SIZE_OF_PIXEL sizeof(uchar) * 3

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

typedef char mat;
const int matSize = MAT_DIM_X * MAT_DIM_Y * sizeof(mat);
const int resultMatSize = MAT_DIM_Y * sizeof(mat);

__global__ void calculate_zero_count_simple(mat *mat1, mat *mat2, mat *resultMat, int dimX, int dimY)
{
    const int x = threadIdx.x * blockDim.x + blockIdx.x * blockDim.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dimX || y >= dimY) {
        return;
    }

    const int indexForResultMat = blockIdx.x + y * gridDim.x;
    resultMat[indexForResultMat] = 0;
    for (int i = 0; i < blockDim.x && x + i < dimX; i++) {
        const int index = (x + i + y * dimX);
        if (mat1[i] == 0 && mat1[i] == mat2[i]) {
            resultMat[indexForResultMat]++;
        }
    }
}

__global__ void calculate_zero_count_with_shared_mem(mat *mat1, mat *mat2, mat *resultMat, int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int cacheIndex = threadIdx.x;

    if (x >= dimX || y >= dimY) {
        return;
    }

    __shared__ int cache[16];

    const int index = x + y * dimX;
    if (mat1[index] == 0 && mat1[index] == mat2[index]) {
        cache[cacheIndex] = 1;
    } else {
        cache[cacheIndex] = 0;
    }

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] = cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0) {
        resultMat[0] = cache[0];
    }
}

int getRand(int min, int max)
{
    return srand(time(0)) % (max + 1 - min) + min;
}

const mat* generateMat(int dimX = MAT_DIM_X, int dimY = MAT_DIM_Y)
{
    mat *result = new mat[dimX * dimY];
    for (int i = 0; i < dimX * dimY; i++) {
      result[i] = getRand(0, 10);
    }
    return result;
}

const mat *calculateZeros(const mat *mat1, const mat *mat2, int dimX = MAT_DIM_X, int dimY = MAT_DIM_Y)
{
    mat *result = new mat[dimY];
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

bool isMatEqual(const mat *mat1, const mat *mat2, int dimX = MAT_DIM_X, int dimY = MAT_DIM_Y)
{
    const int count = dimX * dimY;
    for (int i = 0; i < count; i++) {
        if (mat1[i] != mat2[i]) {
            return false;
        }
    }
    return true;
}

void calcSimple(const mat *mat1, const mat *mat2, const mat *resultMat)
{
    char *dev_mat1, *dev_mat2, *dev_resultMat;

    CHECK(cudaMalloc(&dev_mat1, matSize));
    CHECK(cudaMalloc(&dev_mat2, matSize));
    CHECK(cudaMalloc(&dev_resultMat, matSize));
    CHECK(cudaMemcpy(dev_mat1, mat1, matSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_mat2, mat2, matSize, cudaMemcpyHostToDevice));

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA, 0);
    calculate_zero_count_simple<<<grid, threads>>>(dev_mat1, dev_mat2, dev_resultMat, MAT_DIM_X, MAT_DIM_Y);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cudaEventDestroy(startCUDA);
    cudaEventDestroy(stopCUDA);

    mat *cudaResultMat = new mat[matSize];
    CHECK(cudaMemcpy(cudaResultMat, dev_resultMat, matSize, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dev_mat1));
    CHECK(cudaFree(dev_mat2));
    CHECK(cudaFree(dev_resultMat));

    assert(isMatEqual(resultMat, cudaResultMat));

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << matSize / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";
}

void calcWithSharedMem(const mat *mat1, const mat *mat2, const mat *resultMat)
{
    char *dev_mat1, *dev_mat2, *dev_resultMat, *dev_sharedResultMat;

    CHECK(cudaMalloc(&dev_mat1, matSize));
    CHECK(cudaMalloc(&dev_mat2, matSize));
    CHECK(cudaMalloc(&dev_resultMat, matSize));
    CHECK(cudaMalloc(&dev_sharedResultMat, resultMatSize));
    CHECK(cudaMemcpy(dev_mat1, mat1, matSize, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_mat2, mat2, matSize, cudaMemcpyHostToDevice));

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    cudaEventRecord(startCUDA, 0);
    calculate_zero_count_simple<<<grid, threads>>>(dev_mat1, dev_mat2, dev_resultMat, MAT_DIM_X, MAT_DIM_Y);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cudaEventDestroy(startCUDA);
    cudaEventDestroy(stopCUDA);

    mat *cudaResultMat = new mat[matSize];
    CHECK(cudaMemcpy(cudaResultMat, dev_resultMat, matSize, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dev_mat1));
    CHECK(cudaFree(dev_mat2));
    CHECK(cudaFree(dev_resultMat));

    assert(isMatEqual(resultMat, cudaResultMat));

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << matSize / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";
}

int main(void)
{
    const mat *mat1 = generateMat();
    const mat *mat2 = generateMat();
    const mat *resultMat = calculateZeros(mat1, mat2);

    calcSimple(mat1, mat2, resultMat);
    calcWithSharedMem(mat1, mat2, resultMat);

    delete mat1;
    delete mat2;
    delete resultMat;

    return 0;
}
