// #define BLOCK_SIZE 64
// #define MAT_DIM_X 128
#define BLOCK_SIZE 16
#define MAT_DIM_X 128
#define MAT_DIM_Y 1000000

#define MAT_INDEX(x, y, dimX) x + y * dimX

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        std::cout << "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

static const dim3 s_threadDim(BLOCK_SIZE, BLOCK_SIZE);
static const dim3 s_gridDim(ceil((float)MAT_DIM_X / s_threadDim.x), ceil((float)MAT_DIM_Y / s_threadDim.y));

typedef char mat;
typedef int result_mat;
const int matSize = MAT_DIM_X * MAT_DIM_Y * sizeof(mat);

// __global__ void calc_zero_count_simple(mat *mat1, mat *mat2, result_mat *resultMat, int dimX, int dimY)
// {
//     const int x = threadIdx.x * blockDim.x + blockIdx.x * blockDim.x * blockDim.x;
//     const int y = threadIdx.y + blockIdx.y * blockDim.y;

//     if (x >= dimX || y >= dimY) {
//         return;
//     }

//     const int indexForResultMat = MAT_INDEX(threadIdx.x + blockIdx.x * blockDim.x, y, gridDim.x);
//     resultMat[indexForResultMat] = 0;
//     for (int i = 0; i < blockDim.x && x + i < dimX; i++) {
//         const int index = MAT_INDEX(x + i, y, dimX);
//         if (mat1[index] == 0 && mat2[index] == 0) {
//             resultMat[indexForResultMat]++;
//         }
//     }
// }

__global__ void calc_zero_count_simple_part1(mat *mat1, mat *mat2, result_mat *resultMat, int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dimX || y >= dimY) {
        return;
    }

    if (threadIdx.x == 0) {
        const int resultIndex = MAT_INDEX(blockIdx.x, y, gridDim.x);
        resultMat[resultIndex] = 0;
        for (int i = 0; i < blockDim.x && x + i < dimX; i++) {
            const int index = MAT_INDEX(x + i, y, dimX);
            if (mat1[index] == 0 && mat2[index] == 0) {
                resultMat[resultIndex]++;
            }
        }
    }

    



    // const int index = MAT_INDEX(x, y, dimX);
    // if (mat1[index] == 0 && mat2[index] == 0) {
    //     resultMat[index] = 1;
    // } else {
    //     resultMat[index] = 0;
    // }

    // __syncthreads();
    // {
    //     int i = blockDim.x / 2;
    //     while (i != 0) {
    //         if (threadIdx.x < i && x + i < dimX) {
    //             resultMat[MAT_INDEX(x, y, dimX)] += resultMat[MAT_INDEX(x + i, y, dimX)];
    //         }
    //         __syncthreads();
    //         i /= 2;
    //     }
    // }

    // __syncthreads();
    // if (threadIdx.x == 0)
    // {
    //     int i = ceil(gridDim.x / 2.0);
    //     // int i = gridDim.x / 2;
    //     while (i != 0) {
    //         if (blockIdx.x < i && (blockIdx.x + i) * blockDim.x < dimX) {
    //             resultMat[MAT_INDEX(blockIdx.x * blockDim.x, y, dimX)] += resultMat[MAT_INDEX((blockIdx.x + i) * blockDim.x, y, dimX)];
    //         }
    //         __syncthreads();
    //         i /= 2;
    //     }
    // }

    // __syncthreads();
    // if (threadIdx.x == 0) {
    //     for (int i = 1; i < blockDim.x && x + i < dimX; i++) {
    //         resultMat[MAT_INDEX(x, y, dimX)] += resultMat[MAT_INDEX(x + i, y, dimX)];
    //     }
    // }

    // __syncthreads();
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     for (int i = 1; i < gridDim.x && blockDim.x * i < dimX; i++) {
    //         resultMat[MAT_INDEX(0, y, dimX)] += resultMat[MAT_INDEX(blockDim.x * i, y, dimX)];
    //     }
    // }

}

__global__ void calc_zero_count_simple_part2(mat *mat1, mat *mat2, result_mat *resultMat, int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dimX || y >= dimY) {
        return;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 1; i < gridDim.x; i++) {
            resultMat[MAT_INDEX(0, y, gridDim.x)] += resultMat[MAT_INDEX(i, y, gridDim.x)];
        }
    }
}

//https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/examples/chapter07/my-atomic-add.cu
__device__ int cuda_atomic_add(int *address, int value)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + value);

    // Loop while the guess is incorrect.
    while (oldValue != guess)
    {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + value);
    }

    return oldValue;
}

__global__ void calc_zero_count_with_shared_mem(mat *mat1, mat *mat2, result_mat *resultMat, int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dimX || y >= dimY) {
        return;
    }

    __shared__ int cache[BLOCK_SIZE][BLOCK_SIZE];
    
    const int matIndex = MAT_INDEX(x, y, dimX);
    const int cacheIndexX = threadIdx.x;
    const int cacheIndexY = threadIdx.y;

    if (mat1[matIndex] == 0 && mat2[matIndex] == 0) {
        cache[cacheIndexX][cacheIndexY] = 1;
    } else {
        cache[cacheIndexX][cacheIndexY] = 0;
    }

    __syncthreads();
    {
        int i = blockDim.x / 2;
        while (i != 0) {
            if (cacheIndexX < i && x + i < dimX) {
                cache[cacheIndexX][cacheIndexY] += cache[cacheIndexX + i][cacheIndexY];
            }
            __syncthreads();
            i /= 2;
        }
    }

    if (threadIdx.x == 0) {
        cuda_atomic_add(resultMat + y, cache[0][cacheIndexY]);
    }
}

__global__ void calc_zero_count_with_atomic(
    mat *mat1,
    mat *mat2,
    result_mat *matResult,
    int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dimX || y >= dimY) {
        return;
    }

    const int index = MAT_INDEX(x, y, dimX);
    if (mat1[index] == 0 && mat2[index] == 0) {
        cuda_atomic_add(matResult + y, 1);
    }
}
