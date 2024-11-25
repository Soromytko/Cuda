#define BLOCK_SIZE 128
#define MAT_DIM_X 1000000
#define MAT_DIM_Y 128

using namespace std;

#define SIZE_OF_PIXEL sizeof(uchar) * 3

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

static const dim3 s_threadDim(BLOCK_SIZE, BLOCK_SIZE);
static const dim3 s_gridDim(ceil((float)MAT_DIM_X / s_threadDim.x), ceil((float)MAT_DIM_Y/s_threadDim.y));

typedef char mat;
typedef int result_mat;
const int matSize = MAT_DIM_X * MAT_DIM_Y * sizeof(mat);
const int resultMatSize = MAT_DIM_Y * sizeof(mat);

__global__ void calc_zero_count_simple(mat *mat1, mat *mat2, result_mat *resultMat, int dimX, int dimY)
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
        if (mat1[index] == 0 && mat2[index] == 0) {
            resultMat[indexForResultMat]++;
        }
    }
}

__global__ void calc_zero_count_with_shared_mem(mat *mat1, mat *mat2, result_mat *resultMat, int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int cacheIndex = threadIdx.x;

    if (x >= dimX || y >= dimY) {
        return;
    }

    __shared__ int cache[16];

    const int index = x + y * dimX;
    if (mat1[index] == 0 && mat2[index] == 0) {
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

__global__ void calc_zero_count_with_atomic(
    mat *mat1,
    mat *mat2,
    result_mat *matResult,
    int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int cacheIndex = threadIdx.x;

    if (x >= dimX || y >= dimY) {
        return;
    }

    const int index = x + y * dimX;
    if (mat1[index] == 0 && mat2[index] == 0) {
        cuda_atomic_add((matResult + blockIdx.y), 1);
    }
}
