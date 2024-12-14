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

__global__ void calc_zero_count_simple_part(mat *mat1, mat *mat2, result_mat *resultMat, int dimX, int dimY)
{
    const int y = threadIdx.x + blockIdx.x * blockDim.x;

    if (y >= dimY) {
        return;
    }

    resultMat[y] = 0;
    for (int i = 0; i < dimX; i++) {
        const int index = MAT_INDEX(i, y, dimX);
        if (mat1[index] == 0 && mat2[index] == 0) {
            resultMat[y] += 1;
        }
    }

}

__global__ void calc_zero_count_with_shared_mem(mat *mat1, mat *mat2, result_mat *resultMat, int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dimX || y >= dimY) {
        return;
    }

    __shared__ int cache[BLOCK_SIZE];

    const int matIndex = MAT_INDEX(x, y, dimX);
    const int cacheIndex = threadIdx.y;

    if (threadIdx.x == 0) {
        cache[cacheIndex] = 0;
    }

    __syncthreads();

    if (mat1[matIndex] == 0 && mat2[matIndex] == 0) {
        atomicAdd(cache + cacheIndex, 1);
    }

    if (threadIdx.x == 0) {
        atomicAdd(resultMat + y, cache[cacheIndex]);
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
        atomicAdd(matResult + y, 1);
    }
}
