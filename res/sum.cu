#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;

#define SIZE_OF_PIXEL sizeof(int) * 3

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

// размер массива ограничен максимальным размером пространства потоков
__global__ void sum_simple(float *a, float *b, float *c, int N)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N)
        c[i] = a[i]+b[i];
}

// работает даже для очень больших массивов
__global__ void sum_universal(float *a, float *b, float *c, int N)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int threadsNum = blockDim.x*gridDim.x;
    for (int i = id; i < N; i+=threadsNum)
        c[i] = a[i]+b[i];
}




__device__ int maxFromArray(int *array, int count)
{
  int result = array[0];
  for (int i = 1; i < count; i++) {
    if (array[i] > result) {
      result = array[i];
    }
  }

  return result;

}

__global__ void detect_bounds(int *sourceData, int *resultData)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
  	int y = threadIdx.y + blockIdx.y * blockDim.y;
  	int offset = x + y * blockDim.x * gridDim.x;
    offset = offset * SIZE_OF_PIXEL;

    // struct Pixel {
    //   int r, g, b;
    // };
    //
    // Pixel pixel {
    //   sourceData[offset + 2],
    //   sourceData[offset + 1],
    //   sourceData[offste + 0],
    // };
    //
    // Pixel pixelX {
    //   sourceData[offset + 2],
    //   sourceData[offset + 1],
    //   sourceData[offste + 0],
    // };
    //
    // Pixel pixelY {
    //   sourceData[offset + 2],
    //   sourceData[offset + 1],
    //   sourceData[offste + 0],
    // };

    int grad_x = maxFromArray(sourceData + offset);
    int grad_y = maxFromArray(sourceData + offset);
    int grad = grad_x > grad_y ? grad_x : grad_y;
    if (grad > 40) {
      for (int i = 0; i < 3; i++) resultData[offset + i] = 255;
    } else {
      for (int i = 0; i < 3; i++) resultData[offset + i] = 0;
    }




}

int main(void)
{
    Mat image;
    image = imread("pic.jpg", IMREAD_COLOR);   // Read the file
    if(!image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    int *dev_sourceData;
    int *dev_resultData;

    const int width = image.cols;
    const int height = image.rows;
    const int dataSize = image.rows * image.cols * 3 * sizeof(int);

    CHECK(cudaMalloc(&dev_sourceData, dataSize));
    CHECK(cudaMalloc(&dev_resultData, dataSize));
    CHECK(cudaMemcpy(dev_sourceData, image.data, dataSize));

    dim3 grid(width / SIZE_OF_PIXEL, height / SIZE_OF_PIXEL);
    dim3 threads(SIZE_OF_PIXEL, SIZE_OF_PIXEL);

    detect_bounds<<<grid, threads>>>(dev_sourceData, dev_resultData);

    int N = 10*1000*1000;
    float *host_a, *host_b, *host_c, *host_c_check;
    float *dev_a, *dev_b, *dev_c;

    cudaEvent_t startCUDA, stopCUDA;
    clock_t startCPU;
    float elapsedTimeCUDA, elapsedTimeCPU;

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);
    host_a = new float[N];
    host_b = new float[N];
    host_c = new float[N];
    host_c_check = new float[N];
    for (int i = 0; i < N; i++)
    {
        host_a[i] = i;
        host_b[i] = 2*i;
    }
    startCPU = clock();

//#pragma omp parallel for
    for (int i = 0; i < N; i++) host_c_check[i] = host_a[i] + host_b[i];
    elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
    cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
    cout << "CPU memory throughput = " << 3*N*sizeof(float)/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";

    CHECK( cudaMalloc(&dev_a, N*sizeof(float)) );
    CHECK( cudaMalloc(&dev_b, N*sizeof(float)) );
    CHECK( cudaMalloc(&dev_c, N*sizeof(float)) );
    CHECK( cudaMemcpy(dev_a, host_a, N*sizeof(float),cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(dev_b, host_b, N*sizeof(float),cudaMemcpyHostToDevice) );

    cudaEventRecord(startCUDA,0);

    // размер массива ограничен максимальным размером пространства потоков
    sum_simple<<<(N+511)/512, 512>>>(dev_a, dev_b, dev_c, N);

    // работает даже для очень больших массивов
    //sum_universal<<<100, 512>>>(dev_a, dev_b, dev_c, N);

    cudaEventRecord(stopCUDA,0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << 3*N*sizeof(float)/elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";

    CHECK( cudaMemcpy(host_c, dev_c, N*sizeof(float),cudaMemcpyDeviceToHost) );

    // check
    for (int i = 0; i < N; i++)
        if (abs(host_c[i] - host_c_check[i]) > 1e-6)
        {
            cout << "Error in element N " << i << ": c[i] = " << host_c[i]
                 << " c_check[i] = " << host_c_check[i] << "\n";
            exit(1);
        }
    CHECK( cudaFree(dev_a) );
    CHECK( cudaFree(dev_b) );
    CHECK( cudaFree(dev_c) );
    return 0;
}
