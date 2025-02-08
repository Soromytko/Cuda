#include <cstdlib>
#include <ctime>
#include <iostream>

#define BLOCK_SIZE 16
#define MAT_DIM_X 1000
#define MAT_DIM_Y 1000

using namespace cv;
using namespace std;

#define SIZE_OF_PIXEL sizeof(uchar) * 3

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }


__global__ void calculate_zero_count(uchar* sourceData, uchar* resultData, int dimX, int dimY)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) {
        return;
    }

    const int index = (x + y * width) * SIZE_OF_PIXEL;
    const int rightIndex = (x + 1 + y * width) * SIZE_OF_PIXEL;
    const int bottomIndex = (x + (y + 1) * width) * SIZE_OF_PIXEL;

}

void waitExit()
{
  while((cv::waitKey() & 0xEFFFFF) != 27);
}

float getRand(float min, float max)
{
  return srand() % (max + 1 - min) + min;
}

float* generateMat(int dimX, int dimY)
{
    const float *result = new float[dimX * dimY];
    for (int i = 0; i < dimX * dimY; i++) {
      result[i] = getRand(-100, 100);
    }
    return result;
}

int main(void)
{
    const float mat1 = generateMat(MAT_DIM_X, MAT_DIM_Y)
    const float mat2 = generateMat(MAT_DIM_X, MAT_DIM_Y)
    const int countZeroe

    const int width = image.cols;
    const int height = image.rows;
    const int dataElementCount = image.rows * image.cols * 3;
    const int dataSize = dataElementCount * sizeof(uchar);

    uchar* dev_sourceData;
    uchar* dev_resultData;

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    CHECK(cudaMalloc(&dev_sourceData, dataSize));
    CHECK(cudaMalloc(&dev_resultData, dataSize));
    CHECK(cudaMemcpy(dev_sourceData, image.data, dataSize, cudaMemcpyHostToDevice));

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    // dim3 grid(width / 16, height / 16);
    // dim3 threads(16, 16);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil((float)width / threads.x), ceil((float)height/threads.y));

    cudaEventRecord(startCUDA, 0);
    detect_bounds<<<grid, threads>>>(dev_sourceData, dev_resultData, width, height);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cudaEventDestroy(startCUDA);
    cudaEventDestroy(stopCUDA);

    cout << "width " << width << "\nheight " << height << endl;
    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << dataSize / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";

    uchar* resultData = new uchar[dataElementCount];
    CHECK(cudaMemcpy(resultData, dev_resultData, dataSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dev_sourceData));
    CHECK(cudaFree(dev_resultData));

    Mat resultImage = image.clone();
    for(int i = 0; i < dataElementCount; i++) resultImage.data[i] = resultData[i];

    imwrite("pic2.jpg", resultImage);

    //show image
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", resultImage);
    // waitKey(0);
    waitExit();

    return 0;
}
