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

__device__ int getMaxFromArray(const int* data, int count)
{
    int result = data[0];
    for (int i = 1; i < count; i++) {
        if (data[i] > result) {
            result = data[i];
        }
    }
    return result;

}

__global__ void detect_bounds(int* sourceData, int* resultData, int dimX, int dimY)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dimX - 1 || y >= dimY - 1) {
        return;
    }

    int index = (x + y * blockDim.x * gridDim.x) * SIZE_OF_PIXEL;
    int rightIndex = (x + 1 + y * blockDim.x * gridDim.x) * SIZE_OF_PIXEL;
    int bottomIndex = (x + (y + 1) * blockDim.x * gridDim.x) * SIZE_OF_PIXEL;

    int grad_x = getMaxFromArray((sourceData + index), 3);
    int grad_y = getMaxFromArray((sourceData + index), 3);
    int grad = grad_x > grad_y ? grad_x : grad_y;
    if (grad > 40) {
        for (int i = 0; i < 3; i++) {
            resultData[index + i] = 255;
        }
    }
    else {
        for (int i = 0; i < 3; i++) {
            resultData[index + i] = 0;
        }
    }
}

int main(void)
{
    Mat image;
    image = imread("pic.jpg", IMREAD_COLOR);
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    const int width = image.cols;
    const int height = image.rows;
    const int dataElementCount = image.rows * image.cols * 3;
    const int dataSize = dataElementCount * sizeof(int);

    int* dev_sourceData;
    int* dev_resultData;

    cudaEvent_t startCUDA, stopCUDA;
    float elapsedTimeCUDA;

    CHECK(cudaMalloc(&dev_sourceData, dataSize));
    CHECK(cudaMalloc(&dev_resultData, dataSize));
    CHECK(cudaMemcpy(dev_sourceData, image.data, dataSize));

    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    dim3 grid(width / 16, height / 16);
    dim3 threads(16, 16);

    cudaEventRecord(startCUDA, 0);
    detect_bounds<<<grid, threads >>>(dev_sourceData, dev_resultData, width, height);
    cudaEventRecord(stopCUDA, 0);
    cudaEventSynchronize(stopCUDA);
    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cudaEventDestroy(startCUDA);
    cudaEventDestroy(stopCUDA);

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << 3 * N * sizeof(float) / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";

    const int* resultData = new int[dataElementCount];
    CHECK(cudaMemcpy(resultData, dev_resultData, dataSize));

    CHECK(cudaFree(dev_sourceData));
    CHECK(cudaFree(dev_resultData));

    Mat resultImage;
    resultImage.data = resultData;

    imwrite("pic2.jpg", resultImage);

    //show image
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", resultImage);
    waitKey(0);

    return 0;
}
