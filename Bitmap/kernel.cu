#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#define BLOCK_SIZE 16
#define MAX_GRAD 40

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

__device__ int getMax(int v0, int v1, int v2)
{
    return max(max(v0, v1), v2);
}

__device__ int getGrad(const uchar* data, const int i0, const int i1)
{
    const uchar r0 = data[i0 + 2];
    const uchar g0 = data[i0 + 1];
    const uchar b0 = data[i0 + 0];

    const uchar r1 = data[i1 + 2];
    const uchar g1 = data[i1 + 1];
    const uchar b1 = data[i1 + 0];

    const int diffR = abs(r0 - r1);
    const int diffG = abs(g0 - g1);
    const int diffB = abs(b0 - b1);

    return getMax(diffR, diffG, diffB);
}

__global__ void detect_bounds(uchar* sourceData, uchar* resultData, int width, int height)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width - 1 || y >= height - 1) {
        return;
    }

    const int index = (x + y * width) * SIZE_OF_PIXEL;
    const int rightIndex = (x + 1 + y * width) * SIZE_OF_PIXEL;
    const int bottomIndex = (x + (y + 1) * width) * SIZE_OF_PIXEL;

    const int grad_right = getGrad(sourceData, index, rightIndex);
    const int grad_bottom = getGrad(sourceData, index, bottomIndex);
    const int grad = grad_right > grad_bottom ? grad_right : grad_bottom;
    if (grad > MAX_GRAD) {
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

void waitExit()
{
    // Wait Esc
    while((cv::waitKey() & 0xEFFFFF) != 27);
}

int main(void)
{
    Mat image;
    image = imread("pic.jpg", IMREAD_COLOR);
    if(!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

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
