#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <OpenCL/cl.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#define BLOCK_SIZE 16

using namespace cv;
using namespace std;

void waitExit()
{
    // Wait Esc
    while((cv::waitKey() & 0xEFFFFF) != 27);
}

cl::Platform getPlatform()
{
    // Get all platforms (drivers), e.g. NVIDIA.
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    
    cl::Platform result = all_platforms[0];
    std::cout << "Using platform: "<< result.getInfo<CL_PLATFORM_NAME>() << "\n";

    return result;
}

cl::Device getDevice(cl::Platform platform)
{
    // Get default device (CPUs, GPUs) of the default platform.
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size() == 0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    // use device[1] because that's a GPU; device[0] is the CPU.
    cl::Device result = all_devices[1];
    std::cout<< "Using device: "<< default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    return result;
}

cl::Program createProgram(cl::Context context, const std::string &programSrc)
{
    // Create the program that we want to execute on the device.
    cl::Program::Sources sources;
    sources.push_back({programSrc.c_str(), programSrc.lenght()});

    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

    return program;
}

std::string loadSrc(const std::string &path)
{
    std::fstream fragStream(path);
    if (!fragStream.is_open())
    {
        std::cout << "Failed to open file " << path << std::endl;
        return false;
    }

    std::string result((std::istreambuf_iterator<char>(fragStream)), std::istreambuf_iterator<char>());
    return result;
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

    cl::Platform default_platform = getPlatform();
    cl::Device default_device = getDevice(default_platform);
    // A context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    cl::Context context({default_device});

    const std::string &src = loadSrc("bitmap.clh");
    cl::Program program = createProgram(context, src);

    // Create a queue (a queue of commands that the GPU will execute).
    cl::CommandQueue queue(context, default_device);

    // Create buffers on device (allocate space on GPU).
    cl::Buffer buffer_SourceData(context, CL_MEM_READ_WRITE, dataSize);
    cl::Buffer buffer_ResultData(context, CL_MEM_READ_WRITE, dataSize);
    
    // Push write commands to queue.
    queue.enqueueWriteBuffer(buffer_SourceData, CL_TRUE, 0, dataSize, image.data);

    // Run kernel.
    cl::kernel kernel(program, "detect_bounds"); 
    cl::NDRange global(ceil((float)width / BLOCK_SIZE), ceil((float)height / BLOCK_SIZE));
    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);
    cl::KernelFunctor detect_bounds = cl::KernelFunctor<cl::Buffer&, cl::Buffer&, int, int>(kernel);
    cl::EnqueueArgs args(queue, global, local);
    detect_bounds(args, buffer_SourceData, buffer_ResultData, width, height);

    // Read result from GPU to here
    uchar* resultData = new uchar[dataElementCount];
    queue.enqueueReadBuffer(buffer_ResultData, CL_TRUE, 0, dataSize, resultData);

    Mat resultImage = image.clone();
    for(int i = 0; i < dataElementCount; i++) resultImage.data[i] = resultData[i];

    imwrite("pic2.jpg", resultImage);

    // Show image
    namedWindow("Display window", WINDOW_AUTOSIZE);
    imshow("Display window", resultImage);
    waitExit();

    return 0;
}
