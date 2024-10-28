#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

float max1(float a, float b, float c) {
  if (a>=b && a>=c) return a;
  if (b>=a && b>=c) return b;
  if (c>=a && c>=b) return c;
}

int main( int argc, char** argv )
{
    Mat image;
    image = imread("pic.jpg", IMREAD_COLOR);   // Read the file
    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    Mat res = image.clone();
    for(int i = 0; i < image.rows-1; i++)
    {
        //pointer to 1st pixel in row
        Vec3b* p = image.ptr<Vec3b>(i);
        Vec3b* q = image.ptr<Vec3b>(i+1);
        for (int j = 0; j < image.cols-1; j++)  {
          float grad_x = max1(fabs(p[j+1][0] - p[j][0]), fabs(p[j+1][1] - p[j][1]), fabs(p[j+1][2] - p[j][2]));
          float grad_y = max1(fabs(q[j][0] - p[j][0]), fabs(q[j][1] - p[j][1]), fabs(q[j][2] - p[j][2]));
          float grad = max(grad_x, grad_y);
          if (grad > 40)
            for (int ch=0; ch<3; ch++) res.data[i*image.cols*3 + j*3 +ch] = 255;
          else
            for (int ch=0; ch<3; ch++) res.data[i*image.cols*3 + j*3 +ch] = 0;
          }
    }
    imwrite("pic2.jpg", res);

    //show image
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", res );                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
