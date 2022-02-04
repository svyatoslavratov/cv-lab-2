#include <iostream>
#include <opencv2/opencv.hpp>
#include "lib.h"

using namespace std;

void main()
{
    string imgPath = "../assets/balls.jpg";
    cv::Mat srcImg = cv::imread(imgPath);
    cout << "Initial params: " << srcImg.cols << ", " << srcImg.rows << ", " << srcImg.channels() << endl;

    auto filters = getDistribution(0, 1);

    cv::Mat convolutionLayer = convolution(srcImg, 1, filters);
    cout << "After convolution: " << convolutionLayer.cols << ", " << convolutionLayer.rows << ", " << convolutionLayer.channels() << endl;

    cv::Mat normalizeLayer = normalize(convolutionLayer, 1, 1);
    cout << "After normalize: " << normalizeLayer.cols << ", " << normalizeLayer.rows << ", " << normalizeLayer.channels() << endl;

    cv::Mat reluLayer = relu(normalizeLayer);
    cout << "After relu: " << reluLayer.cols << ", " << reluLayer.rows << ", " << reluLayer.channels() << endl;

    cv::Mat maxPoolingLayer = maxPooling(reluLayer, 2, 2);
    cout << "After max pooling: " << maxPoolingLayer.cols << ", " << maxPoolingLayer.rows << ", " << maxPoolingLayer.channels() << endl;

    cv::imshow("Image", srcImg);
    cv::waitKey(0);

    system("pause");
}
