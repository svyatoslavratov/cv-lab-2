#pragma once
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
typedef cv::Vec<float, 5> Vec5f;

typedef vector<vector<vector<vector<double>>>> filter;

filter getDistribution(double x_mu, double x_sigma)
{
    mt19937 gen((std::random_device())());
    normal_distribution<double> nd(x_mu, x_sigma);
    filter result;

    for (int i = 0; i < 3; i++)
    {
        result.push_back(vector<vector<vector<double>>>());

        for (int j = 0; j < 3; j++)
        {
            result[i].push_back(vector<vector<double>>());

            for (int k = 0; k < 3; k++)
            {
                result[i][j].push_back(vector<double>());

                for (int l = 0; l < 5; l++)
                {
                    result[i][j][k].push_back(nd(gen));
                }
            }
        }
    }

    return result;
}

cv::Mat convolution(cv::Mat img, int step, filter filters)
{
    int filtersWidth = filters.size();
    int filtersHeight = filters[0].size();
    int filtersChannels = filters[0][0].size();
    int filtersCount = filters[0][0][0].size();
    int layerWidth = int((img.rows - filtersWidth) / step + 1);
    int layerHeight = int((img.rows - filtersHeight) / step + 1);
    cv::Mat layer = cv::Mat(cv::Size(layerWidth, layerHeight), CV_32FC(filtersCount));

    for (int i = 0; i < filtersCount; i++)
    {
        for (int j = 0; j < layerHeight; j++)
        {
            for (int k = 0; k < layerWidth; k++)
            {
                double resVal = 0;
                for (int n = 0; n < filtersHeight; n++)
                {
                    for (int m = 0; m < filtersWidth; m++)
                    {
                        for (int l = 0; l < filtersChannels; l++)
                        {
                            resVal += img.at<cv::Vec3b>(k * step + m, j * step + n)[l] * filters[n][m][l][i];
                        }
                    }
                }
                layer.at<Vec5f>(j, k)[i] = resVal;
            }
        }
    }

    return layer;
}

cv::Mat normalize(cv::Mat img, int gamma, float beta)
{
    vector<double> std;
    vector<double> mean;
    cv::meanStdDev(img, mean, std);
    cv::Mat tmp;
    cv::subtract(img, mean[0], tmp);
    cv::Mat div;
    cv::divide(tmp, sqrt(std[0] * std[0]), div);
    div = div.mul(gamma);

    return div;
}

cv::Mat relu(cv::Mat img)
{
    for (int i = 0; i < img.cols; i++)
    {
        for (int j = 0; j < img.rows; j++)
        {
            for (int k = 0; k < 5; k++)
            {
                if (img.at<Vec5f>(i, j)[k] < 0)
                {
                    img.at<Vec5f>(i, j)[k] = 0;
                }
            }
        }
    }

    return img;
}

cv::Mat maxPooling(cv::Mat img, int height, int width)
{
    int count = 5;
    int layerWidth = int(img.rows / height);
    int layerHeight = int(img.cols / width);
    cv::Mat layer = cv::Mat(cv::Size(layerWidth, layerHeight), CV_32FC(count));

    for (int i = 0; i < layerHeight; i = i + 3)
    {
        for (int j = 0; j < layerWidth; j = j + 3)
        {
            Vec5f layerVal = Vec5f(0, 0, 0, 0, 0);
            for (int n = height * i; n < height * (i + 1); n++)
            {
                for (int m = width * j; m < width * (j + 1); m++)
                {
                    for (int k = 0; k < 5; k++)
                    {
                        if (img.at<Vec5f>(m, n)[k] > layerVal[k])
                            layerVal[k] = img.at<Vec5f>(m, n)[k];
                    }
                }
            }
            layer.at<Vec5f>(i, j) = layerVal;
        }
    }

    return layer;
}

cv::Mat softmax(cv::Mat img)
{
    cv::Mat layer = cv::Mat(cv::Size(img.cols, img.rows), CV_32FC(5));
    Vec5f max = Vec5f(0, 0, 0, 0, 0);

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int k = 0; k < 5; k++)
            {
                if (img.at<Vec5f>(i, j)[k] > max[k])
                    max[k] = img.at<Vec5f>(i, j)[k];
            }
        }
    }

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            Vec5f t(0, 0, 0, 0, 0);
            float sum = 0;
            for (int k = 0; k < 5; k++)
            {
                t[k] = exp(img.at<Vec5f>(i, j)[k] - max[k]);
                sum += t[k];
            }
            for (int k = 0; k < 5; k++)
            {
                layer.at<Vec5f>(i, j)[k] = t[k] / sum;
            }
        }
    }

    return layer;
}
