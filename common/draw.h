#ifndef DRAW_H
#define DRAW_H

#include <opencv2/opencv.hpp>
using namespace cv;

cv::Mat loadGrayImage(const char* filePath);
cv::Mat convertToColor(const cv::Mat& grayImage);
void saveImage(const std::string& filePath, const cv::Mat& image);
void drawLines(cv::Mat &image, int *accumulator, int threshold, float rMax, float rScale, int degreeBins, float radInc, int rBins, int xCent, int yCent);

#endif
