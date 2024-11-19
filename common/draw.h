#ifndef DRAW_H
#define DRAW_H

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

// Function to load image in grayscale
cv::Mat loadImage(const char* filePath);

// Function to convert grayscale image to color
cv::Mat convertToColor(const cv::Mat& grayImage);

// Function to save image to file
void saveImage(const std::string& filePath, const cv::Mat& image);

// Function to draw lines detected by Hough transform
void drawLines(cv::Mat &image, int *accumulator, int threshold, float rMax, float rScale, 
              int degreeBins, float radInc, int rBins, int xCent, int yCent);

#endif