#include "draw.h"

cv::Mat loadImage(const char* filePath) {
    return imread(filePath);
}

cv::Mat convertToColor(const cv::Mat& grayImage) {
    cv::Mat colorImage;
    cvtColor(grayImage, colorImage, COLOR_GRAY2BGR);
    return colorImage;
}

void saveImage(const std::string& filePath, const cv::Mat& image) {
    imwrite(filePath, image);
}

void drawLines(cv::Mat &image, int *accumulator, int threshold, float rMax, float rScale, 
              int degreeBins, float radInc, int rBins, int xCent, int yCent) {

    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            // Solo dibuja líneas que superen el threshold
            if (accumulator[rIdx * degreeBins + tIdx] > threshold) {
                float r = rIdx * rScale - rMax;
                float theta = tIdx * radInc;
                
                // Calcular puntos extremos de la línea
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a * r, y0 = b * r;
                
                // Extender la línea hasta los bordes de la imagen
                pt1.x = cvRound(x0 + 1000 * (b)) + xCent;
                pt1.y = cvRound(-y0 + 1000 * (a)) + yCent;
                pt2.x = cvRound(x0 - 1000 * (b)) + xCent;
                pt2.y = cvRound(-y0 - 1000 * (a)) + yCent;
                
                // Dibujar línea en rojo
                line(image, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
            }
        }
    }
}