#include "draw.h"

// Función para cargar una imagen en escala de grises
cv::Mat loadGrayImage(const char* filePath) {
    return imread(filePath, IMREAD_GRAYSCALE);
}

// Función para convertir una imagen a color
cv::Mat convertToColor(const cv::Mat& grayImage) {
    cv::Mat colorImage;
    cvtColor(grayImage, colorImage, COLOR_GRAY2BGR);
    return colorImage;
}

// Función para guardar una imagen
void saveImage(const std::string& filePath, const cv::Mat& image) {
    imwrite(filePath, image);
}

// Función para dibujar líneas
void drawLines(cv::Mat &image, int *accumulator, int threshold, float rMax, float rScale, int degreeBins, float radInc, int rBins, int xCent, int yCent) {
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            if (accumulator[rIdx * degreeBins + tIdx] > threshold) {
                float r = rIdx * rScale - rMax;
                float theta = tIdx * radInc;

                // Calcular puntos para la línea
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a * r, y0 = b * r;

                pt1.x = cvRound(x0 + 1000 * (-b)) + xCent;
                pt1.y = cvRound(y0 + 1000 * (a)) + yCent;
                pt2.x = cvRound(x0 - 1000 * (-b)) + xCent;
                pt2.y = cvRound(y0 - 1000 * (a)) + yCent;


                // Dibujar la línea en color rojo
                line(image, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
            }
        }
    }
}
