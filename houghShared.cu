/*
 ============================================================================
 Transformada de Hough con Memoria COMPARTIDA
 
 Para usar: Modificar Makefile a houghShared.cu
 Para ejecutar: ./houghShared <nombre_imagen>
 ============================================================================
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include "common/draw.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************

//*****************************************************************
// MEMORIA CONSTANTE
// Usar memoria constante para la tabla de senos y cosenos
// inicializarlas en main y pasarlas al device
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];
//*****************************************************************

//**********************************************************
// Función para calcular el umbral

int calculateThreshold(int *accumulator, int size, float factor) {
    
    int maxVal = 0;
    for (int i = 0; i < size; i++) {
        if (accumulator[i] > maxVal) maxVal = accumulator[i];
    }
    
    return maxVal * factor;
}

//*********************************************************


// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTranShared (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
    // Calcular: int gloID
    int gloID = blockIdx.x * blockDim.x + threadIdx.x; 

    // MEMORIA COMPARTIDA:
    // a.	Defina en el kernel un locID usando los IDs de los hilos del bloque.
    int locID = threadIdx.x;

    // MEMORIA COMPARTIDA:
    // b.	Defina en el kernel un acumulador local en memoria compartida llamado localAcc, que tenga degreeBins * rBins elementos.
    extern __shared__ int localAcc[];

    // MEMORIA COMPARTIDA:
    // c.	Inicialize a 0 todos los elementos de este acumulador local. Recuerde que la memoria Compartida solamente puede manejarse desde el device (kernel).
    for(int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        localAcc[i] = 0;
    }

    // MEMORIA COMPARTIDA:
    // d.	Incluya una barrera para los hilos del bloque que controle que todos los hilos hayan completado el proceso de inicialización del acumulador local.
    __syncthreads();    
    
    if (gloID > w * h) return;      // in case of extra threads in block

    int xCent = w / 2;
    int yCent = h / 2;

    //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // MEMORIA COMPARTIDA:
    // e.	Modifique la actualización del acumulador global acc para usar el acumulador local localAcc. Para coordinar el acceso a memoria y garantizar que la operación de suma sea completada por cada hilo, use una operación de suma atómica.
    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
        }
    }

    // MEMORIA COMPARTIDA:
    // f.	Incluya una segunda barrera para los hilos del bloque que controle que todos los hilos hayan completado el proceso de incremento del acumulador local.
    __syncthreads();

    // MEMORIA COMPARTIDA:
    // g.	Agregue un loop al final del kernel que inicie en locID hasta degreeBins * rBins . Este loop sumará los valores del acumulador local localAcc al acumulador global acc
    for(int i = locID; i < degreeBins * rBins; i += blockDim.x) {
        atomicAdd(&acc[i], localAcc[i]);
    }
    //utilizar operaciones atomicas para seguridad
    //faltara sincronizar los hilos del bloque en algunos lados
}



//*****************************************************************
int main (int argc, char **argv)
{
    int i;

    PGMImage inImg (argv[1]);

    // Leer imagen usando OpenCV desde draw.cpp
    cv::Mat inputImage = loadImage(argv[1]);
    if (inputImage.empty()) {
        printf("Error al cargar la imagen.\n");
        return -1;
    }

    cv::Mat processedImage;
    GaussianBlur(inputImage, inputImage, Size(3,3), 0);
    Canny(inputImage, processedImage, 30, 100);
    dilate(processedImage, processedImage, Mat(), Point(-1,-1), 1);

    unsigned char* pixels = processedImage.data;

    int *cpuht;
    int w = inputImage.cols;
    int h = inputImage.rows;

    //float* d_Cos;
    //float* d_Sin;

    // MEMORIA CONSTANTE
    // YA NO ES NECESARIO RESERVAR MEMORIA EN EL DEVICE PARA:
    //cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
    //cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

    // CPU calculation
    CPU_HoughTran(pixels, w, h, &cpuht);

    // pre-compute values to be stored
    float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
    float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
    float rad = 0;
    for (i = 0; i < degreeBins; i++)
    {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
    }

    float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // MEMORIA CONSTANTE
    // Copiar los valores precalculados de seno y coseno a memoria constante
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);

    // setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = pixels; // h_in contiene los pixeles de la imagen

    h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

    cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
    cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
    cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

    // Definir eventos de inicio y fin
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inicia la medicion de tiempo
    cudaEventRecord(start, 0);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    //1 thread por pixel

    int blockNum = ceil (w * h / 256);

    // MEMORIA COMPARTIDA:
    // Establecer el tamaño de memoria compartida (acumulador local) como degreeBins * rBins
    size_t sharedMemSize = degreeBins * rBins * sizeof(int);
    GPU_HoughTranShared <<< blockNum, 256, sharedMemSize >>> (d_in, w, h, d_hough, rMax, rScale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error en el lanzamiento del kernel: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // get results from device
    cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Termina la medicion de tiempo
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calcula el tiempo en milisegundos
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Tiempo de ejecución del kernel: %f ms\n", elapsedTime);

    // Liberar eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //*************************************************** */

    // Seccion de output

    // Calcular el umbral
    int threshold = calculateThreshold(h_hough, degreeBins * rBins, 0.46);

    int xCent = w / 2;
    int yCent = h / 2;

    printf("threshold: %d\n", threshold);

    // Dibujar líneas detectadas
    drawLines(inputImage, h_hough, threshold, rMax, rScale, degreeBins, radInc, rBins, xCent, yCent);

    // Guardar imagen con líneas detectadas
    saveImage("output_with_lines.png", inputImage);
    printf("Imagen de salida guardada como output_with_lines.png\n");

    // compare CPU and GPU results
    for (i = 0; i < degreeBins * rBins; i++)
    {
    if (cpuht[i] != h_hough[i])
        printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }

    // ********************************************************************************

    free(cpuht);
    free(pcCos);
    free(pcSin);
    free(h_hough);
    cudaFree(d_in);
    cudaFree(d_hough);

    // MEMORIA CONSTANTE:
    // No es necesario liberar d_Cos y d_Sin

    //cudaFree(d_Cos);
    //cudaFree(d_Sin);
return 0;
}