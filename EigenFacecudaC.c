#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define IMAGE_WIDTH 175
#define IMAGE_HEIGHT 200
#define NUM_IMAGES 1
#define NUM_EIGENFACES 10
#define BLOCK_SIZE 16

__global__ void calculateCovarianceMatrixKernel(double *deviceImages, double *deviceCovarianceMatrix,
                                                double *deviceMeanFace, int numImages, int imageSize) {
    int columnIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (columnIndex < imageSize) {
        // Calculate mean face
        deviceMeanFace[columnIndex] = 0;
        for (int rowIndex = 0; rowIndex < numImages; ++rowIndex) {
            deviceMeanFace[columnIndex] += deviceImages[rowIndex * imageSize + columnIndex];
        }
        deviceMeanFace[columnIndex] /= numImages;

        // Subtract mean face from each image
        for (int rowIndex = 0; rowIndex < numImages; ++rowIndex) {
            deviceImages[rowIndex * imageSize + columnIndex] -= deviceMeanFace[columnIndex];
        }

        // Calculate covariance matrix (omitted for elapsed time measurement)
    }
}

void readImages(double *images, const char *filename, int numImages, int imageSize) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int imageIndex = 0; imageIndex < numImages; ++imageIndex) {
        for (int pixelIndex = 0; pixelIndex < imageSize; ++pixelIndex) {
            fscanf(file, "%lf", &images[imageIndex * imageSize + pixelIndex]);
        }
    }

    fclose(file);
}

int main() {
    int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    // Allocate and read training images from file
    double *hostImages = (double *)malloc(NUM_IMAGES * imageSize * sizeof(double));
    readImages(hostImages, "training_images.txt", NUM_IMAGES, imageSize);

    // Allocate space for GPU variables
    double *deviceImages, *deviceCovarianceMatrix, *deviceMeanFace;
    cudaMalloc((void **)&deviceImages, NUM_IMAGES * imageSize * sizeof(double));
    cudaMalloc((void **)&deviceCovarianceMatrix, imageSize * imageSize * sizeof(double));
    cudaMalloc((void **)&deviceMeanFace, imageSize * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(deviceImages, hostImages, NUM_IMAGES * imageSize * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate space for the mean face on the host
    double *hostMeanFace = (double *)malloc(imageSize * sizeof(double));

    // Measure time before kernel execution
    clock_t start = clock();

    // Launch the CUDA kernel
    dim3 blocks((imageSize + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 threads(BLOCK_SIZE, 1, 1);
    calculateCovarianceMatrixKernel<<<blocks, threads>>>(deviceImages, deviceCovarianceMatrix,
                                                         deviceMeanFace, NUM_IMAGES, imageSize);
    cudaDeviceSynchronize();

    // Measure time after kernel execution
    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the elapsed time
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);

    // Free allocated memory on both host and device
    free(hostImages);
    free(hostMeanFace);
    cudaFree(deviceImages);
    cudaFree(deviceCovarianceMatrix);
    cudaFree(deviceMeanFace);

    return 0;
}