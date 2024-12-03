#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IMAGE_WIDTH 92
#define IMAGE_HEIGHT 112
#define NUM_IMAGES 1
#define NUM_EIGENFACES 10

void calculateCovarianceMatrix(double *images, double *covarianceMatrix, double *meanFace, int numImages, int imageSize) {
    // Calculate mean face
    #pragma acc parallel loop present(images[:numImages*imageSize], meanFace[:imageSize])
    for (int columnIndex = 0; columnIndex < imageSize; ++columnIndex) {
        meanFace[columnIndex] = 0;
        #pragma acc loop reduction(+:meanFace[:imageSize])
        for (int rowIndex = 0; rowIndex < numImages; ++rowIndex) {
            meanFace[columnIndex] += images[rowIndex * imageSize + columnIndex];
        }
        meanFace[columnIndex] /= numImages;
    }

    // Subtract mean face from each image
    #pragma acc parallel loop present(images[:numImages*imageSize], meanFace[:imageSize])
    for (int columnIndex = 0; columnIndex < imageSize; ++columnIndex) {
        #pragma acc loop
        for (int rowIndex = 0; rowIndex < numImages; ++rowIndex) {
            images[rowIndex * imageSize + columnIndex] -= meanFace[columnIndex];
        }
    }

    // Calculate covariance matrix (omitted for elapsed time measurement)
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

    // Allocate space for host variables
    double *hostCovarianceMatrix = (double *)malloc(imageSize * imageSize * sizeof(double));
    double *hostMeanFace = (double *)malloc(imageSize * sizeof(double));

    // Measure time before OpenACC computation
    clock_t start = clock();

    // Launch the OpenACC computation
    calculateCovarianceMatrix(hostImages, hostCovarianceMatrix, hostMeanFace, NUM_IMAGES, imageSize);

    // Measure time after OpenACC computation
    clock_t end = clock();
    double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the elapsed time
    printf("Elapsed Time: %.6f seconds\n", elapsed_time);

    // Free allocated memory on host
    free(hostImages);
    free(hostCovarianceMatrix);
    free(hostMeanFace);

    return 0;
}