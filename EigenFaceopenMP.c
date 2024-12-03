#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <lapacke.h>
#include <omp.h>

#define IMAGE_WIDTH 92     // Width of the input images
#define IMAGE_HEIGHT 112   // Height of the input images
#define NUM_TRAINING_IMAGES 1   // Number of training images
#define NUM_EIGENFACES 10   // Number of eigenfaces to use

// Function to read image data from files
void readImages(double **trainingImages, char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
        for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT; ++j) {
            fscanf(file, "%lf", &trainingImages[i][j]);
        }
    }

    fclose(file);
}

// Function to calculate the transpose of a matrix
void calculateTranspose(double **matrixA, double **transposeA, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposeA[j][i] = matrixA[i][j];
        }
    }
}

// Function to calculate the dot product of two vectors
double calculateDotProduct(double *vectorA, double *vectorB, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += vectorA[i] * vectorB[i];
    }
    return result;
}

// Function to normalize a vector
void normalizeVector(double *vector, int size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += vector[i] * vector[i];
    }
    norm = sqrt(norm);

    for (int i = 0; i < size; ++i) {
        vector[i] /= norm;
    }
}

// Function to calculate the covariance matrix
void calculateCovariance(double **trainingImages, double **covarianceMatrix, double *meanFace, int numImages, int imageSize) {
    // Calculate mean face
    for (int i = 0; i < imageSize; ++i) {
        meanFace[i] = 0;
        for (int j = 0; j < numImages; ++j) {
            meanFace[i] += trainingImages[j][i];
        }
        meanFace[i] /= numImages;
    }

    // Subtract mean face from each image
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            trainingImages[i][j] -= meanFace[j];
        }
    }

    // Calculate covariance matrix
    for (int i = 0; i < imageSize; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            covarianceMatrix[i][j] = 0;
            for (int k = 0; k < numImages; ++k) {
                covarianceMatrix[i][j] += trainingImages[k][i] * trainingImages[k][j];
            }
            covarianceMatrix[i][j] /= (numImages - 1);
        }
    }
}

// Function to perform eigenvalue decomposition using LAPACKE
void performEigenDecomposition(double **matrix, double **eigenvectors, int size, int numEigenvectors) {
    // LAPACKE variables
    lapack_int matrixSize = size;
    lapack_int leadingDimensionA = size;
    double* eigenvalues = (double*)malloc(size * sizeof(double));

    // LAPACKE_dsyev function for eigenvalue decomposition
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'L', matrixSize, *matrix, leadingDimensionA, eigenvalues);

    // Check for errors in LAPACKE_dsyev
    if (info > 0) {
        fprintf(stderr, "LAPACKE_dsyev failed with error %d\n", info);
        exit(EXIT_FAILURE);
    }

    // Extract the eigenvectors
    for (int i = 0; i < numEigenvectors; ++i) {
        for (int j = 0; j < size; ++j) {
            eigenvectors[i][j] = matrix[j][i];
        }
    }

    free(eigenvalues);
}

double getCurrentTimeInSeconds() {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main() {
    // Read training images from file
    double **trainingImages = (double **)malloc(NUM_TRAINING_IMAGES * sizeof(double *));
    for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
        trainingImages[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
    }
    readImages(trainingImages, "training_images.txt");

    // Create space for covariance matrix, eigenfaces, and mean face
    double **covarianceMatrix = (double **)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double *));
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        covarianceMatrix[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
    }

    double **eigenfaces = (double **)malloc(NUM_EIGENFACES * sizeof(double *));
    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        eigenfaces[i] = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));
    }

    double *meanFace = (double *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(double));

    int numTrainingImages = NUM_TRAINING_IMAGES;
    int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    // Start the timer
    double startTime = getCurrentTimeInSeconds();
    
    // Calculate covariance matrix
    calculateCovariance(trainingImages, covarianceMatrix, meanFace, numTrainingImages, imageSize);

    // Perform eigenvalue decomposition to get eigenvectors (eigenfaces)
    performEigenDecomposition(covarianceMatrix, eigenfaces, imageSize, NUM_EIGENFACES);

    // Normalize the eigenfaces (optional but common step)
    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        normalizeVector(eigenfaces[i], imageSize);
    }
    
    // Stop the timer
    double endTime = getCurrentTimeInSeconds();

    // Print the elapsed time
    printf("Elapsed time: %.4f seconds\n", endTime - startTime);
    
    // Free allocated memory
    for (int i = 0; i < NUM_TRAINING_IMAGES; ++i) {
        free(trainingImages[i]);
    }
    free(trainingImages);

    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
        free(covarianceMatrix[i]);
    }
    free(covarianceMatrix);

    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        free(eigenfaces[i]);
    }
    free(eigenfaces);

    free(meanFace);

    return 0;
}
