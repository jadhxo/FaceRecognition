#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <lapacke.h>

#define WIDTH 92   // Width of the input images
#define HEIGHT 112 // Height of the input images
#define NUM_IMAGES 1 // Number of training images
#define NUM_EIGENFACES 10 // Number of eigenfaces to use

// Function to read image data from files
void readImages(double **images, char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < NUM_IMAGES; ++i) {
        for (int j = 0; j < WIDTH * HEIGHT; ++j) {
            fscanf(file, "%lf", &images[i][j]);
        }
    }

    fclose(file);
}

// Function to calculate the transpose of a matrix
void transpose(double **A, double **A_T, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A_T[j][i] = A[i][j];
        }
    }
}

// Function to calculate the dot product of two vectors
double dotProduct(double *a, double *b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Function to normalize a vector
void normalizeVector(double *v, int size) {
    double norm = 0.0;
    for (int i = 0; i < size; ++i) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);

    for (int i = 0; i < size; ++i) {
        v[i] /= norm;
    }
}

// Function to calculate the covariance matrix
void calculateCovariance(double **images, double **covarianceMatrix, double *meanFace, int numImages, int image_size) {
    // Calculate mean face
    for (int i = 0; i < image_size; ++i) {
        meanFace[i] = 0;
        for (int j = 0; j < numImages; ++j) {
            meanFace[i] += images[j][i];
        }
        meanFace[i] /= numImages;
    }

    // Subtract mean face from each image
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < image_size; ++j) {
            images[i][j] -= meanFace[j];
        }
    }

    // Calculate covariance matrix
    for (int i = 0; i < image_size; ++i) {
        for (int j = 0; j < image_size; ++j) {
            covarianceMatrix[i][j] = 0;
            for (int k = 0; k < numImages; ++k) {
                covarianceMatrix[i][j] += images[k][i] * images[k][j];
            }
            covarianceMatrix[i][j] /= (numImages - 1);
        }
    }
}

// Function to perform eigenvalue decomposition using LAPACKE
void eigenDecomposition(double **matrix, double **eigenvectors, int size, int num_eigenvectors) {
    // LAPACKE variables
    lapack_int n = size;
    lapack_int lda = size;
    double* w = (double*)malloc(size * sizeof(double));

    // LAPACKE_dsyev function for eigenvalue decomposition
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'L', n, *matrix, lda, w);

    // Check for errors in LAPACKE_dsyev
    if (info > 0) {
        fprintf(stderr, "LAPACKE_dsyev failed with error %d\n", info);
        exit(EXIT_FAILURE);
    }

    // Extract the eigenvectors
    for (int i = 0; i < num_eigenvectors; ++i) {
        for (int j = 0; j < size; ++j) {
            eigenvectors[i][j] = matrix[j][i];
        }
    }

    free(w);
}

double getCurrentTimeInSeconds() {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main() {
    // Read training images from file
    double **images = (double **)malloc(NUM_IMAGES * sizeof(double *));
    for (int i = 0; i < NUM_IMAGES; ++i) {
        images[i] = (double *)malloc(WIDTH * HEIGHT * sizeof(double));
    }
    readImages(images, "training_images.txt");

    // Create space for covariance matrix, eigenfaces, and mean face
    double **covarianceMatrix = (double **)malloc(WIDTH * HEIGHT * sizeof(double *));
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        covarianceMatrix[i] = (double *)malloc(WIDTH * HEIGHT * sizeof(double));
    }

    double **eigenfaces = (double **)malloc(NUM_EIGENFACES * sizeof(double *));
    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        eigenfaces[i] = (double *)malloc(WIDTH * HEIGHT * sizeof(double));
    }

    double *meanFace = (double *)malloc(WIDTH * HEIGHT * sizeof(double));

    int numImages = NUM_IMAGES;
    int image_size = WIDTH * HEIGHT;

    // Start the timer
    double startTime = getCurrentTimeInSeconds();
    
    // Calculate covariance matrix
    calculateCovariance(images, covarianceMatrix, meanFace, numImages, image_size);

    // Perform eigenvalue decomposition to get eigenvectors (eigenfaces)
    eigenDecomposition(covarianceMatrix, eigenfaces, image_size, NUM_EIGENFACES);

    // Normalize the eigenfaces (optional but common step)
    for (int i = 0; i < NUM_EIGENFACES; ++i) {
        normalizeVector(eigenfaces[i], image_size);
    }
    
    // Stop the timer
    double endTime = getCurrentTimeInSeconds();

    // Print the elapsed time
    printf("Elapsed time: %.4f seconds\n", endTime - startTime);
    
    // Free allocated memory
    for (int i = 0; i < NUM_IMAGES; ++i) {
        free(images[i]);
    }
    free(images);

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
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
