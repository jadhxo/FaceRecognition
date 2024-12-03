#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <lapacke.h>

#define IMAGE_WIDTH 92   // Width of the input images
#define IMAGE_HEIGHT 112 // Height of the input images
#define NUM_TRAINING_IMAGES 1 // Number of training images
#define NUM_EIGENFACES 10 // Number of eigenfaces to use

// Function to read image data from files
void readTrainingImages(double **trainingImages, char *filename) {
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
void calculateTranspose(double **matrix, double **transposeMatrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposeMatrix[j][i] = matrix[i][j];
        }
    }
}

// Function to calculate the dot product of two vectors
double calculateDotProduct(double *a, double *b, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += a[i] * b[i];
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
void calculateCovarianceMatrix(double **trainingImages, double **covarianceMatrix,
                                double *meanImage, int numImages, int imageSize) {
    // Calculate mean image
    for (int i = 0; i < imageSize; ++i) {
        meanImage[i] = 0;
        for (int j = 0; j < numImages; ++j) {
            meanImage[i] += trainingImages[j][i];
        }
        meanImage[i] /= numImages;
    }

    // Subtract mean image from each training image
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < imageSize; ++j) {
            trainingImages[i][j] -= meanImage[j];
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
void performEigenvalueDecomposition(double **matrix, double **eigenvectors, int size, int numEigenvectors) {
    // LAPACKE variables
    lapack_int n = size;
    lapack_int lda = size;
    double* eigenvalues = (double*)malloc(size * sizeof(double));

    // LAPACKE_dsyev function for eigenvalue decomposition
    lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'L', n, *matrix, lda, eigenvalues);

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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read training images from file in one process
    double **trainingImages = NULL;
    double *meanImage = NULL;
    double **covarianceMatrix = NULL;
    double **eigenvectors = NULL;
    int numImages = NUM_TRAINING_IMAGES;
    int imageSize = IMAGE_WIDTH * IMAGE_HEIGHT;

    if (rank == 0) {
        trainingImages = (double **)malloc(numImages * sizeof(double *));
        for (int i = 0; i < numImages; ++i) {
            trainingImages[i] = (double *)malloc(imageSize * sizeof(double));
        }

        readTrainingImages(trainingImages, "training_images.txt");

        meanImage = (double *)malloc(imageSize * sizeof(double));

        covarianceMatrix = (double **)malloc(imageSize * sizeof(double *));
        for (int i = 0; i < imageSize; ++i) {
            covarianceMatrix[i] = (double *)malloc(imageSize * sizeof(double));
        }

        eigenvectors = (double **)malloc(NUM_EIGENFACES * sizeof(double *));
        for (int i = 0; i < NUM_EIGENFACES; ++i) {
            eigenvectors[i] = (double *)malloc(imageSize * sizeof(double));
        }
    }

    // Broadcast necessary data to all processes
    MPI_Bcast(&numImages, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        trainingImages = (double **)malloc(numImages * sizeof(double *));
        for (int i = 0; i < numImages; ++i) {
            trainingImages[i] = (double *)malloc(imageSize * sizeof(double));
        }

        meanImage = (double *)malloc(imageSize * sizeof(double));

        covarianceMatrix = (double **)malloc(imageSize * sizeof(double *));
        for (int i = 0; i < imageSize; ++i) {
            covarianceMatrix[i] = (double *)malloc(imageSize * sizeof(double));
        }

        eigenvectors = (double **)malloc(NUM_EIGENFACES * sizeof(double *));
        for (int i = 0; i < NUM_EIGENFACES; ++i) {
            eigenvectors[i] = (double *)malloc(imageSize * sizeof(double));
        }
    }

    MPI_Bcast(&(trainingImages[0][0]), numImages * imageSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Create a local covariance matrix for each process
    double **localCovarianceMatrix = (double **)malloc(imageSize * sizeof(double *));
    for (int i = 0; i < imageSize; ++i) {
        localCovarianceMatrix[i] = (double *)malloc(imageSize * sizeof(double));
    }

    // Start the timer
    double startTime = getCurrentTimeInSeconds();

    // Calculate local covariance matrix
    calculateCovarianceMatrix(trainingImages, localCovarianceMatrix, meanImage, numImages, imageSize);

    // Perform global reduction to get the final covariance matrix
    MPI_Allreduce(localCovarianceMatrix[0], covarianceMatrix[0], imageSize * imageSize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Perform eigenvalue decomposition to get eigenvectors (eigenfaces)
    if (rank == 0) {
        performEigenvalueDecomposition(covarianceMatrix, eigenvectors, imageSize, NUM_EIGENFACES);

        // Normalize the eigenvectors (optional but common step)
        for (int i = 0; i < NUM_EIGENFACES; ++i) {
            normalizeVector(eigenvectors[i], imageSize);
        }

        // Stop the timer
        double endTime = getCurrentTimeInSeconds();

        // Print the elapsed time
        printf("Elapsed time: %.4f seconds\n", endTime - startTime);
    }

    // Free allocated memory
    for (int i = 0; i < numImages; ++i) {
        free(trainingImages[i]);
    }
    free(trainingImages);

    for (int i = 0; i < imageSize; ++i) {
        free(localCovarianceMatrix[i]);
    }
    free(localCovarianceMatrix);

    if (rank == 0) {
        for (int i = 0; i < imageSize; ++i) {
            free(covarianceMatrix[i]);
        }
        free(covarianceMatrix);

        for (int i = 0; i < NUM_EIGENFACES; ++i) {
            free(eigenvectors[i]);
        }
        free(eigenvectors);

        free(meanImage);
    }

    MPI_Finalize();

    return 0;
}
