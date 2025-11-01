#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "matrix_io.h" // include your matrix IO functions

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Helper to flatten 2D matrix to 1D array
float* flattenMatrix(const std::vector<std::vector<float>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    float* arr = new float[rows * cols];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            arr[i * cols + j] = mat[i][j]; // row-major
    return arr;
}

void printMatrix(const float* mat, int n, int m) {
    for (int i = 0; i < n * m; ++i) {
        std::cout << mat[i] << " ";
        if ((i + 1) % m == 0) std::cout << std::endl;
    }
}

int main() {
    int l = 3, n = 2, m = 4;

    // Step 1: generate matrices and save to files
    generateAndSaveMatrices(l, n, m, "matrixA.txt", "matrixB.txt");

    // Step 2: load matrices from files
    auto matA = loadMatrix("matrixA.txt"); // l x n
    auto matB = loadMatrix("matrixB.txt"); // n x m

    float* h_A = flattenMatrix(matA);
    float* h_B = flattenMatrix(matB);
    float* h_C = new float[l * m];

    // Step 3: Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, l * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, l * m * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, l * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, n * m * sizeof(float), cudaMemcpyHostToDevice));

    // Step 4: cuBLAS multiplication
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS is column-major; swap A and B for row-major
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, l, n,
                             &alpha,
                             d_B, m,
                             d_A, n,
                             &beta,
                             d_C, m));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, l * m * sizeof(float), cudaMemcpyDeviceToHost));

    // Step 5: Print results
    std::cout << "Matrix A:" << std::endl; printMatrix(h_A, l, n);
    std::cout << "\nMatrix B:" << std::endl; printMatrix(h_B, n, m);
    std::cout << "\nMatrix C = A * B:" << std::endl; printMatrix(h_C, l, m);

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
