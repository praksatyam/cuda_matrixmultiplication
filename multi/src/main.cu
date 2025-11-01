#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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

float* generateRandomMatrix(int n, int m) {
    float* mat = new float[n * m];
    for (int i = 0; i < n * m; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return mat;
}

void printMatrix(const float* mat, int n, int m) {
    for (int i = 0; i < n * m; ++i) {
        std::cout << mat[i] << " ";
        if ((i + 1) % m == 0) std::cout << std::endl;
    }
}

int main() {
    int n = 10000; 
    int k = 2000; 
    int m = 4000; 

    float* h_A = generateRandomMatrix(n, k);
    float* h_B = generateRandomMatrix(k, m);
    float* h_C = new float[n * m]; 

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, n * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, k * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, n * m * sizeof(float)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, n * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * m * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_B, m,
                             d_A, k,
                             &beta,
                             d_C, m));

 
    CUDA_CHECK(cudaMemcpy(h_C, d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Matrix A:" << std::endl; printMatrix(h_A, n, k);
    std::cout << "\nMatrix B:" << std::endl; printMatrix(h_B, k, m);
    std::cout << "\nMatrix C = A * B:" << std::endl; printMatrix(h_C, n, m);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
