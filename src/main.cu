#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

// Stops the program if a CUDA call fails (same as in vector_add)
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                              \
        }                                                         \
    } while (0)

// Same idea, but for cuBLAS calls
#define CUBLAS_CHECK(call)                                        \
    do {                                                          \
        cublasStatus_t s = (call);                                \
        if (s != CUBLAS_STATUS_SUCCESS) {                         \
            fprintf(stderr, "cuBLAS error at %s:%d: code %d\n",   \
                    __FILE__, __LINE__, (int)s);                  \
            exit(1);                                              \
        }                                                         \
    } while (0)

int main() {
    // ---------- 1. Matrix size ----------
    int N = 4096;
    size_t bytes = (size_t)N * N * sizeof(float);

    // ---------- 2. Make A and B on the CPU, fill with random numbers in [-1, 1] ----------
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    // ---------- 3. Make space on the GPU, copy A and B there ----------
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    // ---------- 4. Set up cuBLAS ----------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;   // so it computes plain C = A x B

    // ---------- 5. Warmup run (not timed) ----------
    // Note: B is passed BEFORE A. That's the row-major trick.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 6. Time 10 runs with CUDA events ----------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int runs = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                 &alpha, d_B, N, d_A, N, &beta, d_C, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / runs;

    // ---------- 7. Print time and GFLOPS ----------
    double flops = 2.0 * N * N * N;    // 2N^3, using double so it doesn't overflow
    double gflops = flops / (avg_ms / 1000.0) / 1e9;
    printf("cuBLAS  N=%d  %.3f ms  %.1f GFLOPS\n", N, avg_ms, gflops);

    // ---------- 8. Clean up ----------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(A);
    free(B);
    return 0;
}
