#include <cstdio>
#include <cstdlib>
#include <cmath>                     // NEW: gives us fabsf
#include <cublas_v2.h>
#include "kernels/01_naive.cuh"      // NEW: pulls in your naive kernel
#include "kernels/02_tiled.cuh"

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                              \
        }                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                        \
    do {                                                          \
        cublasStatus_t s = (call);                                \
        if (s != CUBLAS_STATUS_SUCCESS) {                         \
            fprintf(stderr, "cuBLAS error at %s:%d: code %d\n",   \
                    __FILE__, __LINE__, (int)s);                  \
            exit(1);                                              \
        }                                                         \
    } while (0)

int main(){
    int N = 4096;
    size_t bytes = (size_t)N*N*sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    for(int i = 0;i<N*N;i++){
         A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
         B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    float *d_A,*d_B,*d_C,*d_C_mine;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));               
    CUDA_CHECK(cudaMalloc(&d_C_mine, bytes));          
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha=1.0f,beta=0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, d_B, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());
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

    // ---------- 7. Print cuBLAS time and GFLOPS ----------
    double flops = 2.0 * N * N * N;
    double gflops = flops / (avg_ms / 1000.0) / 1e9;
    printf("cuBLAS  N=%d  %.3f ms  %.1f GFLOPS\n", N, avg_ms, gflops);

    dim3 threads(32, 32);                              // 32 x 32 = 1024 threads per block
    dim3 blocks((N + 31) / 32, (N + 31) / 32);         // enough blocks to cover all of C
    sgemm_naive<<<blocks, threads>>>(N, d_A, d_B, d_C_mine);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float *C_ref  = (float*)malloc(bytes);            
    float *C_mine = (float*)malloc(bytes);             
    CUDA_CHECK(cudaMemcpy(C_ref,  d_C,      bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C_mine, d_C_mine, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;

    for (int i = 0; i < N * N; i++) {

        if (fabsf(C_mine[i] - C_ref[i]) > 1e-2f) {
            printf("naive: WRONG at index %d\n", i);
            printf("cuBLAS = %f, naive = %f\n",
                   C_ref[i], C_mine[i]);

            correct = false;
            break;
        }
    }

    if (correct) {
    printf("naive: correct\n");
}


    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        sgemm_naive<<<blocks, threads>>>(N, d_A, d_B, d_C_mine);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms_naive = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    float avg_naive = ms_naive / runs;
    double gflops_naive = flops / (avg_naive / 1000.0) / 1e9;
    printf("naive   N=%d  %.3f ms  %.1f GFLOPS  (%.1f%% of cuBLAS)\n",
           N, avg_naive, gflops_naive, 100.0 * gflops_naive / gflops);

    // ---------- 11. NEW: run tiled once (also the warmup) ----------
    CUDA_CHECK(cudaMemset(d_C_mine, 0, bytes));
    

    dim3 tthreads(TILE_SIZE, TILE_SIZE);
    dim3 tblocks(N / TILE_SIZE, N / TILE_SIZE);
    sgemm_tiled<<<tblocks, tthreads>>>(N, d_A, d_B, d_C_mine);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C_mine, d_C_mine, bytes, cudaMemcpyDeviceToHost));

    correct = true;
    for (int i = 0; i < N * N; i++) {

        if (fabsf(C_mine[i] - C_ref[i]) > 1e-2f) {
            printf("tiled: WRONG at index %d\n", i);
            printf("cuBLAS = %f, tilted = %f\n",
                   C_ref[i], C_mine[i]);

            correct = false;
            break;
        }
    }

    if (correct) {
    printf("tilted: correct\n");
}


 CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
         sgemm_tiled<<<tblocks, tthreads>>>(N, d_A, d_B, d_C_mine);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms_tiled = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, start, stop));
    float avg_tiled = ms_tiled / runs;
    double gflops_tiled = flops / (avg_tiled / 1000.0) / 1e9;
    printf("tiled   N=%d  %.3f ms  %.1f GFLOPS  (%.1f%% of cuBLAS)\n",
           N, avg_tiled, gflops_tiled, 100.0 * gflops_tiled / gflops);


    
    
    

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_mine));                    // NEW
    free(A);
    free(B);
    free(C_ref);                                       // NEW
    free(C_mine);                                      // NEW
    return 0;

    
}
