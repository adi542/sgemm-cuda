#include <cstdio>
#include <cstdlib>
#include <cmath>                     // NEW: gives us fabsf
#include <cublas_v2.h>
#include "kernels/01_naive.cuh"      // NEW: pulls in your naive kernel
#include "kernels/02_tiled.cuh"
#include "kernels/00_uncoalesced.cuh"
#include <iostream>
#include <string>
#include <vector>

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



void run_kernel(int id, int N, const float* d_A, const float* d_B, float* d_C) {
    switch (id) {
        case 0: {   // uncoalesced
            dim3 threads(32, 32);
            dim3 blocks(N / 32, N / 32);
            sgemm_uncoalesced<<<blocks, threads>>>(N, d_A, d_B, d_C);
            break;
        }
        case 1: {   // naive (coalesced)
            dim3 threads(32, 32);
            dim3 blocks(N / 32, N / 32);
            sgemm_naive<<<blocks, threads>>>(N, d_A, d_B, d_C);
            break;
        }
        case 2: {   // shared memory tiled
            dim3 threads(TILE_SIZE, TILE_SIZE);
            dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);
            sgemm_tiled<<<blocks, threads>>>(N, d_A, d_B, d_C);
            break;
        }
        default:
            printf("Unknown kernel id %d\n", id);
            exit(1);
    }
}

bool check_result(const std::string& name, const float* C_mine, const float* C_ref, int N){
     for (int i = 0; i < N * N; i++) {

        if (fabsf(C_mine[i] - C_ref[i]) > 1e-2f) {
           std::cout << name << ": WRONG at index " << i
          << " (cuBLAS = " << C_ref[i] << ", mine = " << C_mine[i] << ")" << std::endl;
            return false;
        }
    
    }
std::cout<<name<<" :Correct"<<std::endl;
return true;
    
}

float time_kernel(int id, int N, const float* d_A, const float* d_B, float* d_C, int runs){
    run_kernel(id,N,d_A,d_B,d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for(int i = 0;i<runs;i++){
        run_kernel(id,N,d_A,d_B,d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms/runs;
    
    
    
}

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

    // rest of the kernal code

    float *C_ref  = (float*)malloc(bytes);            
    float *C_mine = (float*)malloc(bytes);             
    CUDA_CHECK(cudaMemcpy(C_ref,  d_C,      bytes, cudaMemcpyDeviceToHost));
    
    std::vector<std::string> names = {"uncoalesced", "naive", "tiled"};
    int num_kernels = 3;
    for(int i = 0;i<num_kernels;i++){
        CUDA_CHECK(cudaMemset(d_C_mine, 0, bytes));
        run_kernel(i, N, d_A, d_B, d_C_mine);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(C_mine, d_C_mine, bytes, cudaMemcpyDeviceToHost));
        check_result(names[i], C_mine, C_ref, N);
        float avg_kernel = time_kernel(i, N, d_A, d_B, d_C_mine, runs);
        double gflops_kernel = flops / (avg_kernel / 1000.0) / 1e9;
        std::cout<<names[i]<<" N="<<N<<" avg: "<<avg_kernel<<" GFLOPS: "<<gflops_kernel<<
        " ("<<(gflops_kernel/gflops)*100<<"%"<<" of cublas"<<" )"<<std::endl;
        
        
        
        
    }

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
