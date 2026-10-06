#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                              \
        }                                                         \
    } while (0)




__global__ void vectorAdd(float* a,float*b,float*c,int N){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<N){
    c[i] = a[i] + b[i];
  }
}

int main(){
  int N = 1<<20;
  size_t bytes = N * sizeof(float);
  float *a =(float*)malloc(bytes);
  float *b =(float*)malloc(bytes);
  float *z =(float*)malloc(bytes);
  for (int i = 0; i < N; i++)
  {
    a[i] = i;
    b[i] = i+2;
  }
  float *d_a,*d_b,*d_z;
  CUDA_CHECK(cudaMalloc((void**)&d_a,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_b,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_z,N*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b,b,N*sizeof(float),cudaMemcpyHostToDevice));

  

  int threadsPerBlock = 256;
  int block = (N+threadsPerBlock-1)/threadsPerBlock;
  vectorAdd<<<block,threadsPerBlock>>>(d_a,d_b,d_z,N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); 
  CUDA_CHECK(cudaMemcpy(z,d_z,N*sizeof(float),cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++)
    {
        if(z[i] != a[i] + b[i]){
          printf("mismatch at index %d\n", i);
          return 1;

        }
    }
  printf("everything is fine\n");

    // 9. Free GPU memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_z));
    free(a);
    free(b);
    free(z);

    return 0;
}
