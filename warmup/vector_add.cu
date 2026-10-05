#include <cstdio>
__global__ void vectorAdd(float* a,float*b,float*c,int N){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i<N){
    c[i] = a[i] + b[i];
  }
}

int main(){
  int N = 8;
  float a[] = {1,2,3,4,5,6,7,8};
  float b[] = {10,20,30,40,50,60,70,80};
  float z[8];
  float *d_a,*d_b,*d_z;
  cudaMalloc((void**)&d_a,N*sizeof(float));
  cudaMalloc((void**)&d_b,N*sizeof(float));
  cudaMalloc((void**)&d_z,N*sizeof(float));
  cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b,N*sizeof(float),cudaMemcpyHostToDevice);
  int threadsPerBlock = 4;
  int block = (N+threadsPerBlock-1)/threadsPerBlock;
  vectorAdd<<<block,threadsPerBlock>>>(d_a,d_b,d_z,N);
  cudaMemcpy(z,d_z,N*sizeof(float),cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++)
    {
        printf("%f ", z[i]);
    }

    // 9. Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_z);

    return 0;
}
