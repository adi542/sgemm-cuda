__global__ void sgemm_uncoalesced(int N, const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;   // SWAPPED: row now from x
    int col = blockIdx.y * blockDim.y + threadIdx.y;   // SWAPPED: col now from y

    
   if(row < N && col < N){
        float sum = 0.0f;
        for(int k = 0;k<N;k++){
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
}
}
