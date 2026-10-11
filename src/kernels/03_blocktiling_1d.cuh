__global__ void sgemm_blocktiling_1d(int N, const float* A, const float* B, float* C){
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int threadCol = threadIdx.x % BN;
    int threadRow = threadIdx.x / BN;
    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;
    float result[TM] = {0.0f};

    for(int t = 0;t<N/BK;t++){
        As[innerRowA][innerColA] = A[(cRow * BM + innerRowA) * N + (t * BK + innerColA)];
        Bs[innerRowB][innerColB] = B[(t * BK + innerRowB) * N + (cCol * BN + innerColB)];
        __syncthreads();

        for(int dot = 0;dot<BK;dot++){
        float b = Bs[dot][threadCol];
        for(int i = 0;i<TM;i++){
            result[i] += As[threadRow * TM + i][dot] * b;
        }
    }
        __syncthreads();

    }


    


    for (int i = 0; i < TM; i++) {
        int row = cRow * BM + threadRow * TM + i;
        int col = cCol * BN + threadCol;
        C[row * N + col] = result[i];
    }
    
    
}
