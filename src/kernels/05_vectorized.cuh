
__global__ void sgemm_vectorized(int N, const float* A, const float* B, float* C) {   // CHANGED: name

   
    constexpr int BM = 128;   // rows of C per block
    constexpr int BN = 128;   // columns of C per block
    constexpr int BK = 8;     // chunk width along k
    constexpr int TM = 8;     // rows of the thread's square
    constexpr int TN = 8;     // columns of the thread's square

   
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    
    __shared__ float As[BK][BM];                  // CHANGED: transposed, 8 x 128 (k first, then row)
    __shared__ __align__(16) float Bs[BK][BN];    // CHANGED: aligned for float4 stores, 8 x 128

    
    int threadCol = threadIdx.x % (BN / TN);   // 0..15
    int threadRow = threadIdx.x / (BN / TN);   // 0..15

    
    
    int innerRowA = threadIdx.x / (BK / 4);    // 0..127
    int innerColA = threadIdx.x % (BK / 4);    // 0..1

    
    int innerRowB = threadIdx.x / (BN / 4);    // 0..7
    int innerColB = threadIdx.x % (BN / 4);    // 0..31

    
    float results[TM][TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    // =========================================== walk along k: 512 chunks
    for (int t = 0; t < N / BK; t++) {

        
        
        float4 tmp = reinterpret_cast<const float4*>(
            &A[(cRow * BM + innerRowA) * N + (t * BK + innerColA * 4)])[0];
        As[innerColA * 4 + 0][innerRowA] = tmp.x;
        As[innerColA * 4 + 1][innerRowA] = tmp.y;
        As[innerColA * 4 + 2][innerRowA] = tmp.z;
        As[innerColA * 4 + 3][innerRowA] = tmp.w;

        
       
        reinterpret_cast<float4*>(&Bs[innerRowB][innerColB * 4])[0] =
            reinterpret_cast<const float4*>(
                &B[(t * BK + innerRowB) * N + (cCol * BN + innerColB * 4)])[0];

       
        __syncthreads();

        
        for (int dot = 0; dot < BK; dot++) {

           
            for (int i = 0; i < TM; i++) {
                regM[i] = As[dot][threadRow * TM + i];    // CHANGED: indices swapped
            }

            
            for (int j = 0; j < TN; j++) {
                regN[j] = Bs[dot][threadCol * TN + j];
            }

            
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    results[i][j] += regM[i] * regN[j];
                }
            }
        }

       
        __syncthreads();
    }

    // =========================================== WRITE (unchanged)
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = cRow * BM + threadRow * TM + i;
            int col = cCol * BN + threadCol * TN + j;
            C[row * N + col] = results[i][j];
        }
    }
}
