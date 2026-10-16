__global__ void sgemm_blocktiling_2d(int N, const float* A, const float* B, float* C){

    constexpr int BM = 128;   // rows of C per block
    constexpr int BN = 128;   // columns of C per block
    constexpr int BK = 8;     // chunk width along k
    constexpr int TM = 8;     // rows of the thread's square
    constexpr int TN = 8;     // columns of the thread's square
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);   // 16384 / 64 = 256

    // main code
    // this is loction of the block in C matrix
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int threadCol = threadIdx.x % (BN/TN);
    int threadRow = threadIdx.x / (BN/TN);

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    constexpr int strideA = NUM_THREADS / BK;

    int innerRowB = threadIdx.x / BN;           
    int innerColB = threadIdx.x % BN;       
    constexpr int strideB = NUM_THREADS / BN; 

    float results[TM][TN] = {0.0f};   // the thread's 64 running sums
    float regM[TM];                   // 8 values of A (an 8 x 1 column)
    float regN[TN];                   // 8 values of B (a 1 x 8 row)

    for(int t = 0;t<N/BK;t++){

        //load this chunk vaue in As
        for(int offset = 0;offset<BM;offset+=strideA){
            As[innerRowA+offset][innerColA] = A[(cRow*BM+innerRowA+offset)*N + (t*BK+innerColA)];
        }
        for(int offset = 0;offset<BK;offset+=strideB){
            Bs[innerRowB+offset][innerColB] = B[(t * BK + innerRowB+offset) * N + (cCol*BN+innerColB)];
        }

        __syncthreads();

        for(int dot = 0;dot <BK;dot++){
            for(int i = 0;i<TM;i++){
            regM[i] = As[threadRow * TM + i][dot];
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

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = cRow * BM + threadRow * TM + i;  
            int col = cCol * BN + threadCol * TN + j; 
            C[row * N + col] = results[i][j];
        }
    }


    
    
}
