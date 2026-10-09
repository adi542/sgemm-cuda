#define TILE_SIZE 32
__global__ void sgemm_tiled(int N,const float* A,const float* B,float* C){
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    for(int tile = 0;tile < N/TILE_SIZE;tile++){
        int A_col = tile * TILE_SIZE + tx;
        tileA[ty][tx] = A[row * N + A_col];
        int B_row = tile * TILE_SIZE + ty;
        tileB[ty][tx] = B[B_row * N + col];
        __syncthreads();
        for(int k = 0;k<TILE_SIZE;k++){
            sum += tileA[ty][k] * tileB[k][tx];
        }
        __syncthreads();
        
    }
     C[row * N + col] = sum;

}
