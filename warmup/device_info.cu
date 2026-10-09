#include <cstdio>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);   // 0 = the first GPU

    printf("Name:                   %s\n", p.name);
    printf("Compute capability:     %d.%d\n", p.major, p.minor);
    printf("Number of SMs:          %d\n", p.multiProcessorCount);
    printf("Warp size:              %d\n", p.warpSize);
    printf("Max threads per block:  %d\n", p.maxThreadsPerBlock);
    printf("Max threads per SM:     %d\n", p.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM:      %d\n", p.maxBlocksPerMultiProcessor);
    printf("Registers per SM:       %d\n", p.regsPerMultiprocessor);
    printf("Shared mem per block:   %zu bytes\n", p.sharedMemPerBlock);
    printf("Shared mem per SM:      %zu bytes\n", p.sharedMemPerMultiprocessor);
    printf("Global memory:          %zu MB\n", p.totalGlobalMem / (1024 * 1024));
    return 0;
}
