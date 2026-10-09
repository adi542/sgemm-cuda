DAY:LINUX DONE
Day 1: compiled and ran vector_add with nvcc
Day 1: compiled and ran vector_add with nvcc
Day 2: cuBLAS at N=4096: 34.1 ms, 4028 GFLOPS
Day 3: naive N=4096: 233.0 ms, 589.9 GFLOPS, 15.6% of cuBLAS
Day 3: tiled N=4096: 149.0 ms, 922.3 GFLOPS, 22.3% of cuBLAS (1.5x naive)
Day 4: uncoalesced N=4096: ~1.4% of cuBLAS, ~10x slower than naive (coalescing)
