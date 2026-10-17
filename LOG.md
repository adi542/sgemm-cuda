DAY:LINUX DONE
Day 1: compiled and ran vector_add with nvcc
Day 1: compiled and ran vector_add with nvcc
Day 2: cuBLAS at N=4096: 34.1 ms, 4028 GFLOPS
Day 3: naive N=4096: 233.0 ms, 589.9 GFLOPS, 15.6% of cuBLAS
Day 3: tiled N=4096: 149.0 ms, 922.3 GFLOPS, 22.3% of cuBLAS (1.5x naive)
Day 4: uncoalesced N=4096: ~1.4% of cuBLAS, ~10x slower than naive (coalescing)
Day 4: tiled with TILE=16: 714 GFLOPS (17.7%), slower than TILE=32 (~900). More blocks didn't help; less reuse hurt.
Day 6: blocktiling_1d N=4096: 73.5 ms, 1869 GFLOPS, 47.4% of cuBLAS (2.1x tiled)
Day 8: blocktiling_2d N=4096: ~41-43 ms, ~3200-3350 GFLOPS, ~78-93% of cuBLAS across runs (~1.75x 1D blocktiling)
Day 9: vectorized (kernel 6) N=4096: ~40 ms, ~3420 GFLOPS, ~87% of cuBLAS. No real gain over kernel 5: regs 124->102 but still 2 blocks/SM (cliff at 85); LDG 8->2 (LDG.E.128) but loads weren't the bottleneck.
