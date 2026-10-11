# SGEMM from scratch in CUDA

Optimizing single-precision matrix multiplication (C = A x B) step by step on an
NVIDIA Tesla T4, following Simon Boehm's worklog. Every kernel is written by hand,
checked for correctness against cuBLAS, and benchmarked.

## Results

N = 4096, FP32, Tesla T4. Average of 10 runs after a warmup.

| Kernel | Time (ms) | GFLOPS | % of cuBLAS |
| --- | --- | --- | --- |
| 0. Uncoalesced naive | 2220.8 | 61.9 | 1.6% |
| 1. Naive (coalesced) | 242.7 | 566.4 | 14.4% |
| 2. Shared memory tiling | 155.8 | 882.2 | 22.4% |
| 3. 1D blocktiling | 73.5 | 1868.8 | 47.4% |
| cuBLAS | 34.9 | 3941.2 | 100% |

## The kernels

### 0. Uncoalesced naive
Each thread computes one element of C, but threads in a warp are mapped to
different rows. So when a warp reads A and writes C, its 32 addresses are N floats
apart: 32 separate memory transactions instead of 1. Most of each fetch is wasted.

### 1. Naive (coalesced)
Same code, but row and column are swapped so that threadIdx.x moves along columns.
Now within a warp, the read of A is a broadcast (all 32 threads read the same
value), and the reads of B and writes of C are coalesced into single transactions.
Two changed lines, ~10x faster.

### 2. Shared memory tiling
Each block loads a 32x32 tile of A and of B from global memory (DRAM) into shared
memory (fast on-chip SRAM), which all threads in the block then reuse. Each value
is loaded from DRAM once per block instead of once per thread: 32x fewer global
loads. It is only ~1.5x faster because the bottleneck moves to shared memory: every
multiply-add still needs about 1.25 shared memory reads.

### 3. 1D blocktiling
Each thread computes 8 outputs in a column instead of 1. At each step it reads one
value of B from shared memory into a register and reuses it for all 8 outputs, so
shared memory reads per multiply-add drop from 1.25 to 0.375 (measured in the SASS).
Smaller blocks (512 threads) also let 2 blocks run on each SM at once.

## Things I measured along the way

- **Coalescing:** swapping two index lines made the same kernel ~10x slower.
- **Tile size:** a 16x16 tile (4 blocks per SM) was slower than 32x32 (1 block per SM).
  More blocks did not help; less data reuse hurt.
- **Clock throttling:** under heavy load the T4 hits its 70 W power limit and drops
  from 1590 MHz to ~1140 MHz, so its real FP32 peak is ~5.8 TFLOPS, not the 8.1 on
  the spec sheet.
- **Machine code (SASS):** shared memory reads per multiply-add dropped from
  1.25 (tiling) to 0.375 (1D blocktiling).



## Reference

Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
