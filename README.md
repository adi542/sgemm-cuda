# SGEMM from scratch in CUDA

Optimizing single-precision matrix multiplication (C = A x B) step by step on an
NVIDIA Tesla T4, following Simon Boehm's worklog. Every kernel is written by hand,
checked for correctness against cuBLAS, benchmarked, and analysed with the compiler's
register reports and the SASS machine code.

## Results

N = 4096, FP32, Tesla T4 (Kaggle). Typical values across several runs, each run being
the average of 10 launches after a warmup. Run-to-run variation is about 5-10%, mostly
because the T4 throttles its clock at its 70 W power limit.

| Kernel | Time (ms) | GFLOPS | % of cuBLAS |
| --- | --- | --- | --- |
| 0. Uncoalesced naive | ~2222 | ~62 | ~1.6% |
| 1. Naive (coalesced) | ~240 | ~570 | ~15% |
| 2. Shared memory tiling | ~155 | ~880 | ~22% |
| 3. 1D blocktiling | ~73 | ~1880 | ~50% |
| 4. 2D blocktiling | ~41 | ~3300 | ~85% |
| 5. Vectorized loads | ~40 | ~3400 | ~85% |
| cuBLAS | ~35 | ~3900 | 100% |

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
multiply-add still needs about 1.25 shared memory reads, and with 1024-thread blocks
only one block fits per SM, so all warps stall together at each barrier.

### 3. 1D blocktiling
Each thread computes 8 outputs in a column instead of 1. At each step it reads one
value of B from shared memory into a register and reuses it for all 8 outputs, so
shared memory reads per multiply-add drop from 1.25 to 0.375 (measured in the SASS).
Smaller blocks (512 threads) also let 2 blocks run on each SM at once.

### 4. 2D blocktiling
Each thread computes an 8x8 square of outputs. At each step it loads 8 values of A
and 8 values of B into registers and does all 64 combinations (an outer product),
so both A and B are reused. That is 4 multiply-adds per float read from shared
memory, above the 2 needed to keep the T4's 64 FMA units fed by 32 shared memory
banks. With 128x128 block tiles, global arithmetic intensity is 32 FLOPs/byte, above
the T4's ridge point: the first compute-side kernel.

### 5. Vectorized loads
Two changes to kernel 4: global loads use float4 (LDG.E.128), so each thread loads
its 8 floats per chunk in 2 instructions instead of 8; and As is stored transposed,
so each step's 8 values of A are contiguous in shared memory. Registers dropped from
124 to 102. On the T4 it gave no measurable speedup (see below).

## Things I measured along the way

- **Coalescing:** swapping two index lines made the same kernel ~10x slower.
- **Tile size:** a 16x16 tile (4 blocks per SM) was slower than 32x32 (1 block per SM).
  More blocks did not help; less data reuse hurt.
- **Clock throttling:** under heavy load the T4 hits its 70 W power limit and drops
  from 1590 MHz to ~1140 MHz, so its real FP32 peak is ~5.8 TFLOPS, not the 8.1 on
  the spec sheet.
- **Machine code (SASS):** shared memory reads per multiply-add went from 1.25
  (tiling) to 0.375 (1D blocktiling) to 0.0625 (2D blocktiling).
- **Low occupancy, fastest kernel:** 2D blocktiling uses 124 registers per thread,
  so only 2 blocks fit per SM (50% occupancy), yet it is the fastest. Each thread has
  64 independent multiply-adds per step, so latency is hidden through
  instruction-level parallelism instead of more warps.
- **Compute vs memory:** 2D blocktiling uses ~56% of the T4's real compute peak and
  ~32% of its memory bandwidth: compute-side, as its arithmetic intensity predicts.
- **An optimization that did nothing:** vectorized loads gave Boehm ~14% on an A6000,
  but nothing here. Registers fell from 124 to 102, still 2 blocks per SM (a third
  would need 85 or fewer); global loads were a tiny fraction of the work in a
  compute-side kernel; and shared memory loads stayed the same. Measure every change.

## Build and run

    make
    ./bin/sgemm

Requires the CUDA toolkit and an NVIDIA GPU. Set `-arch` in the Makefile for your GPU
(`sm_75` for a T4).

## Reference

Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
