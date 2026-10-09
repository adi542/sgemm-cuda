NVCC = nvcc
FLAGS = -O3 -arch=sm_75

all: bin/vector_add bin/sgemm bin/device_info

bin/vector_add: warmup/vector_add.cu
	mkdir -p bin
	$(NVCC) $(FLAGS) -o bin/vector_add warmup/vector_add.cu

bin/sgemm: src/main.cu src/kernels/*.cuh
	mkdir -p bin
	$(NVCC) $(FLAGS) -o bin/sgemm src/main.cu -lcublas
bin/device_info: warmup/device_info.cu
	mkdir -p bin
	$(NVCC) $(FLAGS) -o bin/device_info warmup/device_info.cu

clean:
	rm -rf bin

.PHONY: all clean
