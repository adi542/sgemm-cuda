NVCC = nvcc
FLAGS = -O3 -arch=sm_75
all: bin/vector_add
bin/vector_add: warmup/vector_add.cu
	mkdir -p bin
	$(NVCC) $(FLAGS) -o bin/vector_add warmup/vector_add.cu
clean:
	rm -rf bin
.PHONY: all clean
