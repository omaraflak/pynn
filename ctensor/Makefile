flags = -arch=sm_75

all: libtensor.so

libtensor.so: tensor.o cpu.o gpu.o iterator.o
	nvcc $(flags) -shared tensor.o cpu.o gpu.o iterator.o -o libtensor.so -O3

tensor.o: tensor.cu
	nvcc $(flags) -Xcompiler -fPIC -c tensor.cu -o tensor.o -O3

gpu.o: gpu.cu
	nvcc $(flags) -Xcompiler -fPIC -c gpu.cu -o gpu.o -O3

cpu.o: cpu.cu
	nvcc $(flags) -Xcompiler -fPIC -c cpu.cu -o cpu.o -O3

iterator.o: iterator.cu
	nvcc $(flags) -Xcompiler -fPIC -c iterator.cu -o iterator.o -O3
