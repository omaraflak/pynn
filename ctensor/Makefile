flags = -arch=sm_75

all: libtensor.so

libtensor.so: tensor.o cpu.o gpu.o
	nvcc $(flags) -shared tensor.o cpu.o gpu.o -o libtensor.so

tensor.o: tensor.cu
	nvcc $(flags) -Xcompiler -fPIC -c tensor.cu -o tensor.o

gpu.o: gpu.cu
	nvcc $(flags) -Xcompiler -fPIC -c gpu.cu -o gpu.o

cpu.o: cpu.cu
	nvcc $(flags) -Xcompiler -fPIC -c cpu.cu -o cpu.o
