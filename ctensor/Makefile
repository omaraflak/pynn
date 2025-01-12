all: libtensor.so

libtensor.so: tensor.o cpu.o gpu.o iterator.o
	nvcc -shared tensor.o cpu.o gpu.o iterator.o -o libtensor.so

tensor.o: tensor.cu
	nvcc -Xcompiler -fPIC -c tensor.cu -o tensor.o

gpu.o: gpu.cu
	nvcc -Xcompiler -fPIC -c gpu.cu -o gpu.o

cpu.o: cpu.cu
	nvcc -Xcompiler -fPIC -c cpu.cu -o cpu.o

iterator.o: iterator.cu
	nvcc -Xcompiler -fPIC -c iterator.cu -o iterator.o
