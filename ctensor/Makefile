all: libtensor.so

libtensor.so: tensor.o
	nvcc -shared tensor.o -o libtensor.so

tensor.o: tensor.cu
	nvcc -Xcompiler -fPIC -c tensor.cu -o tensor.o
