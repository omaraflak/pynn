all: libtensor_gpu.so libtensor.so

libtensor_gpu.so: tensor_gpu.o
	nvcc -shared tensor_gpu.o -o libtensor_gpu.so

tensor_gpu.o: tensor_gpu.cu
	nvcc -Xcompiler -fPIC -c tensor_gpu.cu -o tensor_gpu.o

libtensor.so: tensor.o
	gcc -shared tensor.o -o libtensor.so

tensor.o: tensor.c
	gcc -c tensor.c -o tensor.o
