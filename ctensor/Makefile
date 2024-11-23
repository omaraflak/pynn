all: libtensor.so

libtensor.so: tensor.o cpu.o gpu.o
	gcc -shared tensor.o cpu.o -o libtensor.so

tensor.o: tensor.c
	gcc -c tensor.c -o tensor.o

cpu.o: cpu.c
	gcc -c cpu.c -o cpu.o

gpu.o: gpu.c
	nvcc -Xcompiler -fPIC -c gpu.cu -o gpu.o

# libtensor_gpu.so: tensor_gpu.o
# 	nvcc -shared tensor_gpu.o -o libtensor_gpu.so

# tensor_gpu.o: tensor_gpu.cu
# 	nvcc -Xcompiler -fPIC -c tensor_gpu.cu -o tensor_gpu.o
