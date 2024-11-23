all: libtensor.so

libtensor.so: tensor.o cpu.o
	gcc -shared tensor.o cpu.o -o libtensor.so

tensor.o: tensor.c
	gcc -c tensor.c -o tensor.o

cpu.o: cpu.c
	gcc -c cpu.c -o cpu.o

gpu.o: gpu.cu
	gcc -c gpu.cu -o gpu.o

# all: libtensor.so

# libtensor.so: tensor.o cpu.o gpu.o
# 	nvcc -shared tensor.o cpu.o gpu.o -o libtensor.so

# tensor.o: tensor.c
# 	nvcc -Xcompiler -fPIC -c tensor.c -o tensor.o

# cpu.o: cpu.c
# 	nvcc -Xcompiler -fPIC -c cpu.c -o cpu.o

# gpu.o: gpu.cu
# 	nvcc -Xcompiler -fPIC -c gpu.cu -o gpu.o