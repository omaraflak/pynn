#if !defined(GPU)
#define GPU

#include "tensor.h"

void delete_device_tensor(Tensor *tensor);
void host_to_device(Tensor *tensor);
void device_to_host(Tensor *tensor);

__global__ void add_tensors_kernel(float *a, float *b, uint32_t n, float *result);
void add_tensors_gpu(Tensor *a, Tensor *b, float *result);

__global__ void matmul_tensors_kernel(float *a, float *b, uint32_t m, uint32_t p, uint32_t n, float *result);
void matmul_tensors_gpu(Tensor *a, Tensor *b, float *result);

#endif // GPU