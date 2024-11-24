#if !defined(GPU)
#define GPU

#include "tensor.h"

__global__ void fill_tensor_kernel(float *a, uint32_t n, float value);
void fill_tensor_gpu(Tensor *a, float value);

__global__ void unary_minus_tensor_kernel(float *a, uint32_t n, float *result);
void unary_minus_tensor_gpu(Tensor *a, float *result);

__global__ void add_tensors_kernel(float *a, float *b, uint32_t n, float *result);
void add_tensors_gpu(Tensor *a, Tensor *b, float *result);

__global__ void subtract_tensors_kernel(float *a, float *b, uint32_t n, float *result);
void subtract_tensors_gpu(Tensor *a, Tensor *b, float *result);

__global__ void multiply_tensors_kernel(float *a, float *b, uint32_t n, float *result);
void multiply_tensors_gpu(Tensor *a, Tensor *b, float *result);

__global__ void divide_tensors_kernel(float *a, float *b, uint32_t n, float *result);
void divide_tensors_gpu(Tensor *a, Tensor *b, float *result);

__global__ void matmul_tensors_kernel(float *a, float *b, uint32_t m, uint32_t p, uint32_t n, float *result);
void matmul_tensors_gpu(Tensor *a, Tensor *b, float *result);

__global__ void broadcast_add_tensor_kernel(float *a, float value, uint32_t n, float *result);
void broadcast_add_tensor_gpu(Tensor *a, float value, float *result);

__global__ void broadcast_subtract_tensor_kernel(float *a, float value, uint32_t n, float *result);
void broadcast_subtract_tensor_gpu(Tensor *a, float value, float *result);

__global__ void broadcast_multiply_tensor_kernel(float *a, float value, uint32_t n, float *result);
void broadcast_multiply_tensor_gpu(Tensor *a, float value, float *result);

__global__ void broadcast_divide_tensor_kernel(float *a, float value, uint32_t n, float *result);
void broadcast_divide_tensor_gpu(Tensor *a, float value, float *result);

__global__ void broadcast_right_divide_tensor_kernel(float *a, float value, uint32_t n, float *result);
void broadcast_right_divide_tensor_gpu(Tensor *a, float value, float *result);

#endif // GPU