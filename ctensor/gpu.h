#if !defined(GPU)
#define GPU

#include "tensor.h"

__global__ void tensor_fill_kernel(float *a, uint32_t n, float value);
void tensor_fill_gpu(Tensor *a, float value);

__global__ void tensor_unary_minus_kernel(float *a, uint32_t n, float *result);
void tensor_unary_minus_gpu(Tensor *a, float *result);

__global__ void tensor_add_kernel(float *a, float *b, uint32_t n, float *result);
void tensor_add_gpu(Tensor *a, Tensor *b, float *result);

__global__ void tensor_subtract_kernel(float *a, float *b, uint32_t n, float *result);
void tensor_subtract_gpu(Tensor *a, Tensor *b, float *result);

__global__ void tensor_multiply_kernel(float *a, float *b, uint32_t n, float *result);
void tensor_multiply_gpu(Tensor *a, Tensor *b, float *result);

__global__ void tensor_divide_kernel(float *a, float *b, uint32_t n, float *result);
void tensor_divide_gpu(Tensor *a, Tensor *b, float *result);

__global__ void tensor_matmul_kernel(float *a, float *b, uint32_t m, uint32_t p, uint32_t n, float *result);
void tensor_matmul_gpu(Tensor *a, Tensor *b, float *result);

__global__ void tensor_broadcast_add_kernel(float *a, float value, uint32_t n, float *result);
void tensor_broadcast_add_gpu(Tensor *a, float value, float *result);

__global__ void tensor_broadcast_subtract_kernel(float *a, float value, uint32_t n, float *result);
void tensor_broadcast_subtract_gpu(Tensor *a, float value, float *result);

__global__ void tensor_broadcast_multiply_kernel(float *a, float value, uint32_t n, float *result);
void tensor_broadcast_multiply_gpu(Tensor *a, float value, float *result);

__global__ void tensor_broadcast_divide_kernel(float *a, float value, uint32_t n, float *result);
void tensor_broadcast_divide_gpu(Tensor *a, float value, float *result);

__global__ void tensor_broadcast_right_divide_kernel(float *a, float value, uint32_t n, float *result);
void tensor_broadcast_right_divide_gpu(Tensor *a, float value, float *result);

#endif // GPU