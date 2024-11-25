#if !defined(GPU)
#define GPU

#include "tensor.h"

__global__ void tensor_fill_kernel(float *a, uint32_t n, float value);
void tensor_fill_gpu(Tensor *a, float value);

__global__ void tensor_fill_random_uniform_kernel(float *a, uint32_t n, float min, float max);
void tensor_fill_random_uniform_gpu(Tensor *a, float min, float max);

__global__ void tensor_fill_random_normal_kernel(float *a, uint32_t n, float mean, float std);
void tensor_fill_random_normal_gpu(Tensor *a, float mean, float std);

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

__global__ void tensor_broadcast_add_kernel(float *a, uint32_t n, float value, float *result);
void tensor_broadcast_add_gpu(Tensor *a, float value, float *result);

__global__ void tensor_broadcast_subtract_kernel(float *a, uint32_t n, float value, float *result);
void tensor_broadcast_subtract_gpu(Tensor *a, float value, float *result);

__global__ void tensor_broadcast_multiply_kernel(float *a, uint32_t n, float value, float *result);
void tensor_broadcast_multiply_gpu(Tensor *a, float value, float *result);

__global__ void tensor_broadcast_divide_kernel(float *a, uint32_t n, float value, float *result);
void tensor_broadcast_divide_gpu(Tensor *a, float value, float *result);

__global__ void tensor_broadcast_right_divide_kernel(float *a, uint32_t n, float value, float *result);
void tensor_broadcast_right_divide_gpu(Tensor *a, float value, float *result);

__global__ void tensor_power_kernel(float *a, uint32_t n, float power, float *result);
void tensor_power_gpu(Tensor *a, float power, float *result);

__global__ void tensor_exp_kernel(float *a, uint32_t n, float *result);
void tensor_exp_gpu(Tensor *a, float *result);

__global__ void tensor_log_kernel(float *a, uint32_t n, float *result);
void tensor_log_gpu(Tensor *a, float *result);

__global__ void tensor_log10_kernel(float *a, uint32_t n, float *result);
void tensor_log10_gpu(Tensor *a, float *result);

__global__ void tensor_logb_kernel(float *a, uint32_t n, float base, float *result);
void tensor_logb_gpu(Tensor *a, float base, float *result);

__global__ void tensor_sin_kernel(float *a, uint32_t n, float *result);
void tensor_sin_gpu(Tensor *a, float *result);

__global__ void tensor_cos_kernel(float *a, uint32_t n, float *result);
void tensor_cos_gpu(Tensor *a, float *result);

__global__ void tensor_tanh_kernel(float *a, uint32_t n, float *result);
void tensor_tanh_gpu(Tensor *a, float *result);

#endif // GPU