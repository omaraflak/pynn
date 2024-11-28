#if !defined(CPU)
#define CPU

#include "tensor.h"

void tensor_fill_cpu(Tensor *a, float value);
void tensor_fill_random_uniform_cpu(Tensor *a, float min, float max);
void tensor_fill_random_normal_cpu(Tensor *a, float mean, float std);
void tensor_fill_identity_cpu(Tensor *a);

void tensor_unary_minus_cpu(Tensor *a, float *result);

void tensor_add_cpu(Tensor *a, Tensor *b, float *result);
void tensor_subtract_cpu(Tensor *a, Tensor *b, float *result);
void tensor_multiply_cpu(Tensor *a, Tensor *b, float *result);
void tensor_divide_cpu(Tensor *a, Tensor *b, float *result);
void tensor_matmul_cpu(Tensor *a, Tensor *b, int32_t batch, float *result);

void tensor_broadcast_add_cpu(Tensor *a, float value, float *result);
void tensor_broadcast_subtract_cpu(Tensor *a, float value, float *result);
void tensor_broadcast_multiply_cpu(Tensor *a, float value, float *result);
void tensor_broadcast_divide_cpu(Tensor *a, float value, float *result);
void tensor_broadcast_right_divide_cpu(Tensor *a, float value, float *result);

void tensor_power_cpu(Tensor *a, float power, float *result);
void tensor_exp_cpu(Tensor *a, float *result);
void tensor_log_cpu(Tensor *a, float *result);
void tensor_log10_cpu(Tensor *a, float *result);
void tensor_logb_cpu(Tensor *a, float base, float *result);
void tensor_sin_cpu(Tensor *a, float *result);
void tensor_cos_cpu(Tensor *a, float *result);
void tensor_tanh_cpu(Tensor *a, float *result);

float tensor_sum_cpu(Tensor *a);
float tensor_mean_cpu(Tensor *a);
float tensor_min_cpu(Tensor *a);
float tensor_max_cpu(Tensor *a);

#endif // CPU
