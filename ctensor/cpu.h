#if !defined(CPU)
#define CPU

#include "tensor.h"

void tensor_fill_cpu(Tensor *a, float value);
void tensor_unary_minus_cpu(Tensor *a, float *result);

void tensor_add_cpu(Tensor *a, Tensor *b, float *result);
void tensor_subtract_cpu(Tensor *a, Tensor *b, float *result);
void tensor_multiply_cpu(Tensor *a, Tensor *b, float *result);
void tensor_divide_cpu(Tensor *a, Tensor *b, float *result);
void tensor_matmul_cpu(Tensor *a, Tensor *b, float *result);

void tensor_broadcast_add_cpu(Tensor *a, float value, float *result);
void tensor_broadcast_subtract_cpu(Tensor *a, float value, float *result);
void tensor_broadcast_multiply_cpu(Tensor *a, float value, float *result);
void tensor_broadcast_divide_cpu(Tensor *a, float value, float *result);
void tensor_broadcast_right_divide_cpu(Tensor *a, float value, float *result);

#endif // CPU
