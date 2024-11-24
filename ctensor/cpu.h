#if !defined(CPU)
#define CPU

#include "tensor.h"

void fill_tensor_cpu(Tensor *a, float value);

void add_tensors_cpu(Tensor *a, Tensor *b, float *result);
void subtract_tensors_cpu(Tensor *a, Tensor *b, float *result);
void multiply_tensors_cpu(Tensor *a, Tensor *b, float *result);
void divide_tensors_cpu(Tensor *a, Tensor *b, float *result);
void matmul_tensors_cpu(Tensor *a, Tensor *b, float *result);

void broadcast_add_tensor_cpu(Tensor *a, float value, float *result);
void broadcast_subtract_tensor_cpu(Tensor *a, float value, float *result);
void broadcast_multiply_tensor_cpu(Tensor *a, float value, float *result);
void broadcast_divide_tensor_cpu(Tensor *a, float value, float *result);

#endif // CPU
