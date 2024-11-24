#if !defined(CPU)
#define CPU

#include "tensor.h"

void add_tensors_cpu(Tensor *a, Tensor *b, float *result);
void subtract_tensors_cpu(Tensor *a, Tensor *b, float *result);
void multiply_tensors_cpu(Tensor *a, Tensor *b, float *result);
void divide_tensors_cpu(Tensor *a, Tensor *b, float *result);
void matmul_tensors_cpu(Tensor *a, Tensor *b, float *result);

#endif // CPU
