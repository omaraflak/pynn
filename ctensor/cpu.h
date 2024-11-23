#if !defined(CPU)
#define CPU

#include "tensor.h"

void fill_tensor_data_cpu(Tensor *tensor, float value);
void add_tensors_cpu(Tensor *a, Tensor *b, float *result);
void matmul_tensors_cpu(Tensor *a, Tensor *b, float *result);

#endif // CPU
