#if !defined(GPU)
#define GPU

#include "tensor.h"

__global__ void add_tensors_kernel(float *a, float *b, uint32_t n, float *result);
__global__ void matmul_tensors_kernel(float *a, float *b, uint32_t m, uint32_t p, uint32_t n, float *result);

#ifdef __cplusplus
extern "C"
{
#endif

    void add_tensors_gpu(Tensor *a, Tensor *b, float *result);
    void matmul_tensors_gpu(Tensor *a, Tensor *b, float *result);

#ifdef __cplusplus
}
#endif

#endif // GPU