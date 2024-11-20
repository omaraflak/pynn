#if !defined(TENSOR_GPU)
#define TENSOR_GPU

#include "tensor.h"

__device__ uint32_t get_index_in_tensor(Tensor *tensor, uint32_t *indices);
__global__ void matmul_tensor_kernel(Tensor *x, Tensor *y, Tensor *result);

#ifdef __cplusplus
extern "C"
{
#endif
    Tensor *create_device_tensor(uint32_t *shape, uint32_t dims);
    void delete_device_tensor(Tensor *tensor);

    Tensor *to_device(Tensor *tensor);
    Tensor *to_host(Tensor *tensor);

    Tensor *matmul_tensor_gpu(Tensor *tensor1, Tensor *tensor2);
#ifdef __cplusplus
}
#endif

#endif // TENSOR_GPU