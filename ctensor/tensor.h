#if !defined(TENSOR)
#define TENSOR

#include <stdint.h>

typedef struct
{
    float *data;
    uint32_t *shape;
    uint32_t *stride;
    uint32_t dims;
    uint32_t size;
} Tensor;

__device__ uint32_t get_index_in_tensor(Tensor *tensor, uint32_t *indices);
__global__ void matmul_tensor_kernel(Tensor *x, Tensor *y, Tensor *result);

#ifdef __cplusplus
extern "C"
{
#endif

    // Host function declarations
    Tensor *create_tensor(uint32_t *shape, uint32_t dims);
    void delete_tensor(Tensor *tensor);

    Tensor *create_device_tensor(uint32_t *shape, uint32_t dims);
    void delete_device_tensor(Tensor *tensor);

    uint32_t get_size_from_shape(uint32_t *shape, uint32_t dims);
    void fill_tensor(Tensor *x, float value);

    Tensor *to_device(Tensor *tensor);
    Tensor *to_host(Tensor *tensor);

    Tensor *add_tensor(Tensor *a, Tensor *b);
    Tensor *subtract_tensor(Tensor *a, Tensor *b);
    Tensor *multiply_tensor(Tensor *a, Tensor *b);
    Tensor *divide_tensor(Tensor *a, Tensor *b);
    Tensor *matmul_tensor(Tensor *a, Tensor *b);

#ifdef __cplusplus
}
#endif

#endif // TENSOR