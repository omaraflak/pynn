#if !defined(TENSOR)
#define TENSOR

#include <stdlib.h>
#include <stdint.h>

typedef struct
{
    float *data;
    uint32_t *shape;
    uint32_t *stride;
    uint32_t dims;
    uint32_t size;
    uint32_t device;
} Tensor;

extern "C"
{
    void tensor_print_info(Tensor *tensor);

    Tensor *tensor_create(float *data, uint32_t *shape, uint32_t dims);
    Tensor *tensor_create_empty(uint32_t *shape, uint32_t dims);
    Tensor *tensor_copy(Tensor *tensor);
    void tensor_delete(Tensor *tensor);

    void tensor_cpu_to_gpu(Tensor *tensor);
    void tensor_gpu_to_cpu(Tensor *tensor);

    void tensor_fill(Tensor *tensor, float value);
    void tensor_fill_random_uniform(Tensor *tensor, float min, float max);
    void tensor_reshape(Tensor *tensor, uint32_t *shape, uint32_t dims);
    float tensor_get_item(Tensor *tensor, uint32_t *indices);

    Tensor *tensor_unary_minus(Tensor *a);

    Tensor *tensor_add(Tensor *a, Tensor *b);
    Tensor *tensor_subtract(Tensor *a, Tensor *b);
    Tensor *tensor_multiply(Tensor *a, Tensor *b);
    Tensor *tensor_divide(Tensor *a, Tensor *b);
    Tensor *tensor_matmul(Tensor *a, Tensor *b);

    Tensor *tensor_broadcast_add(Tensor *a, float value);
    Tensor *tensor_broadcast_subtract(Tensor *a, float value);
    Tensor *tensor_broadcast_multiply(Tensor *a, float value);
    Tensor *tensor_broadcast_divide(Tensor *a, float value);
    Tensor *tensor_broadcast_right_divide(Tensor *a, float value);
}

#endif // TENSOR