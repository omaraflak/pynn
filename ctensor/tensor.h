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
    void print_tensor_info(Tensor *tensor);

    Tensor *create_tensor(float *data, uint32_t *shape, uint32_t dims);
    Tensor *copy_tensor(Tensor *tensor);
    void delete_tensor(Tensor *tensor);

    void tensor_cpu_to_gpu(Tensor *tensor);
    void tensor_gpu_to_cpu(Tensor *tensor);

    void fill_tensor(Tensor *tensor, float value);
    void reshape_tensor(Tensor *tensor, uint32_t *shape, uint32_t dims);
    float get_tensor_item(Tensor *tensor, uint32_t *indices);

    Tensor *unary_minus_tensor(Tensor *a);

    Tensor *add_tensors(Tensor *a, Tensor *b);
    Tensor *subtract_tensors(Tensor *a, Tensor *b);
    Tensor *multiply_tensors(Tensor *a, Tensor *b);
    Tensor *divide_tensors(Tensor *a, Tensor *b);
    Tensor *matmul_tensors(Tensor *a, Tensor *b);

    Tensor *broadcast_add_tensor(Tensor *a, float value);
    Tensor *broadcast_subtract_tensor(Tensor *a, float value);
    Tensor *broadcast_multiply_tensor(Tensor *a, float value);
    Tensor *broadcast_divide_tensor(Tensor *a, float value);
    Tensor *broadcast_right_divide_tensor(Tensor *a, float value);
}

#endif // TENSOR