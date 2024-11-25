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
    uint32_t device;
} Tensor;

extern "C"
{
    Tensor *tensor_create(float *data, uint32_t *shape, uint32_t dims);
    Tensor *tensor_create_empty(uint32_t *shape, uint32_t dims);
    Tensor *tensor_copy(Tensor *tensor);
    void tensor_delete(Tensor *tensor);

    void tensor_cpu_to_gpu(Tensor *tensor);
    void tensor_gpu_to_cpu(Tensor *tensor);

    void tensor_fill(Tensor *tensor, float value);
    void tensor_fill_random_uniform(Tensor *tensor, float min, float max);
    void tensor_fill_random_normal(Tensor *tensor, float mean, float std);
    void tensor_fill_identity(Tensor *tensor);
    void tensor_reshape(Tensor *tensor, uint32_t *shape, uint32_t dims);

    float tensor_get_item(Tensor *tensor, uint32_t *indices);
    void tensor_set_item(Tensor *tensor, uint32_t *indices, float value);

    float tensor_sum(Tensor *tensor);
    float tensor_mean(Tensor *tensor);
    float tensor_min(Tensor *tensor);
    float tensor_max(Tensor *tensor);

    Tensor *tensor_unary_minus(Tensor *tensor);
    Tensor *tensor_transpose(Tensor *tensor);

    void tensor_add_into(Tensor *a, Tensor *b);
    void tensor_subtract_into(Tensor *a, Tensor *b);
    void tensor_multiply_into(Tensor *a, Tensor *b);
    void tensor_divide_into(Tensor *a, Tensor *b);

    Tensor *tensor_add(Tensor *a, Tensor *b);
    Tensor *tensor_subtract(Tensor *a, Tensor *b);
    Tensor *tensor_multiply(Tensor *a, Tensor *b);
    Tensor *tensor_divide(Tensor *a, Tensor *b);
    Tensor *tensor_matmul(Tensor *a, Tensor *b);

    Tensor *tensor_broadcast_add(Tensor *tensor, float value);
    Tensor *tensor_broadcast_subtract(Tensor *tensor, float value);
    Tensor *tensor_broadcast_multiply(Tensor *tensor, float value);
    Tensor *tensor_broadcast_divide(Tensor *tensor, float value);
    Tensor *tensor_broadcast_right_divide(Tensor *tensor, float value);

    Tensor *tensor_power(Tensor *tensor, float power);
    Tensor *tensor_exp(Tensor *tensor);
    Tensor *tensor_log(Tensor *tensor);
    Tensor *tensor_log10(Tensor *tensor);
    Tensor *tensor_logb(Tensor *tensor, float base);
    Tensor *tensor_sin(Tensor *tensor);
    Tensor *tensor_cos(Tensor *tensor);
    Tensor *tensor_tanh(Tensor *tensor);
}

#endif // TENSOR