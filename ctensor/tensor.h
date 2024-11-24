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

#ifdef __cplusplus
extern "C"
{
#endif
    void print_tensor_info(Tensor *tensor);

    Tensor *create_tensor(float *data, uint32_t *shape, uint32_t dims);
    Tensor *copy_tensor(Tensor *tensor);
    void delete_tensor(Tensor *tensor);

    void tensor_cpu_to_gpu(Tensor *tensor);
    void tensor_gpu_to_cpu(Tensor *tensor);

    void fill_tensor_data(Tensor *tensor, float value);
    void reshape_tensor(Tensor *tensor, uint32_t *shape, uint32_t dims);
    float get_tensor_item(Tensor *tensor, uint32_t *indices);

    Tensor *add_tensors(Tensor *a, Tensor *b);
    Tensor *matmul_tensors(Tensor *a, Tensor *b);

#ifdef __cplusplus
}
#endif

#endif // TENSOR