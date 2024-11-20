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
} Tensor;

Tensor *create_tensor(uint32_t *shape, uint32_t dims);
void delete_tensor(Tensor *tensor);

uint32_t get_size_from_shape(uint32_t *shape, uint32_t dims);
void fill_tensor(Tensor *tensor, float value);
void set_tensor(Tensor *tensor, float *data);

void reshape_tensor(Tensor *tensor, uint32_t *shape, uint32_t dims);
float get_tensor_item(Tensor *tensor, uint32_t *indices);

Tensor *add_tensors(Tensor *a, Tensor *b);
Tensor *subtract_tensors(Tensor *a, Tensor *b);
Tensor *multiply_tensors(Tensor *a, Tensor *b);
Tensor *divide_tensors(Tensor *a, Tensor *b);
Tensor *matmul_tensors(Tensor *a, Tensor *b);

#endif // TENSOR