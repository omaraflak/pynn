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
void fill_tensor(Tensor *x, float value);

Tensor *add_tensor(Tensor *a, Tensor *b);
Tensor *subtract_tensor(Tensor *a, Tensor *b);
Tensor *multiply_tensor(Tensor *a, Tensor *b);
Tensor *divide_tensor(Tensor *a, Tensor *b);
Tensor *matmul_tensor(Tensor *a, Tensor *b);

#endif // TENSOR