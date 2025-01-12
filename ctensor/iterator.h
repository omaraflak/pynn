#if !defined(ITERATOR)
#define ITERATOR

#include <stdint.h>
#include "tensor.h"

struct Iterator
{
    int32_t *indices;
    int32_t index;
    int32_t value;
    bool first;
};

Iterator *iterator_create(Tensor *tensor);
Iterator *iterator_create_device(Tensor *tensor);
void iterator_free(Iterator *iterator);
void iterator_free_device(Iterator *iterator);

__device__ __host__ bool iterator_next(Tensor *tensor, Iterator *iterator);
__device__ __host__ void get_indices(Tensor *tensor, int32_t *indices, int32_t index);
__device__ __host__ int32_t get_index(Tensor *tensor, int32_t *indices);
__device__ __host__ int32_t get_index(Tensor *tensor, int32_t index);

#endif // ITERATOR
