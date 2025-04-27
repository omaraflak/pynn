#if !defined(ITERATOR)
#define ITERATOR

#include "_tensor.h"

int32_t get_index(Tensor *tensor, int32_t index);
int32_t get_index(Tensor *tensor, int32_t *indices);

__host__ __device__
int32_t get_index(
    int32_t *shape,
    int32_t* stride,
    int32_t offset,
    int32_t dims,
    bool has_base,
    int32_t index
);

#endif // ITERATOR