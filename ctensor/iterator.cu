#include "iterator.h"


int32_t get_index(Tensor *tensor, int32_t index)
{
    return get_index(
        tensor->shape,
        tensor->stride,
        tensor->offset,
        tensor->dims,
        tensor->base,
        index
    );
}

int32_t get_index(Tensor *tensor, int32_t *indices)
{
    int32_t index = tensor->offset;
    for (int32_t i = 0; i < tensor->dims; i++)
    {
        index += indices[i] * tensor->stride[i];
    }
    return index;
}

__host__ __device__
int32_t get_index(
    int32_t *shape,
    int32_t* stride,
    int32_t offset,
    int32_t dims,
    bool has_base,
    int32_t index
) {
    if (!has_base)
    {
        return index;
    }

    int32_t remaining = index;
    int32_t base_index = offset;

    for (int32_t i=dims-1; i>=0; i--) {
        int32_t dim = remaining % shape[i];
        base_index += dim * stride[i];
        remaining /= shape[i];
    }

    return base_index;
}