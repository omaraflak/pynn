#include "iterator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Iterator *iterator_create(Tensor *tensor)
{
    Iterator *iterator = (Iterator *)malloc(sizeof(Iterator));
    iterator->indices = (int32_t *)malloc(sizeof(int32_t) * tensor->dims);
    iterator->index = tensor->dims - 1;
    iterator->value = 0;
    iterator->first = true;
    memcpy(iterator->indices, tensor->shape, sizeof(int32_t) * tensor->dims);
    return iterator;
}

Iterator *iterator_create_device(Tensor *tensor)
{
    Iterator *device;
    int32_t *indices;

    cudaMalloc(&device, sizeof(Iterator));
    cudaMalloc(&indices, sizeof(int32_t) * tensor->dims);

    Iterator host;
    host.indices = indices;
    host.index = tensor->dims - 1;
    host.value = 0;
    host.first = true;

    cudaMemcpy(device, &host, sizeof(Iterator), cudaMemcpyHostToDevice);
    cudaMemcpy(indices, tensor->shape, sizeof(int32_t) * tensor->dims, cudaMemcpyHostToDevice);

    return device;
}

void iterator_free(Iterator *iterator)
{
    free(iterator->indices);
    free(iterator);
}

void iterator_free_device(Iterator *iterator)
{
    cudaFree(iterator->indices);
    cudaFree(iterator);
}

__device__ __host__ bool iterator_next(Tensor *tensor, Iterator *iterator)
{
    if (iterator->first)
    {
        for (int32_t i = 0; i < tensor->dims; i++)
        {
            iterator->value += tensor->offset[i];
        }
        iterator->first = false;
        return true;
    }

    for (int32_t i = iterator->index; i >= 0; i--)
    {
        if (iterator->indices[i] == 1)
        {
            iterator->index -= 1;
            continue;
        }

        iterator->indices[i] -= 1;
        iterator->value += tensor->stride[i];

        for (int32_t j = i + 1; j < tensor->dims; j++)
        {
            iterator->value -= (tensor->shape[j] - iterator->indices[j]) * tensor->stride[j];
            iterator->indices[j] = tensor->shape[j];
        }
        iterator->index = tensor->dims - 1;
        break;
    }

    if (iterator->index < 0)
    {
        iterator->index = 0;
        return false;
    }

    return true;
}

__device__ __host__ void get_indices(Tensor *tensor, int32_t *indices, int32_t index)
{
    int32_t rest = index;
    for (int32_t i = 0; i < tensor->dims; i++)
    {
        indices[i] = (int32_t)rest / tensor->stride[i];
        rest %= tensor->stride[i];
    }
}

__device__ __host__ int32_t get_index(Tensor *tensor, int32_t *indices)
{
    int32_t index = 0;
    for (int32_t i = 0; i < tensor->dims; i++)
    {
        index += tensor->offset[i] + tensor->stride[i] * indices[i];
    }
    return index;
}

__device__ __host__ int32_t get_index(Tensor *tensor, int32_t index)
{
    int32_t indices[tensor->dims];
    get_indices(tensor, indices, index);
    return get_index(tensor, indices);
}