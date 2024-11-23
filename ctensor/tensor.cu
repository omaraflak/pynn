#include "tensor.h"
#include "cpu.h"
#include "gpu.h"
#include <cstring>

uint32_t get_size_from_shape(uint32_t *shape, uint32_t dims)
{
    uint32_t size = 1;
    for (uint32_t i = 0; i < dims; i++)
    {
        size *= shape[i];
    }
    return size;
}

Tensor *create_tensor(float *data, uint32_t *shape, uint32_t dims)
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->data = data;
    tensor->shape = shape;
    tensor->stride = (uint32_t *)malloc(sizeof(uint32_t) * dims);
    tensor->size = get_size_from_shape(shape, dims);
    tensor->dims = dims;
    tensor->device = 0;
    for (uint32_t i = 0; i < dims; i++)
    {
        tensor->stride[i] = 1;
        for (uint32_t j = i + 1; j < dims; j++)
        {
            tensor->stride[i] *= shape[j];
        }
    }
    return tensor;
}

void delete_tensor(Tensor *tensor)
{
    free(tensor->data);
    free(tensor->shape);
    free(tensor->stride);
    free(tensor);
}

Tensor *copy_tensor(Tensor *tensor)
{
    float *data = (float *)malloc(sizeof(float) * tensor->size);
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * tensor->dims);
    memcpy(data, tensor->data, sizeof(float) * tensor->size);
    memcpy(shape, tensor->shape, sizeof(uint32_t) * tensor->dims);
    return create_tensor(data, shape, tensor->dims);
}

void fill_tensor_data(Tensor *tensor, float value)
{
    for (uint32_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = value;
    }
}

void reshape_tensor(Tensor *tensor, uint32_t *shape, uint32_t dims)
{
    if (tensor->dims != dims)
    {
        free(tensor->shape);
        free(tensor->stride);
        tensor->dims = dims;
        tensor->shape = (uint32_t *)malloc(sizeof(uint32_t) * dims);
        tensor->stride = (uint32_t *)malloc(sizeof(uint32_t) * dims);
    }
    for (uint32_t i = 0; i < dims; i++)
    {
        tensor->shape[i] = shape[i];
    }
    for (uint32_t i = 0; i < dims; i++)
    {
        tensor->stride[i] = 1;
        for (uint32_t j = i + 1; j < dims; j++)
        {
            tensor->stride[i] *= shape[j];
        }
    }
}

float get_tensor_item(Tensor *tensor, uint32_t *indices)
{
    uint32_t index = 0;
    for (uint32_t i = 0; i < tensor->dims; i++)
    {
        index += tensor->stride[i] * indices[i];
    }
    return tensor->data[index];
}

Tensor *add_tensors(Tensor *a, Tensor *b)
{
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * a->dims);
    memcpy(shape, a->shape, sizeof(uint32_t) * a->dims);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        add_tensors_cpu(a, b, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        add_tensors_gpu(a, b, data);
    }

    return create_tensor(data, shape, a->dims);
}

Tensor *matmul_tensors(Tensor *a, Tensor *b)
{
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * 2);
    shape[0] = a->shape[0];
    shape[1] = b->shape[1];
    uint32_t size = shape[0] * shape[1];
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * size);
        matmul_tensors_cpu(a, b, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * size);
        matmul_tensors_gpu(a, b, data);
    }

    return create_tensor(data, shape, 2);
}
