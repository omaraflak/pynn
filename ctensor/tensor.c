#include "tensor.h"

Tensor *create_tensor(uint32_t *shape, uint32_t dims)
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->dims = dims;
    tensor->size = get_size_from_shape(shape, dims);
    tensor->data = (float *)malloc(sizeof(float) * tensor->size);
    tensor->shape = (uint32_t *)malloc(sizeof(uint32_t) * dims);
    tensor->stride = (uint32_t *)malloc(sizeof(uint32_t) * dims);
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
    return tensor;
}

void delete_tensor(Tensor *tensor)
{
    free(tensor->data);
    free(tensor->shape);
    free(tensor->stride);
    free(tensor);
}

uint32_t get_size_from_shape(uint32_t *shape, uint32_t dims)
{
    uint32_t size = 1;
    for (uint32_t i = 0; i < dims; i++)
    {
        size *= shape[i];
    }
    return size;
}

void fill_tensor(Tensor *tensor, float value)
{
    for (uint32_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = value;
    }
}

void set_tensor(Tensor *tensor, float *data)
{
    for (uint32_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = data[i];
    }
}

void reshape_tensor(Tensor *tensor, uint32_t *shape, uint32_t dims)
{
    if (tensor->dims != dims)
    {
        free(tensor->shape);
        free(tensor->stride);
        tensor->dims = dims;
        tensor->shape = malloc(sizeof(uint32_t) * dims);
        tensor->stride = malloc(sizeof(uint32_t) * dims);
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
    Tensor *result = create_tensor(a->shape, a->dims);
    for (uint32_t i = 0; i < result->size; i++)
    {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

Tensor *subtract_tensors(Tensor *a, Tensor *b)
{
    Tensor *result = create_tensor(a->shape, a->dims);
    for (uint32_t i = 0; i < result->size; i++)
    {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

Tensor *multiply_tensors(Tensor *a, Tensor *b)
{
    Tensor *result = create_tensor(a->shape, a->dims);
    for (uint32_t i = 0; i < result->size; i++)
    {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

Tensor *divide_tensors(Tensor *a, Tensor *b)
{
    Tensor *result = create_tensor(a->shape, a->dims);
    for (uint32_t i = 0; i < result->size; i++)
    {
        result->data[i] = a->data[i] / b->data[i];
    }
    return result;
}

Tensor *matmul_tensors(Tensor *a, Tensor *b)
{
    uint32_t shape[2] = {a->shape[0], b->shape[1]};
    Tensor *result = create_tensor(shape, 2);
    for (int i = 0; i < shape[0]; i++)
    {
        for (int j = 0; j < shape[1]; j++)
        {
            float tmp = 0;
            for (int k = 0; k < b->shape[0]; k++)
            {
                tmp += a->data[i * a->shape[0] + k] * b->data[k * b->shape[0] + j];
            }
            result->data[i * shape[0] + j] = tmp;
        }
    }
    return result;
}