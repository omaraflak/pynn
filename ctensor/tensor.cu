#include "tensor.h"
#include "cpu.h"
#include "gpu.h"
#include <cstring>
#include <stdio.h>

uint32_t _get_size_from_shape(uint32_t *shape, uint32_t dims)
{
    uint32_t size = 1;
    for (uint32_t i = 0; i < dims; i++)
    {
        size *= shape[i];
    }
    return size;
}

Tensor *_create_tensor(float *data, uint32_t *shape, uint32_t dims, uint32_t device)
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->data = data;
    tensor->shape = shape;
    tensor->stride = (uint32_t *)malloc(sizeof(uint32_t) * dims);
    tensor->size = _get_size_from_shape(shape, dims);
    tensor->dims = dims;
    tensor->device = device;
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

void print_tensor_info(Tensor *tensor)
{
    printf("Tensor shape: (");
    for (uint32_t i = 0; i < tensor->dims; i++)
    {
        const char *format = i == tensor->dims - 1 ? "%d" : "%d, ";
        printf(format, tensor->shape[i]);
    }
    printf(")\\n");
    printf("Tensor stride: (");
    for (uint32_t i = 0; i < tensor->dims; i++)
    {
        const char *format = i == tensor->dims - 1 ? "%d" : "%d, ";
        printf(format, tensor->stride[i]);
    }
    printf(")\\n");
    printf("Tensor size: %d\\n", tensor->size);
    printf("Tensor device: %d\\n", tensor->device);
}

Tensor *create_tensor(float *data, uint32_t *shape, uint32_t dims)
{
    uint32_t size = _get_size_from_shape(shape, dims);
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->data = (float *)malloc(sizeof(float) * size);
    tensor->shape = (uint32_t *)malloc(sizeof(uint32_t) * dims);
    tensor->stride = (uint32_t *)malloc(sizeof(uint32_t) * dims);
    tensor->size = size;
    tensor->dims = dims;
    tensor->device = 0;
    for (uint32_t i = 0; i < size; i++)
    {
        tensor->data[i] = data[i];
    }
    for (uint32_t i = 0; i < dims; i++)
    {
        tensor->shape[i] = shape[i];
        tensor->stride[i] = 1;
        for (uint32_t j = i + 1; j < dims; j++)
        {
            tensor->stride[i] *= shape[j];
        }
    }
    return tensor;
}

Tensor *copy_tensor(Tensor *tensor)
{
    float *data = (float *)malloc(sizeof(float) * tensor->size);
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * tensor->dims);
    memcpy(data, tensor->data, sizeof(float) * tensor->size);
    memcpy(shape, tensor->shape, sizeof(uint32_t) * tensor->dims);
    return _create_tensor(data, shape, tensor->dims, /* device=*/0);
}

void delete_tensor(Tensor *tensor)
{
    if (tensor->device == 0)
    {
        free(tensor->data);
    }
    else
    {
        cudaFree(tensor->data);
    }
    free(tensor->shape);
    free(tensor->stride);
    free(tensor);
}

void tensor_cpu_to_gpu(Tensor *tensor)
{
    float *data;
    cudaMalloc(&data, sizeof(float) * tensor->size);
    cudaMemcpy(data, tensor->data, sizeof(float) * tensor->size, cudaMemcpyHostToDevice);
    free(tensor->data);
    tensor->data = data;
    tensor->device = 1;
}

void tensor_gpu_to_cpu(Tensor *tensor)
{
    float *data = (float *)malloc(sizeof(float) * tensor->size);
    cudaMemcpy(data, tensor->data, sizeof(float) * tensor->size, cudaMemcpyDeviceToHost);
    cudaFree(tensor->data);
    tensor->data = data;
    tensor->device = 0;
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
        return _create_tensor(data, shape, a->dims, /* device=*/0);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        add_tensors_gpu(a, b, data);
        return _create_tensor(data, shape, a->dims, /* device=*/1);
    }
}

Tensor *subtract_tensors(Tensor *a, Tensor *b)
{
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * a->dims);
    memcpy(shape, a->shape, sizeof(uint32_t) * a->dims);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        subtract_tensors_cpu(a, b, data);
        return _create_tensor(data, shape, a->dims, /* device=*/0);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        subtract_tensors_gpu(a, b, data);
        return _create_tensor(data, shape, a->dims, /* device=*/1);
    }
}

Tensor *multiply_tensors(Tensor *a, Tensor *b)
{
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * a->dims);
    memcpy(shape, a->shape, sizeof(uint32_t) * a->dims);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        multiply_tensors_cpu(a, b, data);
        return _create_tensor(data, shape, a->dims, /* device=*/0);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        multiply_tensors_gpu(a, b, data);
        return _create_tensor(data, shape, a->dims, /* device=*/1);
    }
}

Tensor *divide_tensors(Tensor *a, Tensor *b)
{
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * a->dims);
    memcpy(shape, a->shape, sizeof(uint32_t) * a->dims);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        divide_tensors_cpu(a, b, data);
        return _create_tensor(data, shape, a->dims, /* device=*/0);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        divide_tensors_gpu(a, b, data);
        return _create_tensor(data, shape, a->dims, /* device=*/1);
    }
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
        return _create_tensor(data, shape, /* dims=*/2, /* device=*/0);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * size);
        matmul_tensors_gpu(a, b, data);
        return _create_tensor(data, shape, /* dims=*/2, /* device=*/1);
    }
}
