#include "tensor.h"
#include "cpu.h"
#include "gpu.h"
#include <cstring>

uint32_t *_copy_shape(Tensor *tensor)
{
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * tensor->dims);
    memcpy(shape, tensor->shape, sizeof(uint32_t) * tensor->dims);
    return shape;
}

uint32_t _get_size_from_shape(uint32_t *shape, uint32_t dims)
{
    uint32_t size = 1;
    for (uint32_t i = 0; i < dims; i++)
    {
        size *= shape[i];
    }
    return size;
}

Tensor *_tensor_create(float *data, uint32_t *shape, uint32_t dims, uint32_t device)
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

Tensor *tensor_create(float *data, uint32_t *shape, uint32_t dims)
{
    Tensor *tensor = tensor_create_empty(shape, dims);
    for (uint32_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = data[i];
    }
    return tensor;
}

Tensor *tensor_create_empty(uint32_t *shape, uint32_t dims)
{
    uint32_t size = _get_size_from_shape(shape, dims);
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->data = (float *)malloc(sizeof(float) * size);
    tensor->shape = (uint32_t *)malloc(sizeof(uint32_t) * dims);
    tensor->stride = (uint32_t *)malloc(sizeof(uint32_t) * dims);
    tensor->size = size;
    tensor->dims = dims;
    tensor->device = 0;
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

Tensor *tensor_copy(Tensor *tensor)
{
    uint32_t *shape = _copy_shape(tensor);

    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        memcpy(data, tensor->data, sizeof(float) * tensor->size);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        cudaMemcpy(data, tensor->data, sizeof(float) * tensor->size, cudaMemcpyDeviceToDevice);
    }

    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

void tensor_delete(Tensor *tensor)
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

void tensor_fill(Tensor *tensor, float value)
{
    if (tensor->device == 0)
    {
        tensor_fill_cpu(tensor, value);
    }
    else
    {
        tensor_fill_gpu(tensor, value);
    }
}

void tensor_fill_random_uniform(Tensor *tensor, float min, float max)
{
    if (tensor->device == 0)
    {
        tensor_fill_random_uniform_cpu(tensor, min, max);
    }
    else
    {
        tensor_fill_random_uniform_gpu(tensor, min, max);
    }
}

void tensor_fill_random_normal(Tensor *tensor, float mean, float std)
{
    tensor_fill_random_normal_cpu(tensor, mean, std);
}

void tensor_reshape(Tensor *tensor, uint32_t *shape, uint32_t dims)
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

float tensor_get_item(Tensor *tensor, uint32_t *indices)
{
    uint32_t index = 0;
    for (uint32_t i = 0; i < tensor->dims; i++)
    {
        index += tensor->stride[i] * indices[i];
    }
    return tensor->data[index];
}

float tensor_sum(Tensor *tensor)
{
    return tensor_sum_cpu(tensor);
}

float tensor_mean(Tensor *tensor)
{
    return tensor_mean_cpu(tensor);
}

float tensor_min(Tensor *tensor)
{
    return tensor_min_cpu(tensor);
}

float tensor_max(Tensor *tensor)
{
    return tensor_max_cpu(tensor);
}

Tensor *tensor_unary_minus(Tensor *tensor)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;

    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_unary_minus_cpu(tensor, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_unary_minus_gpu(tensor, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_add(Tensor *a, Tensor *b)
{
    uint32_t *shape = _copy_shape(a);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        tensor_add_cpu(a, b, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        tensor_add_gpu(a, b, data);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

Tensor *tensor_subtract(Tensor *a, Tensor *b)
{
    uint32_t *shape = _copy_shape(a);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        tensor_subtract_cpu(a, b, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        tensor_subtract_gpu(a, b, data);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

Tensor *tensor_multiply(Tensor *a, Tensor *b)
{
    uint32_t *shape = _copy_shape(a);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        tensor_multiply_cpu(a, b, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        tensor_multiply_gpu(a, b, data);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

Tensor *tensor_divide(Tensor *a, Tensor *b)
{
    uint32_t *shape = _copy_shape(a);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        tensor_divide_cpu(a, b, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        tensor_divide_gpu(a, b, data);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

Tensor *tensor_matmul(Tensor *a, Tensor *b)
{
    uint32_t *shape = (uint32_t *)malloc(sizeof(uint32_t) * 2);
    shape[0] = a->shape[0];
    shape[1] = b->shape[1];
    uint32_t size = shape[0] * shape[1];
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * size);
        tensor_matmul_cpu(a, b, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * size);
        tensor_matmul_gpu(a, b, data);
    }
    return _tensor_create(data, shape, /* dims=*/2, a->device);
}

Tensor *tensor_broadcast_add(Tensor *tensor, float value)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;

    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_broadcast_add_cpu(tensor, value, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_broadcast_add_gpu(tensor, value, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_broadcast_subtract(Tensor *tensor, float value)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;

    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_broadcast_subtract_cpu(tensor, value, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_broadcast_subtract_gpu(tensor, value, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_broadcast_multiply(Tensor *tensor, float value)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;

    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_broadcast_multiply_cpu(tensor, value, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_broadcast_multiply_gpu(tensor, value, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_broadcast_divide(Tensor *tensor, float value)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;

    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_broadcast_divide_cpu(tensor, value, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_broadcast_divide_gpu(tensor, value, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_broadcast_right_divide(Tensor *tensor, float value)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;

    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_broadcast_right_divide_cpu(tensor, value, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_broadcast_right_divide_gpu(tensor, value, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_power(Tensor *tensor, float power)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_power_cpu(tensor, power, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_power_gpu(tensor, power, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_exp(Tensor *tensor)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_exp_cpu(tensor, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_exp_gpu(tensor, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_log(Tensor *tensor)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_log_cpu(tensor, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_log_gpu(tensor, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_log10(Tensor *tensor)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_log10_cpu(tensor, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_log10_gpu(tensor, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_logb(Tensor *tensor, float base)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_logb_cpu(tensor, base, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_logb_gpu(tensor, base, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_sin(Tensor *tensor)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_sin_cpu(tensor, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_sin_gpu(tensor, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}

Tensor *tensor_cos(Tensor *tensor)
{
    uint32_t *shape = _copy_shape(tensor);
    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_cos_cpu(tensor, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_cos_gpu(tensor, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}
