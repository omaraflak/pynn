#include "tensor.h"
#include "cpu.h"
#include "gpu.h"
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

int32_t *_copy_shape(Tensor *tensor)
{
    int32_t *shape = (int32_t *)malloc(sizeof(int32_t) * tensor->dims);
    memcpy(shape, tensor->shape, sizeof(int32_t) * tensor->dims);
    return shape;
}

int32_t _get_size_from_shape(int32_t *shape, int32_t dims)
{
    int32_t size = 1;
    for (int32_t i = 0; i < dims; i++)
    {
        size *= shape[i];
    }
    return size;
}

int32_t _get_index(Tensor *tensor, int32_t *indices)
{
    int32_t index = 0;
    for (int32_t i = 0; i < tensor->dims; i++)
    {
        index += tensor->offset[i] + tensor->stride[i] * indices[i];
    }
    return index;
}

int32_t _mod(int32_t a, int32_t b)
{
    int32_t m = a % b;
    if (m < 0)
    {
        m = (b < 0) ? m - b : m + b;
    }
    return m;
}

Tensor *_tensor_create(float *data, int32_t *shape, int32_t dims, int32_t device)
{
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->data = data;
    tensor->shape = shape;
    tensor->stride = (int32_t *)malloc(sizeof(int32_t) * dims);
    tensor->offset = (int32_t *)malloc(sizeof(int32_t) * dims);
    tensor->size = _get_size_from_shape(shape, dims);
    tensor->dims = dims;
    tensor->device = device;
    tensor->base = nullptr;
    for (int32_t i = 0; i < dims; i++)
    {
        tensor->offset[i] = 0;
        tensor->stride[i] = 1;
        for (int32_t j = i + 1; j < dims; j++)
        {
            tensor->stride[i] *= shape[j];
        }
    }
    return tensor;
}

Tensor *tensor_create(float *data, int32_t *shape, int32_t dims)
{
    Tensor *tensor = tensor_create_empty(shape, dims);
    for (int32_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = data[i];
    }
    return tensor;
}

Tensor *tensor_create_empty(int32_t *shape, int32_t dims)
{
    int32_t size = _get_size_from_shape(shape, dims);
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->data = (float *)malloc(sizeof(float) * size);
    tensor->shape = (int32_t *)malloc(sizeof(int32_t) * dims);
    tensor->stride = (int32_t *)malloc(sizeof(int32_t) * dims);
    tensor->offset = (int32_t *)malloc(sizeof(int32_t) * dims);
    tensor->size = size;
    tensor->dims = dims;
    tensor->device = 0;
    tensor->base = nullptr;
    for (int32_t i = 0; i < dims; i++)
    {
        tensor->shape[i] = shape[i];
        tensor->offset[i] = 0;
        tensor->stride[i] = 1;
        for (int32_t j = i + 1; j < dims; j++)
        {
            tensor->stride[i] *= shape[j];
        }
    }
    return tensor;
}

Tensor *tensor_copy(Tensor *tensor)
{
    int32_t *shape = _copy_shape(tensor);

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
    if (tensor->base == nullptr)
    {
        if (tensor->device == 0)
        {
            free(tensor->data);
        }
        else
        {
            cudaFree(tensor->data);
        }
    }
    free(tensor->shape);
    free(tensor->stride);
    free(tensor->offset);
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
    if (tensor->device == 0)
    {
        tensor_fill_random_normal_cpu(tensor, mean, std);
    }
    else
    {
        tensor_fill_random_normal_gpu(tensor, mean, std);
    }
}

void tensor_fill_identity(Tensor *tensor)
{
    if (tensor->device == 0)
    {
        tensor_fill_identity_cpu(tensor);
    }
    else
    {
        tensor_fill_identity_gpu(tensor);
    }
}

void tensor_reshape(Tensor *tensor, int32_t *shape, int32_t dims)
{
    if (tensor->dims != dims)
    {
        free(tensor->shape);
        free(tensor->stride);
        tensor->dims = dims;
        tensor->shape = (int32_t *)malloc(sizeof(int32_t) * dims);
        tensor->stride = (int32_t *)malloc(sizeof(int32_t) * dims);
    }
    for (int32_t i = 0; i < dims; i++)
    {
        tensor->shape[i] = shape[i];
    }
    for (int32_t i = 0; i < dims; i++)
    {
        tensor->stride[i] = 1;
        for (int32_t j = i + 1; j < dims; j++)
        {
            tensor->stride[i] *= shape[j];
        }
    }
}

float tensor_get_item(Tensor *tensor, int32_t *indices)
{
    return tensor->data[_get_index(tensor, indices)];
}

void tensor_set_item(Tensor *tensor, int32_t *indices, float value)
{
    tensor->data[_get_index(tensor, indices)] = value;
}

Tensor *tensor_slice(Tensor *tensor, Slice *slices)
{
    int32_t *shape = (int32_t *)malloc(sizeof(int32_t) * tensor->dims);
    for (int32_t i = 0; i < tensor->dims; i++)
    {
        if (slices[i].start < 0)
        {
            slices[i].start = _mod(slices[i].start, tensor->shape[i]);
        }
        if (slices[i].stop < 0)
        {
            slices[i].stop = _mod(slices[i].stop, tensor->shape[i]);
        }
        shape[i] = ceil((float)(slices[i].stop - slices[i].start) / abs(slices[i].step));
    }

    Tensor *result = _tensor_create(tensor->data, shape, tensor->dims, tensor->device);
    result->base = tensor;

    for (int32_t i = 0; i < tensor->dims; i++)
    {
        result->offset[i] = slices[i].start * tensor->stride[i];
        result->stride[i] *= slices[i].step;
    }

    return result;
}

float tensor_sum(Tensor *tensor)
{
    if (tensor->device > 0)
    {
        fprintf(stderr, "tensor_sum is not supported on GPU.\n");
        exit(1);
    }
    return tensor_sum_cpu(tensor);
}

float tensor_mean(Tensor *tensor)
{
    if (tensor->device > 0)
    {
        fprintf(stderr, "tensor_mean is not supported on GPU.\n");
        exit(1);
    }
    return tensor_mean_cpu(tensor);
}

float tensor_min(Tensor *tensor)
{
    if (tensor->device > 0)
    {
        fprintf(stderr, "tensor_min is not supported on GPU.\n");
        exit(1);
    }
    return tensor_min_cpu(tensor);
}

float tensor_max(Tensor *tensor)
{
    if (tensor->device > 0)
    {
        fprintf(stderr, "tensor_max is not supported on GPU.\n");
        exit(1);
    }
    return tensor_max_cpu(tensor);
}

Tensor *tensor_unary_minus(Tensor *tensor)
{
    int32_t *shape = _copy_shape(tensor);
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

Tensor *tensor_transpose(Tensor *tensor, int32_t axis1, int32_t axis2)
{
    Tensor *copy = tensor_copy(tensor);

    int32_t tmp = copy->shape[axis1];
    copy->shape[axis1] = copy->shape[axis2];
    copy->shape[axis2] = tmp;

    tmp = copy->stride[axis1];
    copy->stride[axis1] = copy->stride[axis2];
    copy->stride[axis2] = tmp;

    return copy;
}

void tensor_add_into(Tensor *a, Tensor *b)
{
    if (a->device == 0 && b->device == 0)
    {
        tensor_add_cpu(a, b, a->data);
    }
    else if (a->device == b->device)
    {
        tensor_add_gpu(a, b, a->data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
}

void tensor_subtract_into(Tensor *a, Tensor *b)
{
    if (a->device == 0 && b->device == 0)
    {
        tensor_subtract_cpu(a, b, a->data);
    }
    else if (a->device == b->device)
    {
        tensor_subtract_gpu(a, b, a->data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
}

void tensor_multiply_into(Tensor *a, Tensor *b)
{
    if (a->device == 0 && b->device == 0)
    {
        tensor_multiply_cpu(a, b, a->data);
    }
    else if (a->device == b->device)
    {
        tensor_multiply_gpu(a, b, a->data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
}

void tensor_divide_into(Tensor *a, Tensor *b)
{
    if (a->device == 0 && b->device == 0)
    {
        tensor_divide_cpu(a, b, a->data);
    }
    else if (a->device == b->device)
    {
        tensor_divide_gpu(a, b, a->data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
}

Tensor *tensor_add(Tensor *a, Tensor *b)
{
    int32_t *shape = _copy_shape(a);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        tensor_add_cpu(a, b, data);
    }
    else if (a->device == b->device)
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        tensor_add_gpu(a, b, data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

Tensor *tensor_subtract(Tensor *a, Tensor *b)
{
    int32_t *shape = _copy_shape(a);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        tensor_subtract_cpu(a, b, data);
    }
    else if (a->device == b->device)
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        tensor_subtract_gpu(a, b, data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

Tensor *tensor_multiply(Tensor *a, Tensor *b)
{
    int32_t *shape = _copy_shape(a);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        tensor_multiply_cpu(a, b, data);
    }
    else if (a->device == b->device)
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        tensor_multiply_gpu(a, b, data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

Tensor *tensor_divide(Tensor *a, Tensor *b)
{
    int32_t *shape = _copy_shape(a);
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * a->size);
        tensor_divide_cpu(a, b, data);
    }
    else if (a->device == b->device)
    {
        cudaMalloc(&data, sizeof(float) * a->size);
        tensor_divide_gpu(a, b, data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

Tensor *tensor_matmul(Tensor *a, Tensor *b)
{
    int32_t batch = 1;
    int32_t *shape = (int32_t *)malloc(sizeof(int32_t) * a->dims);
    for (int32_t i = 0; i < a->dims - 2; i++)
    {
        batch *= a->shape[i];
        shape[i] = a->shape[i];
    }
    shape[a->dims - 2] = a->shape[a->dims - 2];
    shape[a->dims - 1] = b->shape[b->dims - 1];

    int32_t size = batch * a->shape[a->dims - 2] * b->shape[b->dims - 1];
    float *data;

    if (a->device == 0 && b->device == 0)
    {
        data = (float *)malloc(sizeof(float) * size);
        tensor_matmul_cpu(a, b, batch, data);
    }
    else if (a->device == b->device)
    {
        cudaMalloc(&data, sizeof(float) * size);
        tensor_matmul_gpu(a, b, batch, data);
    }
    else
    {
        fprintf(stderr, "Both tensors must be on the same device.\n");
        exit(1);
    }
    return _tensor_create(data, shape, a->dims, a->device);
}

void tensor_broadcast_add_into(Tensor *tensor, float value)
{
    if (tensor->device == 0)
    {
        tensor_broadcast_add_cpu(tensor, value, tensor->data);
    }
    else
    {
        tensor_broadcast_add_gpu(tensor, value, tensor->data);
    }
}

void tensor_broadcast_subtract_into(Tensor *tensor, float value)
{
    if (tensor->device == 0)
    {
        tensor_broadcast_subtract_cpu(tensor, value, tensor->data);
    }
    else
    {
        tensor_broadcast_subtract_gpu(tensor, value, tensor->data);
    }
}

void tensor_broadcast_multiply_into(Tensor *tensor, float value)
{
    if (tensor->device == 0)
    {
        tensor_broadcast_multiply_cpu(tensor, value, tensor->data);
    }
    else
    {
        tensor_broadcast_multiply_gpu(tensor, value, tensor->data);
    }
}

void tensor_broadcast_divide_into(Tensor *tensor, float value)
{
    if (tensor->device == 0)
    {
        tensor_broadcast_divide_cpu(tensor, value, tensor->data);
    }
    else
    {
        tensor_broadcast_divide_gpu(tensor, value, tensor->data);
    }
}

Tensor *tensor_broadcast_add(Tensor *tensor, float value)
{
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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
    int32_t *shape = _copy_shape(tensor);
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

Tensor *tensor_tanh(Tensor *tensor)
{
    int32_t *shape = _copy_shape(tensor);
    float *data;
    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * tensor->size);
        tensor_tanh_cpu(tensor, data);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * tensor->size);
        tensor_tanh_gpu(tensor, data);
    }
    return _tensor_create(data, shape, tensor->dims, tensor->device);
}