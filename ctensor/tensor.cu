#include "tensor.h"
#include "cpu.h"
#include "gpu.h"
#include <cstring>
#include <stdio.h>
#include <stdlib.h>

int32_t *_copy_shape(Tensor *tensor)
{
    int32_t *shape = (int32_t *)malloc(sizeof(int32_t) * tensor->dims);
    memcpy(shape, tensor->shape, sizeof(int32_t) * tensor->dims);
    return shape;
}

int32_t _compute_size(int32_t *shape, int32_t dims)
{
    int32_t size = 1;
    for (int32_t i = 0; i < dims; i++)
    {
        size *= shape[i];
    }
    return size;
}

void _compute_stride(int32_t *stride, int32_t *shape, int32_t dims)
{
    for (int32_t i = 0; i < dims; i++)
    {
        stride[i] = 1;
        for (int32_t j = i + 1; j < dims; j++)
        {
            stride[i] *= shape[j];
        }
    }
}

int32_t _get_slice_size(Slice *slice)
{
    return (slice->stop - slice->start + slice->step - 1) / slice->step;
}

int32_t _get_index(Tensor *tensor, int32_t index)
{
    if (!tensor->base)
    {
        return index;
    }

    int32_t remaining = index;
    int32_t base_index = tensor->offset;

    for (int32_t i=tensor->dims-1; i>=0; i--) {
        int32_t dim = remaining % tensor->shape[i];
        base_index += dim * tensor->stride[i];
        remaining /= tensor->shape[i];
    }

    return base_index;
}

int32_t _get_index(Tensor *tensor, int32_t *indices)
{
    int32_t index = tensor->offset;
    for (int32_t i = 0; i < tensor->dims; i++)
    {
        index += indices[i] * tensor->stride[i];
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
    tensor->size = _compute_size(shape, dims);
    tensor->dims = dims;
    tensor->device = device;
    tensor->base = nullptr;
    tensor->offset = 0;
    _compute_stride(tensor->stride, shape, dims);
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
    int32_t size = _compute_size(shape, dims);
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->data = (float *)malloc(sizeof(float) * size);
    tensor->shape = (int32_t *)malloc(sizeof(int32_t) * dims);
    tensor->stride = (int32_t *)malloc(sizeof(int32_t) * dims);
    tensor->size = size;
    tensor->dims = dims;
    tensor->device = 0;
    tensor->base = nullptr;
    tensor->offset = 0;
    memcpy(tensor->shape, shape, sizeof(int32_t) * dims);
    _compute_stride(tensor->stride, shape, dims);
    return tensor;
}

// TODO: copy sliced data instead of base array
Tensor *tensor_copy(Tensor *tensor)
{
    int32_t *shape = _copy_shape(tensor);
    int32_t size = tensor->base ? tensor->base->size : tensor->size;

    float *data;

    if (tensor->device == 0)
    {
        data = (float *)malloc(sizeof(float) * size);
        memcpy(data, tensor->data, sizeof(float) * size);
    }
    else
    {
        cudaMalloc(&data, sizeof(float) * size);
        cudaMemcpy(data, tensor->data, sizeof(float) * size, cudaMemcpyDeviceToDevice);
    }

    Tensor* result = _tensor_create(data, shape, tensor->dims, tensor->device);
    if (tensor->base) {
        result->base = tensor->base;
        result->offset = tensor->offset;
        memcpy(result->stride, tensor->stride, sizeof(int32_t) * tensor->dims);
    }

    return result;
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
    free(tensor);
}

void tensor_print_info(Tensor* tensor) {
    printf("tensor = %p\n", tensor);
    printf("base = %p\n", tensor->base);
    printf("data = %p\n", tensor->data);
    printf("stride = %p\n", tensor->stride);
    printf("shape = %p\n", tensor->shape);
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

bool _is_squeezing(Tensor *tensor, int32_t *shape, int32_t dims) {
    for (int32_t i=0; i<dims; i++) {
        if (shape[i] == 1) {
            continue;
        }
        bool found = false;
        for (int32_t j=0; j<tensor->dims; j++) {
            if (shape[i] == tensor->shape[j]) {
                found = true;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

Tensor* tensor_reshape(Tensor *tensor, int32_t *shape, int32_t dims)
{
    if (_compute_size(shape, dims) != tensor->size) {
        fprintf(stderr, "Cannot reshape to a size different than tensor size.\n");
        exit(1);
    }

    // make copy if not squeezing 1s dims
    if (tensor->base && !_is_squeezing(tensor, shape, dims)) {
        Tensor* result = tensor_create_empty(shape, dims);
        for (int i=0; i<tensor->size; i++) {
            result->data[i] = tensor->data[_get_index(tensor, i)];
        }
        return result;
    }

    // TODO: fix reshape when base exists
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

    return tensor;
}

float tensor_get_item_at(Tensor *tensor, int32_t index)
{
    return tensor->data[_get_index(tensor, index)];
}

float tensor_get_item(Tensor *tensor, int32_t *indices)
{
    return tensor->data[_get_index(tensor, indices)];
}

void tensor_set_item(Tensor *tensor, int32_t *indices, float value)
{
    tensor->data[_get_index(tensor, indices)] = value;
}

int32_t tensor_get_data_index(Tensor* tensor, int32_t index) {
    return _get_index(tensor, index);
}

Tensor *tensor_slice(Tensor *tensor, Slice *slice)
{
    // offset negative slicing and compute new size and shape
    int32_t *shape = (int32_t *)malloc(sizeof(int32_t) * tensor->dims);
    for (int32_t i = 0; i < tensor->dims; i++)
    {
        if (slice[i].step <= 0)
        {
            fprintf(stderr, "Slice step must be greater than 0.\n");
            exit(1);
        }
        if (slice[i].start < 0)
        {
            slice[i].start = _mod(slice[i].start, tensor->shape[i]);
        }
        if (slice[i].stop < 0)
        {
            slice[i].stop = _mod(slice[i].stop, tensor->shape[i]);
        }
        shape[i] = _get_slice_size(&slice[i]);
    }

    // create new tensor with updated strides
    Tensor *root = tensor->base ? tensor->base : tensor;
    Tensor *result = _tensor_create(root->data, shape, tensor->dims, tensor->device);
    result->base = root;

    result->offset = tensor->offset;
    for (int32_t i = 0; i < tensor->dims; i++)
    {
        result->offset += slice[i].start * tensor->stride[i];
        result->stride[i] = slice[i].step * tensor->stride[i];
    }

    return result;
}

float tensor_sum(Tensor *tensor)
{
    if (tensor->device == 0)
    {
        return tensor_sum_cpu(tensor);
    }
    else
    {
        return tensor_sum_gpu(tensor);
    }
}

float tensor_mean(Tensor *tensor)
{
    if (tensor->device == 0)
    {
        return tensor_mean_cpu(tensor);
    }
    else
    {
        return tensor_mean_gpu(tensor);
    }
}

float tensor_min(Tensor *tensor)
{
    if (tensor->device == 0)
    {
        return tensor_min_cpu(tensor);
    }
    else
    {
        return tensor_min_gpu(tensor);
    }
}

float tensor_max(Tensor *tensor)
{
    if (tensor->device == 0)
    {
        return tensor_max_cpu(tensor);
    }
    else
    {
        return tensor_max_gpu(tensor);
    }
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