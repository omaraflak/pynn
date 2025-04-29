#include "tensor.h"
#include "cpu.h"
#include "gpu.h"
#include "iterator.h"
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
    memcpy(tensor->data, data, sizeof(float) * tensor->size);
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

Tensor *tensor_copy(Tensor *tensor)
{
    int32_t initial_device = tensor->device;
    if (initial_device != 0) {
        tensor_gpu_to_cpu(tensor);
    }
    float* data = (float *)malloc(sizeof(float) * tensor->size);
    for (int32_t i = 0; i < tensor->size; i++)
    {
        data[i] = tensor->data[get_index(tensor, i)];
    }
    int32_t *shape = _copy_shape(tensor);
    Tensor* result = _tensor_create(data, shape, tensor->dims, tensor->device);
    if (initial_device != 0) {
        tensor_cpu_to_gpu(tensor);
        tensor_cpu_to_gpu(result);
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
    if (tensor->device != 0) {
        fprintf(stderr, "Trying to move a tensor already on GPU to GPU.\n");
        exit(1);
    }
    float *data;
    if (tensor->base) {
        int32_t size = tensor->base->size;
        float tmp[size];
        for (int32_t i=0; i<size; i++) {
            tmp[i] = tensor->data[get_index(tensor, i)];
        }
        cudaMalloc(&data, sizeof(float) * size);
        cudaMemcpy(data, tmp, sizeof(float) * size, cudaMemcpyHostToDevice);
        _compute_stride(tensor->stride, tensor->shape, tensor->dims);
        tensor->base = nullptr;
        tensor->offset = 0;
        tensor->data = data;
        tensor->device = 1;
    } else {
        int32_t size = tensor->size;
        cudaMalloc(&data, sizeof(float) * size);
        cudaMemcpy(data, tensor->data, sizeof(float) * size, cudaMemcpyHostToDevice);
        free(tensor->data);
        tensor->data = data;
        tensor->device = 1;
    }
}

void tensor_gpu_to_cpu(Tensor *tensor)
{
    if (tensor->device == 0) {
        fprintf(stderr, "Trying to move a tensor already on CPU to CPU.\n");
        exit(1);
    }
    if (tensor->base) {
        int32_t size = tensor->base->size;
        float *tmp = (float *)malloc(sizeof(float) * size);
        cudaMemcpy(tmp, tensor->data, sizeof(float) * size, cudaMemcpyDeviceToHost);
        float *data = (float *)malloc(sizeof(float) * tensor->size);
        for (int32_t i=0; i<tensor->size; i++) {
            data[i] = tmp[get_index(tensor, i)];
        }
        _compute_stride(tensor->stride, tensor->shape, tensor->dims);
        tensor->base = nullptr;
        tensor->offset = 0;
        tensor->data = data;
        tensor->device = 0;
    } else {
        int32_t size = tensor->size;
        float *data = (float *)malloc(sizeof(float) * size);
        cudaMemcpy(data, tensor->data, sizeof(float) * size, cudaMemcpyDeviceToHost);
        cudaFree(tensor->data);
        tensor->data = data;
        tensor->device = 0;
    }
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

bool _is_squeezing_or_expanding(int32_t* shape1, int32_t dim1, int32_t* shape2, int32_t dim2) {
    int32_t i = 0;
    int32_t j = 0;

    while (i < dim1 && j < dim2) {
        if (shape1[i] == shape2[j]) {
            i++;
            j++;
        } else if (shape1[i] == 1) {
            i++;
        } else if (shape2[j] == 1) {
            j++;
        } else {
            return false;
        }
    }

    if (i < dim1) {
        for (int32_t k=i; k<dim1; k++) {
            if (shape1[k] != 1) {
                return false;
            }
        }
        return true;
    }

    if (j < dim2) {
        for (int32_t k=j; k<dim2; k++) {
            if (shape2[k] != 1) {
                return false;
            }
        }
        return true;
    }

    return true;
}

int32_t _index_of(int32_t* array, int32_t size, int32_t value, int32_t start) {
    for (int32_t i=start; i<size; i++) {
        if (array[i] == value) {
            return i;
        }
    }
    return -1;
}

// Special function to compute strides when a tensor is sliced then reshaped,
// but only expanding or squeezing the shape (adding/removing 1s from the shape)
void _compute_compatible_stride(
    int32_t* old_shape,
    int32_t* old_stride,
    int32_t old_dims,
    int32_t* new_shape,
    int32_t* new_stride,
    int32_t new_dims
) {
    // example 1:
    // (2, 2, 2) -> (2, 1, 2, 2, 1)
    // (2,) -> (2, 1)
    // (2,) -> (2,)
    // (2,) -> (2, 1)
    // (x, y, z) -> (x, x, y, z, z)

    // example 2:
    // (2, 1, 2, 2, 1) -> (2, 2, 2)
    // (2, 1) -> (2,)
    // (2,)   -> (2,)
    // (2, 1) -> (2,)
    // (x, y, z, t, u) -> (x, z, t)

    int32_t i=0;
    int32_t j=-1;

    while (i < new_dims) {
        // get leading 1s
        while (i < new_dims && new_shape[i] == 1) {
            new_stride[i] = 1;
            i++;
        }

        // check that we didn't reach the end
        if (i == new_dims) {
            new_stride[new_dims - 1] = old_stride[old_dims - 1];
            break;
        }

        // get stride for dim
        j = _index_of(old_shape, old_dims, new_shape[i], j + 1);
        new_stride[i] = old_stride[j];
        i++;

        // get trailing 1s
        while (i < new_dims && new_shape[i] == 1) {
            new_stride[i] = 1;
            i++;
        }
    }
}

Tensor* tensor_reshape(Tensor *tensor, int32_t *shape, int32_t dims)
{
    if (_compute_size(shape, dims) != tensor->size) {
        fprintf(stderr, "Cannot reshape to a size different than tensor size.\n");
        exit(1);
    }

    // not a sliced tensor: recompute shape and stride normally
    if (!tensor->base) {
        if (tensor->dims != dims)
        {
            free(tensor->shape);
            free(tensor->stride);
            tensor->dims = dims;
            tensor->shape = (int32_t *)malloc(sizeof(int32_t) * dims);
            tensor->stride = (int32_t *)malloc(sizeof(int32_t) * dims);
        }
        memcpy(tensor->shape, shape, sizeof(int32_t) * dims);
        _compute_stride(tensor->stride, tensor->shape, dims);
        return tensor;
    }

    // sliced tensor: check if we can reference the base array, or have to make a copy
    bool can_reference = _is_squeezing_or_expanding(
        tensor->shape,
        tensor->dims,
        shape,
        dims
    );

    // incomptable reshape: make a copy of the sliced data
    if (!can_reference) {
        Tensor* result = tensor_copy(tensor);
        return tensor_reshape(result, shape, dims);
    }
    

    // compatible reshape: adjust strides
    int32_t* new_shape = (int32_t *)malloc(sizeof(int32_t) * dims);
    int32_t* new_stride = (int32_t *)malloc(sizeof(int32_t) * dims);
    memcpy(new_shape, shape, sizeof(int32_t) * dims);
    _compute_compatible_stride(
        tensor->shape,
        tensor->stride,
        tensor->dims,
        new_shape,
        new_stride,
        dims
    );
    free(tensor->shape);
    free(tensor->stride);
    tensor->shape = new_shape;
    tensor->stride = new_stride;
    tensor->dims = dims;
    return tensor;
}

float tensor_get_item_at(Tensor *tensor, int32_t index)
{
    return tensor->data[get_index(tensor, index)];
}

float tensor_get_item(Tensor *tensor, int32_t *indices)
{
    return tensor->data[get_index(tensor, indices)];
}

void tensor_set_item(Tensor *tensor, int32_t *indices, float value)
{
    tensor->data[get_index(tensor, indices)] = value;
}

int32_t tensor_get_data_index(Tensor* tensor, int32_t index) {
    return get_index(tensor, index);
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