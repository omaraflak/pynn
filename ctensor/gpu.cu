#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

int32_t MAX_BLOCK_DIM = 1024;
int32_t MAX_BLOCK_DIM_1D = 1024;
int32_t MAX_BLOCK_DIM_2D = 32;
int32_t MAX_BLOCK_DIM_3D = 10;

void _get_1d_gpu_config(int32_t *grid_dim, int32_t *block_dim, int32_t size)
{
    if (size > MAX_BLOCK_DIM)
    {
        *block_dim = MAX_BLOCK_DIM_1D;
        *grid_dim = (size + *block_dim - 1) / *block_dim;
    }
    else
    {
        *block_dim = size;
        *grid_dim = 1;
    }
}

void _get_2d_gpu_config(dim3 *grid_dim, dim3 *block_dim, int32_t rows, int32_t cols)
{
    if (rows * cols > MAX_BLOCK_DIM)
    {
        block_dim->x = MAX_BLOCK_DIM_2D;
        block_dim->y = MAX_BLOCK_DIM_2D;
        grid_dim->x = (cols + block_dim->x - 1) / block_dim->x;
        grid_dim->y = (rows + block_dim->y - 1) / block_dim->y;
    }
    else
    {
        block_dim->x = cols;
        block_dim->y = rows;
        grid_dim->x = 1;
        grid_dim->y = 1;
    }
}

void _get_3d_gpu_config(dim3 *grid_dim, dim3 *block_dim, int32_t batch, int32_t rows, int32_t cols)
{
    if (batch * rows * cols > MAX_BLOCK_DIM)
    {
        block_dim->x = MAX_BLOCK_DIM_3D;
        block_dim->y = MAX_BLOCK_DIM_3D;
        block_dim->z = MAX_BLOCK_DIM_3D;
        grid_dim->x = (cols + block_dim->x - 1) / block_dim->x;
        grid_dim->y = (rows + block_dim->y - 1) / block_dim->y;
        grid_dim->z = (batch + block_dim->z - 1) / block_dim->z;
    }
    else
    {
        block_dim->x = cols;
        block_dim->y = rows;
        block_dim->z = batch;
        grid_dim->x = 1;
        grid_dim->y = 1;
        grid_dim->z = 1;
    }
}

__global__ void tensor_fill_kernel(float *a, int32_t n, float value)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        a[i] = value;
    }
}

void tensor_fill_gpu(Tensor *a, float value)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_fill_kernel<<<grid_dim, block_dim>>>(a->data, a->size, value);
    cudaDeviceSynchronize();
}

__global__ void tensor_fill_random_uniform_kernel(float *a, int32_t n, float min, float max)
{
    curandState_t state;
    curand_init(clock64(), 0, 0, &state);

    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        a[i] = min + curand_uniform(&state) * (max - min);
    }
}

void tensor_fill_random_uniform_gpu(Tensor *a, float min, float max)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_fill_random_uniform_kernel<<<grid_dim, block_dim>>>(a->data, a->size, min, max);
    cudaDeviceSynchronize();
}

__global__ void tensor_fill_random_normal_kernel(float *a, int32_t n, float mean, float std)
{
    curandState_t state;
    curand_init(clock64(), 0, 0, &state);

    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        a[i] = mean + curand_normal(&state) * std;
    }
}

void tensor_fill_random_normal_gpu(Tensor *a, float mean, float std)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_fill_random_normal_kernel<<<grid_dim, block_dim>>>(a->data, a->size, mean, std);
    cudaDeviceSynchronize();
}

__global__ void tensor_fill_identity_kernel(float *a, int32_t n, int32_t stride_sum)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        a[i] = i % stride_sum == 0 ? 1 : 0;
    }
}

void tensor_fill_identity_gpu(Tensor *a)
{
    int32_t stride_sum = 0;
    for (int32_t i = 0; i < a->dims; i++)
    {
        stride_sum += a->stride[i];
    }
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_fill_identity_kernel<<<grid_dim, block_dim>>>(a->data, a->size, stride_sum);
    cudaDeviceSynchronize();
}

__global__ void tensor_unary_minus_kernel(float *a, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = -a[i];
    }
}

void tensor_unary_minus_gpu(Tensor *a, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_unary_minus_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_add_kernel(float *a, float *b, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] + b[i];
    }
}

void tensor_add_gpu(Tensor *a, Tensor *b, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_add_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_subtract_kernel(float *a, float *b, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] - b[i];
    }
}

void tensor_subtract_gpu(Tensor *a, Tensor *b, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_subtract_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_multiply_kernel(float *a, float *b, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] * b[i];
    }
}

void tensor_multiply_gpu(Tensor *a, Tensor *b, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_multiply_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_divide_kernel(float *a, float *b, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] / b[i];
    }
}

void tensor_divide_gpu(Tensor *a, Tensor *b, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_divide_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

// TxMxP @ TxPxN => TxMxN
__global__ void tensor_matmul_kernel(
    float *a,
    float *b,
    int32_t *a_stride,
    int32_t *b_stride,
    int32_t dims,
    int32_t t,
    int32_t m,
    int32_t p,
    int32_t n,
    float *result)
{
    int32_t x_index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t x_stride = gridDim.x * blockDim.x;
    int32_t y_index = blockDim.y * blockIdx.y + threadIdx.y;
    int32_t y_stride = gridDim.y * blockDim.y;
    int32_t z_index = blockDim.z * blockIdx.z + threadIdx.z;
    int32_t z_stride = gridDim.z * blockDim.z;

    int32_t a_batch_stride = m * p;
    int32_t b_batch_stride = p * n;
    int32_t r_batch_stride = m * n;

    int32_t a_idx, b_idx;

    for (int32_t z = z_index; z < t; z += z_stride)
    {
        for (int32_t i = y_index; i < m; i += y_stride)
        {
            for (int32_t j = x_index; j < n; j += x_stride)
            {
                float tmp = 0;
                for (int32_t k = 0; k < p; k++)
                {
                    a_idx = z * a_batch_stride + i * a_stride[dims - 2] + k * a_stride[dims - 1];
                    b_idx = z * b_batch_stride + k * b_stride[dims - 2] + j * b_stride[dims - 1];
                    tmp += a[a_idx] * b[b_idx];
                }
                result[z * r_batch_stride + i * n + j] = tmp;
            }
        }
    }
}

void tensor_matmul_gpu(Tensor *a, Tensor *b, int32_t batch, float *result)
{
    int32_t rows = a->shape[a->dims - 2];
    int32_t comm = a->shape[a->dims - 1];
    int32_t cols = b->shape[b->dims - 1];
    int32_t *a_stride;
    int32_t *b_stride;
    cudaMalloc(&a_stride, sizeof(int32_t) * a->dims);
    cudaMalloc(&b_stride, sizeof(int32_t) * b->dims);
    cudaMemcpy(a_stride, a->stride, sizeof(int32_t) * a->dims, cudaMemcpyHostToDevice);
    cudaMemcpy(b_stride, b->stride, sizeof(int32_t) * b->dims, cudaMemcpyHostToDevice);
    dim3 grid_dim, block_dim;
    _get_3d_gpu_config(&grid_dim, &block_dim, batch, rows, cols);
    tensor_matmul_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a_stride, b_stride, a->dims, batch, rows, comm, cols, result);
    cudaDeviceSynchronize();
    cudaFree(a_stride);
    cudaFree(b_stride);
}

__global__ void tensor_broadcast_add_kernel(float *a, int32_t n, float value, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] + value;
    }
}

void tensor_broadcast_add_gpu(Tensor *a, float value, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_add_kernel<<<grid_dim, block_dim>>>(a->data, a->size, value, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_subtract_kernel(float *a, int32_t n, float value, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] - value;
    }
}

void tensor_broadcast_subtract_gpu(Tensor *a, float value, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_subtract_kernel<<<grid_dim, block_dim>>>(a->data, a->size, value, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_multiply_kernel(float *a, int32_t n, float value, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] * value;
    }
}

void tensor_broadcast_multiply_gpu(Tensor *a, float value, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_multiply_kernel<<<grid_dim, block_dim>>>(a->data, a->size, value, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_divide_kernel(float *a, int32_t n, float value, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] / value;
    }
}

void tensor_broadcast_divide_gpu(Tensor *a, float value, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_divide_kernel<<<grid_dim, block_dim>>>(a->data, a->size, value, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_right_divide_kernel(float *a, int32_t n, float value, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = value / a[i];
    }
}

void tensor_broadcast_right_divide_gpu(Tensor *a, float value, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_right_divide_kernel<<<grid_dim, block_dim>>>(a->data, a->size, value, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_power_kernel(float *a, int32_t n, float power, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = pow(a[i], power);
    }
}

void tensor_power_gpu(Tensor *a, float power, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_power_kernel<<<grid_dim, block_dim>>>(a->data, a->size, power, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_exp_kernel(float *a, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = exp(a[i]);
    }
}

void tensor_exp_gpu(Tensor *a, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_exp_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_log_kernel(float *a, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = log(a[i]);
    }
}

void tensor_log_gpu(Tensor *a, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_log_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_log10_kernel(float *a, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = log10(a[i]);
    }
}

void tensor_log10_gpu(Tensor *a, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_log10_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_logb_kernel(float *a, int32_t n, float base, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    float inverse_log_base = 1.0 / log(base);
    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = log(a[i]) * inverse_log_base;
    }
}

void tensor_logb_gpu(Tensor *a, float base, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_logb_kernel<<<grid_dim, block_dim>>>(a->data, a->size, base, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_sin_kernel(float *a, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = sin(a[i]);
    }
}

void tensor_sin_gpu(Tensor *a, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_sin_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_cos_kernel(float *a, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = cos(a[i]);
    }
}

void tensor_cos_gpu(Tensor *a, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_cos_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_tanh_kernel(float *a, int32_t n, float *result)
{
    int32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;

    for (int32_t i = index; i < n; i += stride)
    {
        result[i] = tanh(a[i]);
    }
}

void tensor_tanh_gpu(Tensor *a, float *result)
{
    int32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_tanh_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}