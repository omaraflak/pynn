#include "gpu.h"
#include <stdlib.h>

uint32_t MAX_BLOCK_DIM = 1024;
uint32_t MAX_BLOCK_DIM_1D = 1024;
uint32_t MAX_BLOCK_DIM_2D = 32;

void _get_1d_gpu_config(uint32_t *grid_dim, uint32_t *block_dim, uint32_t size)
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

void _get_2d_gpu_config(dim3 *grid_dim, dim3 *block_dim, uint32_t rows, uint32_t cols)
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

__global__ void tensor_fill_kernel(float *a, uint32_t n, float value)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        a[i] = value;
    }
}

void tensor_fill_gpu(Tensor *a, float value)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_fill_kernel<<<grid_dim, block_dim>>>(a->data, a->size, value);
    cudaDeviceSynchronize();
}

__global__ void tensor_unary_minus_kernel(float *a, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = -a[i];
    }
}

void tensor_unary_minus_gpu(Tensor *a, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_unary_minus_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_add_kernel(float *a, float *b, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] + b[i];
    }
}

void tensor_add_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_add_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_subtract_kernel(float *a, float *b, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] - b[i];
    }
}

void tensor_subtract_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_subtract_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_multiply_kernel(float *a, float *b, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] * b[i];
    }
}

void tensor_multiply_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_multiply_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_divide_kernel(float *a, float *b, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] / b[i];
    }
}

void tensor_divide_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_divide_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

// MxP @ PxN => MxN
__global__ void tensor_matmul_kernel(float *a, float *b, uint32_t m, uint32_t p, uint32_t n, float *result)
{
    uint32_t x_index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t x_stride = gridDim.x * blockDim.x;
    uint32_t y_index = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t y_stride = gridDim.y * blockDim.y;

    for (uint32_t i = y_index; i < m; i += y_stride)
    {
        for (uint32_t j = x_index; j < n; j += x_stride)
        {
            float tmp = 0;
            for (int k = 0; k < p; k++)
            {
                tmp += a[i * p + k] * b[k * n + j];
            }
            result[i * n + j] = tmp;
        }
    }
}

void tensor_matmul_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t m = a->shape[0];
    uint32_t p = a->shape[1];
    uint32_t n = b->shape[1];
    dim3 grid_dim, block_dim;
    _get_2d_gpu_config(&grid_dim, &block_dim, m, n);
    tensor_matmul_kernel<<<grid_dim, block_dim>>>(a->data, b->data, m, p, n, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_add_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] + value;
    }
}

void tensor_broadcast_add_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_add_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_subtract_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] - value;
    }
}

void tensor_broadcast_subtract_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_subtract_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_multiply_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] * value;
    }
}

void tensor_broadcast_multiply_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_multiply_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_divide_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] / value;
    }
}

void tensor_broadcast_divide_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_divide_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void tensor_broadcast_right_divide_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = value / a[i];
    }
}

void tensor_broadcast_right_divide_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    tensor_broadcast_right_divide_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}