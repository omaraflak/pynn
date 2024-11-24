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

__global__ void fill_tensor_kernel(float *a, uint32_t n, float value)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        a[i] = value;
    }
}

void fill_tensor_gpu(Tensor *a, float value)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    fill_tensor_kernel<<<grid_dim, block_dim>>>(a->data, a->size, value);
    cudaDeviceSynchronize();
}

__global__ void unary_minus_tensor_kernel(float *a, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = -a[i];
    }
}

void unary_minus_tensor_gpu(Tensor *a, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    unary_minus_tensor_kernel<<<grid_dim, block_dim>>>(a->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void add_tensors_kernel(float *a, float *b, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] + b[i];
    }
}

void add_tensors_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    add_tensors_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void subtract_tensors_kernel(float *a, float *b, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] - b[i];
    }
}

void subtract_tensors_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    subtract_tensors_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void multiply_tensors_kernel(float *a, float *b, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] * b[i];
    }
}

void multiply_tensors_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    multiply_tensors_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void divide_tensors_kernel(float *a, float *b, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] / b[i];
    }
}

void divide_tensors_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    divide_tensors_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
    cudaDeviceSynchronize();
}

// MxP @ PxN => MxN
__global__ void matmul_tensors_kernel(float *a, float *b, uint32_t m, uint32_t p, uint32_t n, float *result)
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

void matmul_tensors_gpu(Tensor *a, Tensor *b, float *result)
{
    uint32_t m = a->shape[0];
    uint32_t p = a->shape[1];
    uint32_t n = b->shape[1];
    dim3 grid_dim, block_dim;
    _get_2d_gpu_config(&grid_dim, &block_dim, m, n);
    matmul_tensors_kernel<<<grid_dim, block_dim>>>(a->data, b->data, m, p, n, result);
    cudaDeviceSynchronize();
}

__global__ void broadcast_add_tensor_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] + value;
    }
}

void broadcast_add_tensor_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    broadcast_add_tensor_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void broadcast_subtract_tensor_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] - value;
    }
}

void broadcast_subtract_tensor_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    broadcast_subtract_tensor_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void broadcast_multiply_tensor_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] * value;
    }
}

void broadcast_multiply_tensor_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    broadcast_multiply_tensor_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void broadcast_divide_tensor_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = a[i] / value;
    }
}

void broadcast_divide_tensor_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    broadcast_divide_tensor_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}

__global__ void broadcast_right_divide_tensor_kernel(float *a, float value, uint32_t n, float *result)
{
    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t i = index; i < n; i += stride)
    {
        result[i] = value / a[i];
    }
}

void broadcast_right_divide_tensor_gpu(Tensor *a, float value, float *result)
{
    uint32_t grid_dim, block_dim;
    _get_1d_gpu_config(&grid_dim, &block_dim, a->size);
    broadcast_right_divide_tensor_kernel<<<grid_dim, block_dim>>>(a->data, value, a->size, result);
    cudaDeviceSynchronize();
}