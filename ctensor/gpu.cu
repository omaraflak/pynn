#include "gpu.h"
#include "tensor.h"
#include <stdlib.h>

void delete_device_tensor(Tensor *tensor)
{
    cudaFree(tensor->data);
    free(tensor->shape);
    free(tensor->stride);
    free(tensor);
}

void host_to_device(Tensor *tensor)
{
    float *data;
    cudaMalloc(&data, sizeof(float) * tensor->size);
    cudaMemcpy(data, tensor->data, sizeof(float) * tensor->size, cudaMemcpyHostToDevice);
    free(tensor->data);
    tensor->data = data;
    tensor->device = 1;
}

void device_to_host(Tensor *tensor)
{
    float *data = (float *)malloc(sizeof(float) * tensor->size);
    cudaMemcpy(data, tensor->data, sizeof(float) * tensor->size, cudaMemcpyDeviceToHost);
    cudaFree(tensor->data);
    tensor->data = data;
    tensor->device = 0;
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
    uint32_t block_dim = 256;
    uint32_t grid_dim = (a->size + block_dim - 1) / block_dim;
    add_tensors_kernel<<<grid_dim, block_dim>>>(a->data, b->data, a->size, result);
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

    dim3 block_dim(m, n);
    dim3 grid_dim(1, 1);
    if (m * n > 1024)
    {
        block_dim.x = 32;
        block_dim.y = 32;
        grid_dim.x = (n + block_dim.x - 1) / block_dim.x;
        grid_dim.y = (m + block_dim.y - 1) / block_dim.y;
    }

    matmul_tensors_kernel<<<grid_dim, block_dim>>>(a->data, b->data, m, p, n, result);
    cudaDeviceSynchronize();
}