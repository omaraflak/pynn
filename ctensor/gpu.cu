#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define cudaCheckError()                                                                     \
    {                                                                                        \
        cudaError_t e = cudaGetLastError();                                                  \
        if (e != cudaSuccess)                                                                \
        {                                                                                    \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    }

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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
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
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}

__global__ void tensor_sum_kernel(float *a, float *outputs, int32_t n)
{
    // Shared memory for block-level reduction
    extern __shared__ float sdata[];

    // Thread index within the block
    int32_t tid = threadIdx.x;

    // Global index
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < n) ? a[i] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int32_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        outputs[blockIdx.x] = sdata[0];
    }
}

float tensor_sum_gpu(Tensor *a)
{
    float *outputs;
    int32_t threadsPerBlock = 256;
    int32_t blocks = (a->size + threadsPerBlock - 1) / threadsPerBlock;
    int32_t sharedSize = threadsPerBlock * sizeof(float);

    cudaMalloc(&outputs, blocks * sizeof(float));

    tensor_sum_kernel<<<blocks, threadsPerBlock, sharedSize>>>(a->data, outputs, a->size);

    float *partial_sums = new float[blocks];
    cudaMemcpy(partial_sums, outputs, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i = 0; i < blocks; i++)
    {
        sum += partial_sums[i];
    }

    delete[] partial_sums;
    cudaFree(outputs);
    return sum;
}

float tensor_mean_gpu(Tensor *a)
{
    return tensor_sum_gpu(a) / a->size;
}

__device__ __forceinline__ float atomicMinFloat(float *addr, float value)
{
    return (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
}

__global__ void tensor_min_kernel(float *a, int32_t n, float *result)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Initialize local min to first element this thread processes
    float local_min = FLT_MAX;

    // Grid-stride loop to handle arrays larger than total thread count
    for (int i = gid; i < n; i += stride)
    {
        local_min = min(local_min, a[i]);
    }

    // Load local min into shared memory
    sdata[tid] = local_min;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result to global memory using atomic operation
    if (tid == 0)
    {
        atomicMinFloat(result, sdata[0]);
    }
}

float tensor_min_gpu(Tensor *a)
{
    float *c_output;
    float h_output = FLT_MAX;

    cudaMalloc(&c_output, sizeof(float));
    cudaMemcpy(c_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    int32_t threadsPerBlock = 256;
    int32_t blocksPerGrid = min((a->size + threadsPerBlock - 1) / threadsPerBlock, 1024);
    int32_t sharedSize = threadsPerBlock * sizeof(float);

    tensor_min_kernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(a->data, a->size, c_output);

    cudaMemcpy(&h_output, c_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(c_output);
    return h_output;
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value)
{
    return (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
}

__global__ void tensor_max_kernel(float *a, int32_t n, float *result)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Initialize local max to first element this thread processes
    float local_max = FLT_MIN;

    // Grid-stride loop to handle arrays larger than total thread count
    for (int i = gid; i < n; i += stride)
    {
        local_max = max(local_max, a[i]);
    }

    // Load local max into shared memory
    sdata[tid] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block result to global memory using atomic operation
    if (tid == 0)
    {
        atomicMaxFloat(result, sdata[0]);
    }
}

float tensor_max_gpu(Tensor *a)
{
    float *c_output;
    float h_output = FLT_MIN;

    cudaMalloc(&c_output, sizeof(float));
    cudaMemcpy(c_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    int32_t threadsPerBlock = 256;
    int32_t blocksPerGrid = max((a->size + threadsPerBlock - 1) / threadsPerBlock, 1024);
    int32_t sharedSize = threadsPerBlock * sizeof(float);

    tensor_max_kernel<<<blocksPerGrid, threadsPerBlock, sharedSize>>>(a->data, a->size, c_output);

    cudaMemcpy(&h_output, c_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(c_output);
    return h_output;
}