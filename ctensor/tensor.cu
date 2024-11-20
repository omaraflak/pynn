#include "tensor.h"

__device__ uint32_t get_index_in_tensor(Tensor *tensor, uint32_t *indices)
{
    uint32_t index = 0;
    for (uint32_t i = 0; i < tensor->dims; i++)
    {
        index += tensor->stride[i] * indices[i];
    }
    return index;
}

__global__ void matmul_tensor_kernel(Tensor *x, Tensor *y, Tensor *result)
{
    uint32_t x_index = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t x_stride = gridDim.x * blockDim.x;
    uint32_t y_index = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t y_stride = gridDim.y * blockDim.y;

    uint32_t indices[2];
    uint32_t idx1, idx2, idx3;

    for (uint32_t i = y_index; i < result->shape[0]; i += y_stride)
    {
        for (uint32_t j = x_index; j < result->shape[1]; j += x_stride)
        {
            float tmp = 0;
            for (uint32_t k = 0; k < y->shape[0]; k++)
            {
                indices[0] = i;
                indices[1] = k;
                idx1 = get_index_in_tensor(x, indices);
                indices[0] = k;
                indices[1] = j;
                idx2 = get_index_in_tensor(y, indices);
                tmp += x->data[idx2] * y->data[idx3];
            }

            indices[0] = i;
            indices[1] = j;
            idx1 = get_index_in_tensor(result, indices);
            result->data[idx1] = tmp;
        }
    }
}

extern "C"
{
    Tensor *create_tensor(uint32_t *shape, uint32_t dims)
    {
        Tensor *tensor = new Tensor();
        tensor->dims = dims;
        tensor->size = get_size_from_shape(shape, dims);
        tensor->data = new float[tensor->size];
        tensor->shape = new uint32_t[dims];
        tensor->stride = new uint32_t[dims];
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
        return tensor;
    }

    void delete_tensor(Tensor *tensor)
    {
        delete[] tensor->data;
        delete[] tensor->shape;
        delete[] tensor->stride;
        delete tensor;
    }

    Tensor *create_device_tensor(uint32_t *shape, uint32_t dims)
    {
        Tensor *host = create_tensor(shape, dims);
        Tensor *device = to_device(host);
        delete_tensor(host);
        return device;
    }

    void delete_device_tensor(Tensor *tensor)
    {
        Tensor host;
        cudaMemcpy(&host, tensor, sizeof(Tensor), cudaMemcpyDeviceToHost);
        cudaFree(host.data);
        cudaFree(host.shape);
        cudaFree(host.stride);
        cudaFree(tensor);
    }

    uint32_t get_size_from_shape(uint32_t *shape, uint32_t dims)
    {
        uint32_t size = 1;
        for (uint32_t i = 0; i < dims; i++)
        {
            size *= shape[i];
        }
        return size;
    }

    void fill_tensor(Tensor *x, float value)
    {
        for (uint32_t i = 0; i < x->size; i++)
        {
            x->data[i] = value;
        }
    }

    Tensor *to_device(Tensor *tensor)
    {
        float *data;
        uint32_t *shape;
        uint32_t *stride;

        cudaMalloc(&data, sizeof(float) * tensor->size);
        cudaMalloc(&shape, sizeof(uint32_t) * tensor->dims);
        cudaMalloc(&stride, sizeof(uint32_t) * tensor->dims);
        cudaMemcpy(data, tensor->data, sizeof(float) * tensor->size, cudaMemcpyHostToDevice);
        cudaMemcpy(shape, tensor->shape, sizeof(uint32_t) * tensor->dims, cudaMemcpyHostToDevice);
        cudaMemcpy(stride, tensor->stride, sizeof(uint32_t) * tensor->dims, cudaMemcpyHostToDevice);

        Tensor host;
        host.data = data;
        host.shape = shape;
        host.stride = stride;
        host.dims = tensor->dims;
        host.size = tensor->size;

        Tensor *device;
        cudaMalloc(&device, sizeof(Tensor));
        cudaMemcpy(device, &host, sizeof(Tensor), cudaMemcpyHostToDevice);
        return device;
    }

    Tensor *to_host(Tensor *tensor)
    {
        Tensor device;
        cudaMemcpy(&device, tensor, sizeof(Tensor), cudaMemcpyDeviceToHost);

        Tensor *host = new Tensor();
        host->data = new float[device.size];
        host->shape = new uint32_t[device.dims];
        host->stride = new uint32_t[device.dims];
        host->dims = device.dims;
        host->size = device.size;

        cudaMemcpy(host->data, device.data, sizeof(float) * device.size, cudaMemcpyDeviceToHost);
        cudaMemcpy(host->stride, device.stride, sizeof(uint32_t) * device.dims, cudaMemcpyDeviceToHost);
        cudaMemcpy(host->shape, device.shape, sizeof(uint32_t) * device.dims, cudaMemcpyDeviceToHost);

        return host;
    }

    Tensor *add_tensor(Tensor *a, Tensor *b)
    {
        Tensor *result = create_tensor(a->shape, a->dims);
        for (uint32_t i = 0; i < result->size; i++)
        {
            result->data[i] = a->data[i] + b->data[i];
        }
        return result;
    }

    Tensor *subtract_tensor(Tensor *a, Tensor *b)
    {
        Tensor *result = create_tensor(a->shape, a->dims);
        for (uint32_t i = 0; i < result->size; i++)
        {
            result->data[i] = a->data[i] - b->data[i];
        }
        return result;
    }

    Tensor *multiply_tensor(Tensor *a, Tensor *b)
    {
        Tensor *result = create_tensor(a->shape, a->dims);
        for (uint32_t i = 0; i < result->size; i++)
        {
            result->data[i] = a->data[i] * b->data[i];
        }
        return result;
    }

    Tensor *divide_tensor(Tensor *a, Tensor *b)
    {
        Tensor *result = create_tensor(a->shape, a->dims);
        for (uint32_t i = 0; i < result->size; i++)
        {
            result->data[i] = a->data[i] / b->data[i];
        }
        return result;
    }

    Tensor *matmul_tensor(Tensor *a, Tensor *b)
    {
        uint32_t shape[2] = {a->shape[0], b->shape[1]};
        Tensor *result = create_tensor(shape, 2);
        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                float tmp = 0;
                for (int k = 0; k < b->shape[0]; k++)
                {
                    tmp += a->data[i * a->shape[0] + k] * b->data[k * b->shape[0] + j];
                }
                result->data[i * shape[0] + j] = tmp;
            }
        }
        return result;
    }

    Tensor *matmul_tensor_gpu(Tensor *tensor1, Tensor *tensor2)
    {
        uint32_t shape[2];
        shape[0] = tensor1->shape[0];
        shape[1] = tensor2->shape[1];

        Tensor *x = to_device(tensor1);
        Tensor *y = to_device(tensor2);
        Tensor *result = create_device_tensor(shape, 2);

        dim3 block_dim(shape[0], shape[1]);
        dim3 grid_dim(1, 1);
        if (shape[0] * shape[1] > 1024)
        {
            block_dim.x = 32;
            block_dim.y = 32;
            grid_dim.x = (shape[1] + block_dim.x - 1) / block_dim.x;
            grid_dim.y = (shape[0] + block_dim.y - 1) / block_dim.y;
        }

        matmul_tensor_kernel<<<grid_dim, block_dim>>>(x, y, result);
        cudaDeviceSynchronize();

        Tensor *host = to_host(result);

        delete_device_tensor(x);
        delete_device_tensor(y);
        delete_device_tensor(result);

        return host;
    }
}