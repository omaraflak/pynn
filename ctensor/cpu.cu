#include "cpu.h"

void fill_tensor_cpu(Tensor *a, float value)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        a->data[i] = value;
    }
}

void add_tensors_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] + b->data[i];
    }
}

void subtract_tensors_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] - b->data[i];
    }
}

void multiply_tensors_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] * b->data[i];
    }
}

void divide_tensors_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] / b->data[i];
    }
}

void matmul_tensors_cpu(Tensor *a, Tensor *b, float *result)
{
    for (int i = 0; i < a->shape[0]; i++)
    {
        for (int j = 0; j < b->shape[1]; j++)
        {
            float tmp = 0;
            for (int k = 0; k < b->shape[0]; k++)
            {
                tmp += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
            }
            result[i * b->shape[1] + j] = tmp;
        }
    }
}

void broadcast_add_tensor_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] + value;
    }
}

void broadcast_subtract_tensor_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] - value;
    }
}

void broadcast_multiply_tensor_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] * value;
    }
}

void broadcast_divide_tensor_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] / value;
    }
}

void broadcast_right_divide_tensor_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = value / a->data[i];
    }
}