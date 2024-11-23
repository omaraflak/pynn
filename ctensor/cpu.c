#include "cpu.h"
#include <string.h>

void fill_tensor_data_cpu(Tensor *tensor, float value)
{
    for (uint32_t i = 0; i < tensor->size; i++)
    {
        tensor->data[i] = value;
    }
}

void add_tensors_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] + b->data[i];
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
