#include "cpu.h"
#include <random>

void tensor_fill_cpu(Tensor *a, float value)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        a->data[i] = value;
    }
}

void tensor_fill_random_uniform_cpu(Tensor *a, float min, float max)
{
    std::mt19937 generator;
    std::uniform_real_distribution<float> uniform(min, max);
    for (uint32_t i = 0; i < a->size; i++)
    {
        a->data[i] = uniform(generator);
    }
}

void tensor_fill_random_normal_cpu(Tensor *a, float mean, float std)
{
    std::mt19937 generator;
    std::normal_distribution<float> normal(mean, std);
    for (uint32_t i = 0; i < a->size; i++)
    {
        a->data[i] = normal(generator);
    }
}

void tensor_unary_minus_cpu(Tensor *a, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = -a->data[i];
    }
}

void tensor_add_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] + b->data[i];
    }
}

void tensor_subtract_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] - b->data[i];
    }
}

void tensor_multiply_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] * b->data[i];
    }
}

void tensor_divide_cpu(Tensor *a, Tensor *b, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] / b->data[i];
    }
}

void tensor_matmul_cpu(Tensor *a, Tensor *b, float *result)
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

void tensor_broadcast_add_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] + value;
    }
}

void tensor_broadcast_subtract_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] - value;
    }
}

void tensor_broadcast_multiply_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] * value;
    }
}

void tensor_broadcast_divide_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[i] / value;
    }
}

void tensor_broadcast_right_divide_cpu(Tensor *a, float value, float *result)
{
    for (uint32_t i = 0; i < a->size; i++)
    {
        result[i] = value / a->data[i];
    }
}

float tensor_sum_cpu(Tensor *a)
{
    float result = 0;
    for (uint32_t i = 0; i < a->size; i++)
    {
        result += a->data[i];
    }
    return result;
}

float tensor_mean_cpu(Tensor *a)
{
    return tensor_sum_cpu(a) / a->size;
}
