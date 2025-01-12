#include "cpu.h"
#include "iterator.h"
#include <random>

void tensor_fill_cpu(Tensor *a, float value)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        a->data[it->value] = value;
    }
    iterator_free(it);
}

void tensor_fill_random_uniform_cpu(Tensor *a, float min, float max)
{
    std::mt19937 generator;
    std::uniform_real_distribution<float> uniform(min, max);
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        a->data[it->value] = uniform(generator);
    }
    iterator_free(it);
}

void tensor_fill_random_normal_cpu(Tensor *a, float mean, float std)
{
    std::mt19937 generator;
    std::normal_distribution<float> normal(mean, std);
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        a->data[it->value] = normal(generator);
    }
    iterator_free(it);
}

void tensor_fill_identity_cpu(Tensor *a)
{
    int32_t stride_sum = 0;
    for (int32_t i = 0; i < a->dims; i++)
    {
        stride_sum += a->stride[i];
    }
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        a->data[it->value] = it->value % stride_sum == 0 ? 1 : 0;
    }
    iterator_free(it);
}

void tensor_unary_minus_cpu(Tensor *a, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = -a->data[it->value];
    }
    iterator_free(it);
}

void tensor_add_cpu(Tensor *a, Tensor *b, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = a->data[it->value] + b->data[it->value];
    }
    iterator_free(it);
}

void tensor_subtract_cpu(Tensor *a, Tensor *b, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = a->data[it->value] - b->data[it->value];
    }
    iterator_free(it);
}

void tensor_multiply_cpu(Tensor *a, Tensor *b, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = a->data[it->value] * b->data[it->value];
    }
    iterator_free(it);
}

void tensor_divide_cpu(Tensor *a, Tensor *b, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = a->data[it->value] / b->data[it->value];
    }
    iterator_free(it);
}

void tensor_matmul_cpu(Tensor *a, Tensor *b, int32_t batch, float *result)
{
    int32_t res_height = a->shape[a->dims - 2];
    int32_t res_width = b->shape[b->dims - 1];
    int32_t common_dim = a->shape[a->dims - 1];
    int32_t a_idx, b_idx;

    int32_t a_batch_stride = res_height * common_dim;
    int32_t b_batch_stride = common_dim * res_width;
    int32_t r_batch_stride = res_height * res_width;

    for (int32_t t = 0; t < batch; t++)
    {
        for (int32_t i = 0; i < res_height; i++)
        {
            for (int32_t j = 0; j < res_width; j++)
            {
                float tmp = 0;
                for (int32_t k = 0; k < common_dim; k++)
                {
                    a_idx = t * a_batch_stride + i * a->stride[a->dims - 2] + k * a->stride[a->dims - 1];
                    b_idx = t * b_batch_stride + k * b->stride[b->dims - 2] + j * b->stride[b->dims - 1];
                    tmp += a->data[a_idx] * b->data[b_idx];
                }
                result[t * r_batch_stride + i * res_width + j] = tmp;
            }
        }
    }
}

void tensor_broadcast_add_cpu(Tensor *a, float value, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = a->data[it->value] + value;
    }
    iterator_free(it);
}

void tensor_broadcast_subtract_cpu(Tensor *a, float value, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = a->data[it->value] - value;
    }
    iterator_free(it);
}

void tensor_broadcast_multiply_cpu(Tensor *a, float value, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = a->data[it->value] * value;
    }
    iterator_free(it);
}

void tensor_broadcast_divide_cpu(Tensor *a, float value, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = a->data[it->value] / value;
    }
    iterator_free(it);
}

void tensor_broadcast_right_divide_cpu(Tensor *a, float value, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = value / a->data[it->value];
    }
    iterator_free(it);
}

void tensor_power_cpu(Tensor *a, float power, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = pow(a->data[it->value], power);
    }
    iterator_free(it);
}

void tensor_exp_cpu(Tensor *a, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = exp(a->data[it->value]);
    }
    iterator_free(it);
}

void tensor_log_cpu(Tensor *a, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = log(a->data[it->value]);
    }
    iterator_free(it);
}

void tensor_log10_cpu(Tensor *a, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = log10(a->data[it->value]);
    }
    iterator_free(it);
}
void tensor_logb_cpu(Tensor *a, float base, float *result)
{
    float inverse_log_base = 1.0 / log(base);
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = log(a->data[it->value]) * inverse_log_base;
    }
    iterator_free(it);
}

void tensor_sin_cpu(Tensor *a, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = sin(a->data[it->value]);
    }
    iterator_free(it);
}

void tensor_cos_cpu(Tensor *a, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = cos(a->data[it->value]);
    }
    iterator_free(it);
}

void tensor_tanh_cpu(Tensor *a, float *result)
{
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result[it->value] = tanh(a->data[it->value]);
    }
    iterator_free(it);
}

float tensor_sum_cpu(Tensor *a)
{
    float result = 0;
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        result += a->data[it->value];
    }
    iterator_free(it);
    return result;
}

float tensor_mean_cpu(Tensor *a)
{
    return tensor_sum_cpu(a) / a->size;
}

float tensor_min_cpu(Tensor *a)
{
    float result = a->data[0];
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        if (a->data[it->value] < result)
        {
            result = a->data[it->value];
        }
    }
    iterator_free(it);
    return result;
}

float tensor_max_cpu(Tensor *a)
{
    float result = a->data[0];
    Iterator *it = iterator_create(a);
    while (iterator_next(a, it))
    {
        if (a->data[it->value] > result)
        {
            result = a->data[it->value];
        }
    }
    iterator_free(it);
    return result;
}