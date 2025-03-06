#include "cpu.h"
#include <random>

int32_t get_slice_size(Slice *slice)
{
    return (slice->stop - slice->start + slice->step - 1) / slice->step;
}

int32_t idx(Tensor *tensor, int32_t index)
{
    if (!tensor->base)
    {
        return index;
    }

    int32_t remaining = index;
    int32_t base_index = 0;

    for (int32_t i = tensor->base->dims - 1; i >= 0; i--)
    {
        int32_t dim_size = get_slice_size(&tensor->slice[i]);
        int32_t dim_idx = remaining % dim_size;
        base_index += (tensor->slice[i].start + dim_idx * tensor->slice[i].step) * tensor->base->stride[i];
        remaining /= dim_size;
    }

    // printf("%d -> %d\n", index, base_index);

    return base_index;
}

void tensor_fill_cpu(Tensor *a, float value)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        a->data[idx(a, i)] = value;
    }
}

void tensor_fill_random_uniform_cpu(Tensor *a, float min, float max)
{
    std::mt19937 generator;
    std::uniform_real_distribution<float> uniform(min, max);
    for (int32_t i = 0; i < a->size; i++)
    {
        a->data[idx(a, i)] = uniform(generator);
    }
}

void tensor_fill_random_normal_cpu(Tensor *a, float mean, float std)
{
    std::mt19937 generator;
    std::normal_distribution<float> normal(mean, std);
    for (int32_t i = 0; i < a->size; i++)
    {
        a->data[idx(a, i)] = normal(generator);
    }
}

void tensor_fill_identity_cpu(Tensor *a)
{
    int32_t stride_sum = 0;
    for (int32_t i = 0; i < a->dims; i++)
    {
        stride_sum += a->stride[i];
    }
    for (int32_t i = 0; i < a->size; i++)
    {
        int32_t j = idx(a, i);
        a->data[j] = j % stride_sum == 0 ? 1 : 0;
    }
}

void tensor_unary_minus_cpu(Tensor *a, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = -a->data[idx(a, i)];
    }
}

void tensor_add_cpu(Tensor *a, Tensor *b, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[idx(a, i)] + b->data[idx(b, i)];
    }
}

void tensor_subtract_cpu(Tensor *a, Tensor *b, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[idx(a, i)] - b->data[idx(b, i)];
    }
}

void tensor_multiply_cpu(Tensor *a, Tensor *b, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[idx(a, i)] * b->data[idx(b, i)];
    }
}

void tensor_divide_cpu(Tensor *a, Tensor *b, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[idx(a, i)] / b->data[idx(b, i)];
    }
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
                    tmp += a->data[idx(a, a_idx)] * b->data[idx(b, b_idx)];
                }
                result[t * r_batch_stride + i * res_width + j] = tmp;
            }
        }
    }
}

void tensor_broadcast_add_cpu(Tensor *a, float value, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[idx(a, i)] + value;
    }
}

void tensor_broadcast_subtract_cpu(Tensor *a, float value, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[idx(a, i)] - value;
    }
}

void tensor_broadcast_multiply_cpu(Tensor *a, float value, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[idx(a, i)] * value;
    }
}

void tensor_broadcast_divide_cpu(Tensor *a, float value, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = a->data[idx(a, i)] / value;
    }
}

void tensor_broadcast_right_divide_cpu(Tensor *a, float value, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = value / a->data[idx(a, i)];
    }
}

void tensor_power_cpu(Tensor *a, float power, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = pow(a->data[idx(a, i)], power);
    }
}

void tensor_exp_cpu(Tensor *a, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = exp(a->data[idx(a, i)]);
    }
}

void tensor_log_cpu(Tensor *a, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = log(a->data[idx(a, i)]);
    }
}

void tensor_log10_cpu(Tensor *a, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = log10(a->data[idx(a, i)]);
    }
}
void tensor_logb_cpu(Tensor *a, float base, float *result)
{
    float inverse_log_base = 1.0 / log(base);
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = log(a->data[idx(a, i)]) * inverse_log_base;
    }
}

void tensor_sin_cpu(Tensor *a, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = sin(a->data[idx(a, i)]);
    }
}

void tensor_cos_cpu(Tensor *a, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = cos(a->data[idx(a, i)]);
    }
}

void tensor_tanh_cpu(Tensor *a, float *result)
{
    for (int32_t i = 0; i < a->size; i++)
    {
        result[i] = tanh(a->data[idx(a, i)]);
    }
}

float tensor_sum_cpu(Tensor *a)
{
    float result = 0;
    for (int32_t i = 0; i < a->size; i++)
    {
        result += a->data[idx(a, i)];
    }
    return result;
}

float tensor_mean_cpu(Tensor *a)
{
    return tensor_sum_cpu(a) / a->size;
}

float tensor_min_cpu(Tensor *a)
{
    float result = a->data[idx(a, 0)];
    for (int32_t i = 0; i < a->size; i++)
    {
        int32_t j = idx(a, i);
        if (a->data[j] < result)
        {
            result = a->data[j];
        }
    }
    return result;
}

float tensor_max_cpu(Tensor *a)
{
    float result = a->data[idx(a, 0)];
    for (int32_t i = 0; i < a->size; i++)
    {
        int32_t j = idx(a, i);
        if (a->data[j] > result)
        {
            result = a->data[j];
        }
    }
    return result;
}