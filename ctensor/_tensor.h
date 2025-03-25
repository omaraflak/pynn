#if !defined(_TENSOR)
#define _TENSOR

typedef struct
{
    int32_t start;
    int32_t stop;
    int32_t step;
} Slice;

typedef struct Tensor Tensor;

struct Tensor
{
    float *data;
    int32_t *shape;
    int32_t *stride;
    int32_t dims;
    int32_t size;
    int32_t device;
    int32_t offset;
    Tensor *base;
};

#endif // _TENSOR