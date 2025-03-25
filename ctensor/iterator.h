#if !defined(ITERATOR)
#define ITERATOR

#include "_tensor.h"

typedef struct
{
	Tensor* tensor;
    int32_t* counters;
    int32_t idx;
} TensorIterator;

TensorIterator* iterator_create(Tensor* tensor);
bool iterator_has_next(TensorIterator* it);
int32_t iterator_next(TensorIterator* it);
void iterator_free(TensorIterator* it);

#endif // ITERATOR
