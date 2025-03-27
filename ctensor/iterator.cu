#include "iterator.h"

TensorIterator* iterator_create(Tensor* tensor) {
	TensorIterator* it = (TensorIterator*) malloc(sizeof(TensorIterator));
	it->tensor = tensor;
	it->idx = tensor->offset;
	if (it->tensor->base) {
		it->counters = (int32_t*) malloc(sizeof(int32_t) * tensor->dims);
		for (int32_t i=0; i<tensor->dims; i++) {
			it->counters[i] = 0;
		}
	}
	return it;
}

void iterator_next(TensorIterator* it) {
	if (!it->tensor->base) {
		it->idx++;
		if (it->idx == it->tensor->size) {
			it->idx = -1;
		}
		return;
	}

	bool increment = false;
	for (int32_t i=it->tensor->dims - 1; i>=0; i--) {
		if (it->counters[i] + 1 < it->tensor->shape[i]) {
			it->counters[i]++;
			it->idx += it->tensor->stride[i];
			increment = true;
			break;
		}
		it->idx -= it->counters[i] * it->tensor->stride[i];
		it->counters[i] = 0;
	}
	if (!increment) {
		it->idx = -1;
	}
}

void iterator_free(TensorIterator* it) {
	if (it->tensor->base) {
		free(it->counters);
	}
	free(it);
}
