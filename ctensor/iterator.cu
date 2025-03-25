#include "iterator.h"

TensorIterator* iterator_create(Tensor* tensor) {
	TensorIterator* it = (TensorIterator*) malloc(sizeof(TensorIterator));
	it->tensor = tensor;
	it->counters = (int32_t*) malloc(sizeof(int32_t) * tensor->dims);
	it->idx = 0;
	for (int32_t i=0; i<tensor->dims; i++) {
		it->counters[i] = 0;
	}
	return it;
}

bool iterator_has_next(TensorIterator* it) {
	for (int32_t i=0; i<it->tensor->dims; i++) {
		if (it->counters[i] != it->tensor->shape[i] - 1) {
			return false;
		}
	}
	return true;
}

int32_t iterator_next(TensorIterator* it) {
	for (int32_t i=it->tensor->dims - 1; i>=0; i--) {
		if (it->counters[i] + 1 < it->tensor->shape[i]) {
			it->counters[i]++;
			it->idx += it->tensor->stride[i];
			break;
		}
		it->counters[i] = 0;
	}
	return it->idx;
}

void iterator_free(TensorIterator* it) {
	free(it->counters);
	free(it);
}
