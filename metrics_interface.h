#ifndef METRIC_INTERFACE_H
#define METRIC_INTERFACE_H

#include "tensor.h"

template <typename T>
class Metric
{
public:
    virtual ~Metric() = default;
    virtual T compute(const Tensor<T> &predicted, const Tensor<T> &target) = 0;
protected:
    T computed_metric;
};

#endif

