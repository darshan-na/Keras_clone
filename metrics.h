#ifndef metrics_h
#define metrics_h
#include "metrics_interface.h" 

template <typename T>
class Accuracy : public Metric<T>
{
public:
    T compute(const Tensor<T> &predicted, const Tensor<T> &target) override;
};

#endif