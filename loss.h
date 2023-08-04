#ifndef loss_h
#define loss_h
#include "tensor.h"
#include <bits/stdc++.h>
template<typename T>
class LossFunction 
{
public:
    virtual ~LossFunction() = default;
    virtual T compute(const Tensor<T> &predicted, const Tensor<T> &target) = 0;
    virtual void compute_gradient(const Tensor<T> &predicted, const Tensor<T> &target, Tensor<T> &gradient) = 0;
    static std::unique_ptr<LossFunction<T>> create_loss_function(const std::string& name);
protected:
    T computed_loss;
    // Tensor<T>computed_gradient({predicted.shape[0],predicted.shape[1],1});
};
template <typename T>
class BinaryCrossentropy : public LossFunction<T>
{
public:
    T compute(const Tensor<T> &predicted, const Tensor<T> &target) override;
    void compute_gradient(const Tensor<T> &predicted, const Tensor<T> &target, Tensor<T> &gradient) override;
};
#endif
