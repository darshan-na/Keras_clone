#ifndef layer_h
#define layer_h
#include <cstddef>
#include "tensor.h"

template <typename T>
class Layer
{
public:
    virtual ~Layer() = default;
    virtual void forward(const Tensor<T> &input, Tensor<T> &output) = 0;
    virtual void backward(Tensor<T> &input, Tensor<T> &output_gradient, Tensor<T> &input_gradient) = 0;
    virtual void update_weights(float learning_rate) = 0;
    virtual void update_bias(float learning_rate) = 0;

public:
    size_t input_size;
    size_t output_size;
    size_t M;
    Tensor<T> weights;
    Tensor<T> bias;
    Tensor<T> dw;
    Tensor<T> db;
};
#endif