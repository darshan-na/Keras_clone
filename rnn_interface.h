#ifndef rnn_interface_h
#define rnn_interface_h
#include "layer.h"
template <typename T>
class Rnn : public Layer<T>
{
public:
    virtual ~Rnn() = default;
    void forward(const Tensor<T> &input, Tensor<T> &output) override
    {
    }
    void backward(Tensor<T> &input, Tensor<T> &output_gradient, Tensor<T> &input_gradient) override
    {
    }

    virtual void forward(const Tensor<T> &input, Tensor<T> &a_prev, Tensor<T> &a_next, Tensor<T> &yt_out) = 0;
    virtual void backward(Tensor<T> &input, Tensor<T> &a_prev, Tensor<T> &curr_a, Tensor<T> &output_grad, Tensor<T> &a_next_grad, Tensor<T> &input_grad, Tensor<T> &a_prev_grad) = 0;

public:
    size_t M;
    size_t n_a;
    size_t n_x;
    size_t n_y;
    Tensor<T> w_x;
    Tensor<T> w_a;
    Tensor<T> w_y;
    Tensor<T> dw_x;
    Tensor<T> dw_y;
    Tensor<T> dw_a;
    Tensor<T> db_a;
    Tensor<T> db_y;
    Tensor<T> b_a;
    Tensor<T> b_y;

};
#endif