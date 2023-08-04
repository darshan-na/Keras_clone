#ifndef activations_h
#define activations_h
#include "activation_interface.h"
using namespace std;
template<typename T>
class sigmoid : public activations<T>
{
public:
    void activation(Tensor<T> &output) override;
    void activation_derivative(Tensor<T> &output) override;
};

template<typename T>
class relu : public activations<T>
{
public:
    void activation(Tensor<T> &output) override;
    void activation_derivative(Tensor<T> &output) override;
};

template<typename T>
class softmax : public activations<T>
{
public:
    void activation(Tensor<T> &output) override;
    void activation_derivative(Tensor<T> &output) override;
};

template<typename T>
class tanh_ : public activations<T>
{
public:
    void activation(Tensor<T> &output) override;
    void activation_derivative(Tensor<T> &output) override;
};
#endif
