#ifndef activation_interface_h
#define activation_interface_h
using namespace std;
#include<bits/stdc++.h>
#include "tensor.h"

template<typename T>
class activations
{
public:
    virtual ~activations() {}
    virtual void activation(Tensor<T> &output) = 0;
    virtual void activation_derivative(Tensor<T>& output) = 0;
public:
     Tensor<T>my_output;
};
#endif