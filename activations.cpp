#include "activations.h"
#include <limits>
using namespace std;
template <typename T>
void sigmoid<T>::activation(Tensor<T> &output)
{
    for (size_t j = 0; j < output.shape[1]; j++)
    {
        for (size_t i = 0; i < output.shape[0]; i++)
        {
            double temp = -1 * static_cast<double>(output(i, j, 0));
            output(i, j, 0) = static_cast<T>(1 / (1 + exp((temp))));
        }
    }
    this->my_output = Tensor<T>(output);    
}

template <typename T>
void sigmoid<T>::activation_derivative(Tensor<T> &output)
{
    for (size_t i = 0; i < output.shape[0]; i++)
    {
        for (size_t j = 0; j < output.shape[1]; j++)
        {
            T temp = this->my_output(i,j,0);
            output(i, j, 0) *= (temp * (1 - temp));
        }
    }
}

template <typename T>
void relu<T>::activation(Tensor<T> &output)
{
    for (size_t j = 0; j < output.shape[1]; j++)
    {
        for (size_t i = 0; i < output.shape[0]; i++)
        {
            output(i, j, 0) = max(static_cast<T>(0), output(i, j, 0));
        }
    }
    this->my_output = Tensor<T>(output);
}

template <typename T>
void relu<T>::activation_derivative(Tensor<T> &output)
{
    //  std::cout << "my_output shape: " << this->my_output.shape[0] << ", " << this->my_output.shape[1] << ", " << this->my_output.shape[2] << std::endl;
    //  std::cout << "Output shape: " << output.shape[0] << ", " << output.shape[1] << ", " << output.shape[2] << std::endl;
    for (size_t i = 0; i < output.shape[0]; i++)
    {
        for (size_t j = 0; j < output.shape[1]; j++)
        {
            T x = this->my_output(i, j, 0);
            if (x <=0)
            {
                output(i, j, 0) = 0;
            }
            else
            {
                output(i, j, 0) *= 1;
            }
        }
    }
}

template <typename T>
void softmax<T>::activation(Tensor<T> &output)
{
    Tensor<T> temp(vector<size_t>({1, output.shape[1], 1}));
    for (size_t j = 0; j < output.shape[1]; j++)
    {
        for (size_t i = 0; i < output.shape[0]; i++)
        {
            temp(0, j, 0) = temp(0, j, 0) + exp(output(i, j, 0));
        }
    }
    for (size_t j = 0; j < output.shape[1]; j++)
    {
        for (size_t i = 0; i < output.shape[0]; i++)
        {
            output(i, j, 0) = exp(output(i, j, 0)) / temp(0, j, 0);
        }
    }
    this->my_output = Tensor<T>(output);
}

template <typename T>
void softmax<T>::activation_derivative(Tensor<T> &output) 
{
    Tensor<T>computed_gradient(vector<size_t>({output.shape[0],output.shape[1],1}));
    for (size_t i = 0; i < output.shape[1]; ++i)
    {
        for (size_t j = 0; j < output.shape[0]; ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < output.shape[0]; ++k)
            {
                T softmax_derivative = (j == k) ? this->my_output(k,i,0) * (1.0 - this->my_output(k,i,0)) : -this->my_output(k,i,0) * this->my_output(j,i,0);
                sum += (output(k,i,0) * softmax_derivative);
            }
            computed_gradient(j,i,0) = sum;
        }
    }
    output = computed_gradient;
}

template<typename T>
void tanh_<T>::activation(Tensor<T>& output)
{
    for(size_t i=0; i<output.shape[1]; i++)
    {
        for(size_t j=0; j<output.shape[0]; j++)
        {
            output(j,i,0) = tanh(output(j,i,0));
        }
    }
    this->my_output = Tensor<T>(output);
}

template<typename T>
void tanh_<T>:: activation_derivative(Tensor<T>& output)
{
    for (size_t i = 0; i < output.shape[0]; i++)
    {
        for (size_t j = 0; j < output.shape[1]; j++)
        {
            T temp = this->my_output(i,j,0);
            output(i, j, 0) *= (1 - pow(temp,2));
        }
    }
}

template class relu<float>;
template class relu<double>;

template class sigmoid<float>;
template class sigmoid<double>;

template class softmax<float>;
template class softmax<double>;

template class tanh_<float>;
template class tanh_<double>;