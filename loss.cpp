#include "loss.h"
#include <bits/stdc++.h>
using namespace std;
template<typename T>
class CategoricalCrossentropy : public LossFunction<T>
{
public:
    // loss = summation over n (-y*log(y_hat)) here n stands for the number of classes in the output 
    T compute(const Tensor<T> &predicted, const Tensor<T> &target) override
    {
        T loss = 0;
        for(size_t i =0; i<predicted.shape[0]; i++)
        {
            for(size_t j=0;j<predicted.shape[1];j++)
            {
                loss-=target(i,j,0) * log(predicted(i,j,0));
            }
        }
        this->computed_loss = loss/static_cast<T>(predicted.shape[1]);
        return abs(this->computed_loss);
    }
    void compute_gradient(const Tensor<T> &predicted, const Tensor<T> &target, Tensor<T> &gradient) override
    {
        for(size_t i =0;i<predicted.shape[0];i++)
        {
            for(size_t j=0;j<predicted.shape[1];j++)
            {
               gradient(i,j,0) =(target(i,j,0)/predicted(i,j,0));
            }
        }
    }
};


template<typename T>
class MeanSquaredError : public LossFunction<T> {
public:
T compute(const Tensor<T> &predicted, const Tensor<T> &target) override
    {
        T loss = 0;
        for(size_t i =0;i<predicted.shape[0];i++)
        {
            for(size_t j=0;j<predicted.shape[1];j++)
            {
                loss+=pow((predicted(i,j,0) - target(i,j,0)),2);
            }
        }
        return this->computed_loss = loss/static_cast<T>(predicted.shape[1]);

    }
    void compute_gradient(const Tensor<T> &predicted, const Tensor<T> &target, Tensor<T> &gradient) override
    {
        T temp = 0;
        for(size_t i =0;i<predicted.shape[0];i++)
        {
            for(size_t j=0;j<predicted.shape[1];j++)
            {
               temp = predicted(i,j,0) - target(i,j,0);
               gradient(i,j,0)=(2/static_cast<T>(predicted.shape[1]))*temp;
            }
        }
    }

};

template <typename T>
T BinaryCrossentropy<T>::compute(const Tensor<T> &predicted, const Tensor<T> &target)
{
    T loss = 0;
    for (size_t i = 0; i < predicted.shape[1]; i++)
    {
        loss -= target(0, i, 0) * log(predicted(0, i, 0)) + (1 - target(0, i, 0)) * log(1 - predicted(0, i, 0));
    }
    loss /= predicted.shape[1];
    // cout << "Predicted: " << predicted(0,0,0) << endl;
    // cout << "Target: " << target(0,0,0) << endl;
    // cout << "Loss: " << loss << endl;
    return loss;
}

template <typename T>
void BinaryCrossentropy<T>::compute_gradient(const Tensor<T> &predicted, const Tensor<T> &target, Tensor<T> &gradient)
{
    for (size_t i = 0; i < predicted.shape[1]; i++)
    {
        gradient(0, i, 0) = (predicted(0, i, 0) - target(0, i, 0)) / ((1 - (target(0, i, 0)) * predicted(0, i, 0)) + (target(0, i, 0) * (1 - predicted(0, i, 0))) + 1e-8);
        // cout << "Gradient: " << gradient(0,0,0) << endl;
    }
}

template<typename T>
unique_ptr<LossFunction<T>> LossFunction<T>::create_loss_function(const string& name)
{
    if(name == "CategoricalCrossentropy")
    {
        return make_unique<CategoricalCrossentropy<T>>();
    }
    else if(name=="MeanSquaredError")
    {
        return make_unique<MeanSquaredError<T>>();
    }
    else if(name=="BinaryCrossentropy")
    {
        return make_unique<BinaryCrossentropy<T>>();
    }
    else
    {
        throw runtime_error("No such loss function");
    }
}

template class CategoricalCrossentropy<float>;
template class MeanSquaredError<float>;
template class BinaryCrossentropy<float>;
template class BinaryCrossentropy<double>;
// comment the following lines if you also plan not to use double
template class CategoricalCrossentropy<double>;
template class MeanSquaredError<double>;

template unique_ptr<LossFunction<float>> LossFunction<float>::create_loss_function(const string& name);
template unique_ptr<LossFunction<double>> LossFunction<double>::create_loss_function(const string& name);