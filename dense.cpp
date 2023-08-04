#include "dense.h"
#include "activations.h"
template <typename T,typename Activations>
void Dense<T,Activations>::forward(const Tensor<T> &input, Tensor<T> &output)
{
    // cout<<input.shape[0]<<' '<<input.shape[1]<<'  '<<input.shape[2]<<endl;
    // cout<<output.shape[0]<<' '<<output.shape[1]<<'  '<<output.shape[2]<<endl;
    for (size_t i = 0; i < this->output_size; i++)
    {
        for (size_t j = 0; j < this->M; j++)
        {
            T temp = 0;
            for (size_t k = 0; k < this->input_size; k++)
            {
                temp += (this->weights(i, k, 0) * input(k, j, 0));
            }
            output(i, j, 0) = temp + this->bias(i, 0, 0);
        }
    }
    act.activation(output);
}
template <typename T,typename Activations>
void Dense<T,Activations>::backward(Tensor<T> &input, Tensor<T> &output_gradient, Tensor<T> &input_gradient)
{
    act.activation_derivative(output_gradient); // dz calcuation
    
    // db == dz

    for(size_t i=0;i<this->output_size;i++)             
    {
        T sum =0;
        for(size_t j=0;j<this->M;j++)
        {
            sum+=output_gradient(i,j,0);
        }
        this->db(i,0,0) = sum;
    }
    for (size_t i = 0; i < this->input_size; i++)
    {
        for (size_t j = 0; j < this->M; j++)
        {
            T temp = 0;
            for (size_t k = 0; k < this->output_size; k++)
            {
                temp += output_gradient(k, j, 0) * this->weights(k, i, 0); // dx calculation
            }
            input_gradient(i, j, 0) = temp;
        }
    }

    for (size_t i = 0; i < this->input_size; i++)
    {
        for (size_t j = 0; j < this->output_size; j++)
        {
            T temp = 0;
            for (size_t k = 0; k < this->M; k++)
            {
                temp += output_gradient(j, k, 0) * input(i, k, 0); // dw calculation
            }
            this->dw(j, i, 0) = temp;
        }
    }
}

template <typename T,typename Activations>
void Dense<T,Activations>::update_weights(float learning_rate)
{
    for (size_t i = 0; i < this->output_size; i++)
    {
        for (size_t j = 0; j < this->input_size; j++)
        {
            this->weights(i, j, 0) = this->weights(i, j, 0) - (learning_rate * this->dw(i, j, 0));
        }
    }
}

template <typename T,typename Activations>
void Dense<T,Activations>::update_bias(float learning_rate)
{
    for (size_t i = 0; i < this->output_size; i++)
    {
            this->bias(i, 0, 0) = this->bias(i, 0, 0) - (learning_rate * this->db(i, 0, 0));
    }
}

template class Dense<float,sigmoid<float>>;
template class Dense<double,sigmoid<double>>;

template class Dense<float,relu<float>>;
template class Dense<double,relu<double>>;

template class Dense<float,softmax<float>>;
template class Dense<double,softmax<double>>;