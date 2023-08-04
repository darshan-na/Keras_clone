#include "vanilla_rnn.h"
#include "activations.h"

template<typename T,typename Activation,typename output_activation>
void vanilla_Rnn<T,Activation,output_activation>::weight_initialization()
{
    random_device rd;
    mt19937 gen(rd()); 
    std::normal_distribution<float> d(0,1);
    generate(this->w_x.data.begin(),this->w_x.data.end(),[&]()->T { return static_cast<T>((d(gen)*0.01));});
    generate(this->w_y.data.begin(),this->w_y.data.end(),[&]()->T { return static_cast<T>((d(gen)*0.01));});
    generate(this->w_a.data.begin(),this->w_a.data.end(),[&]()->T { return static_cast<T>((d(gen)*0.01));});
}

template<typename T,typename Activation,typename output_activation>
void vanilla_Rnn<T,Activation,output_activation>::bias_initialization()
{
    fill(this->b_a.data.begin(),this->b_a.data.end(),0);
    fill(this->b_y.data.begin(),this->b_y.data.end(),0);
}

template<typename T,typename Activation,typename output_activation>
void vanilla_Rnn<T,Activation,output_activation>::forward(const Tensor<T> &input, Tensor<T> &a_prev, Tensor<T> &a_next, Tensor<T> &yt_out)
{
    // a_next 1st half caluclation
    Tensor<T>a_1 = (this->w_x).dot(input);
    Tensor<T>a_2 = (this->w_a).dot(a_prev);
    a_next = a_1 + a_2;
    for(size_t i=0;i<this->n_a;i++)
    {
        for(size_t j=0;j<this->M;j++)
        {
            a_next(i,j,0) = a_next(i,j,0) + this->b_a(i,0,0);
        }
    }
    act.activation(a_next);
    yt_out = (this->w_y).dot(a_next);
    for(size_t i=0;i<this->n_y;i++)
    {
        for(size_t j=0;j<this->M;j++)
        {
            yt_out(i,j,0) = yt_out(i,j,0) + this->b_y(i,0,0);
        }
    }
}

template<typename T,typename Activation,typename output_activation>
void vanilla_Rnn<T,Activation,output_activation>::backward(Tensor<T> &input, Tensor<T> &a_prev, Tensor<T> &curr_a, Tensor<T> &output_grad, Tensor<T> &a_next_grad, Tensor<T> &input_grad, Tensor<T> &a_prev_grad)
{
    //dz_y calculation

    act.activation_derivative(output_grad);

    //db_y calculation
    this->db_y = output_grad.sum();

    //gradient prop from ouptut of the current layer
    Tensor<T>w_yt = (this->w_y).transpose();
    Tensor<T>ouptut_grad_prop = w_yt.dot(output_grad);

    //dw_y calculation
    Tensor<T> curr_a_t = curr_a.transpose();
    this->dw_y = output_grad.dot(curr_a_t);

    Tensor<T> a_net_grad = ouptut_grad_prop + a_next_grad;

    act.activation_derivative(a_net_grad);

    //db_a calculation
    this->db_a = a_net_grad.sum();

    // input grad calculation
    Tensor<T>w_xt = (this->w_x).transpose();
    input_grad = w_xt.dot(a_net_grad);

    // dw_x calculation
    Tensor<T> input_t = input.transpose();
    this->dw_x = a_net_grad.dot(input_t);

    // a_prev_grad calculation
    Tensor<T>w_at = (this->w_a).transpose();
    a_prev_grad = w_at.dot(a_net_grad);

    //dw_a calculation
    Tensor<T> a_prev_t = a_prev.transpose();
    this->dw_a = a_net_grad.dot(a_prev_t);
}

template<typename T,typename Activation,typename output_activation>
void vanilla_Rnn<T,Activation,output_activation>::update_weights(float learning_rate)
{
    Tensor<T> temp_weigths = this->dw_a;

    for (size_t i = 0; i < this->n_y; i++)
    {
        for (size_t j = 0; j < this->n_a; j++)
        {
            this->w_y(i, j, 0) = this->w_y(i, j, 0) - (learning_rate * this->dw_y(i, j, 0));
        }
    }
    for (size_t i = 0; i < this->n_a; i++)
    {
        for (size_t j = 0; j < this->n_x; j++)
        {
            this->w_x(i, j, 0) = this->w_x(i, j, 0) - (learning_rate * this->dw_x(i, j, 0));
        }
    }
    for (size_t i = 0; i < this->n_a; i++)
    {
        for (size_t j = 0; j < this->n_a; j++)
        {
            this->w_a(i, j, 0) = this->w_a(i, j, 0) - (learning_rate * this->dw_a(i, j, 0));
        }
    }

    // if(temp_weigths.data==this->dw_a.data)
    // {
    //     cout<<"failed"<<endl;
    // }
    // else
    // {
    //     cout<<"success"<<endl;
    // }
}

template<typename T,typename Activation,typename output_activation>
void vanilla_Rnn<T,Activation,output_activation>::update_bias(float learning_rate)
{
    for (size_t i = 0; i < this->n_a; i++)
    {
        this->b_a(i, 0, 0) = this->b_a(i, 0, 0) - (learning_rate * this->db_a(i, 0, 0));
    }

    for (size_t i = 0; i < this->n_y; i++)
    {
        this->b_y(i, 0, 0) = this->b_y(i, 0, 0) - (learning_rate * this->db_y(i, 0, 0));
    }
}

template class vanilla_Rnn<float, tanh_<float>, softmax<float>>;
template class vanilla_Rnn<double, tanh_<double>, softmax<double>>;