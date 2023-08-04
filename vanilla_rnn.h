#ifndef vanilla_rnn_h
#define vanilla_rnn_h

#include "rnn_interface.h"

template<typename T,typename Activation,typename output_activation>
class vanilla_Rnn : public Rnn<T> 
{
public:
    vanilla_Rnn(size_t n_a,size_t n_x,size_t n_y,size_t M) : act(),act_out()
    {
        this->n_a = n_a;
        this->n_y = n_y;
        this->n_x = n_x;
        this->M = M;
        this->w_x = Tensor<T>(vector<size_t>({this->n_a,this->n_x,1}));
        this->dw_x = Tensor<T>(vector<size_t>({this->n_a,this->n_x,1}));
        this->w_a = Tensor<T>(vector<size_t>({this->n_a,this->n_a,1}));
        this->dw_a = Tensor<T>(vector<size_t>({this->n_a,this->n_a,1}));
        this->w_y = Tensor<T>(vector<size_t>({this->n_y,this->n_a,1}));
        this->dw_y = Tensor<T>(vector<size_t>({this->n_y,this->n_a,1}));
        this->b_y = Tensor<T>(vector<size_t>({this->n_y,1,1}));
        this->db_y = Tensor<T>(vector<size_t>({this->n_y,1,1}));
        this->b_a = Tensor<T>(vector<size_t>({this->n_a,1,1}));
        this->db_a = Tensor<T>(vector<size_t>({this->n_a,1,1}));
        weight_initialization();
        bias_initialization();
    }
    void forward(const Tensor<T> &input, Tensor<T> &a_prev, Tensor<T> &a_next, Tensor<T> &yt_out) override;
    void backward(Tensor<T> &input, Tensor<T> &a_prev, Tensor<T> &curr_a, Tensor<T> &output_grad, Tensor<T> &a_next_grad, Tensor<T> &input_grad, Tensor<T> &a_prev_grad) override;
    void update_weights(float learning_rate) override;
    void update_bias(float learning_rate) override;
    void weight_initialization();
    void bias_initialization();
public:
    Activation act;
    output_activation act_out;
};
#endif