#ifndef dense_h
#define dense_h
#include "layer.h"
#include "tensor.h"
using namespace std;

template<typename T,typename Activations>
class Dense : public Layer<T>
{
    public:
        Dense(size_t input_size, size_t output_size,size_t M) : act() 
        {
            this->input_size = input_size;
            this->output_size = output_size;
            this->M = M;
            this->weights = Tensor<T>((vector<size_t>({output_size,input_size,1})));
            this->bias = Tensor<T>(vector<size_t>({output_size,1,1}));
            this->dw = Tensor<T>(vector<size_t>({output_size,input_size,1}));
            this->db = Tensor<T>((vector<size_t>{output_size,1,1}));
            weight_initialization(); 
            bias_initialization();
        }
        void forward(const Tensor<T> &input, Tensor<T> &output) override;
        void backward(Tensor<T> &input, Tensor<T> &output_gradient, Tensor<T> &input_gradient) override;
        void update_weights(float learning_rate) override;
        void update_bias(float learning_rate) override;
        void weight_initialization()
        {
            random_device rd;
            mt19937 gen(rd()); 
            std::normal_distribution<float> d(0,1);
            generate(this->weights.data.begin(),this->weights.data.end(),[&]()->T { return static_cast<T>((d(gen)*0.01));});
        }
        void bias_initialization()
        {
            fill(this->bias.data.begin(),this->bias.data.end(),0);
        }
    public:
        Activations act;        
};
#endif