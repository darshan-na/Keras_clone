#ifndef cnn_h
#define cnn_h

#include "layer.h"
#include "activations.h"
template <typename T,typename Activation>
class Convolution : public Layer<T>
{
public:
    Convolution(size_t input_channels, size_t output_channels, size_t kernel_size)
        : input_channels(input_channels), output_channels(output_channels), kernel_size(kernel_size),activation_function()
    {
        this->weights = Tensor<T>({input_channels, output_channels, kernel_size * kernel_size});
        this->bias = Tensor<T>({output_channels, 1, 1});
        weight_initialization(); 
        bias_initialization();
    }
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
    void forward(const Tensor<T> &input, Tensor<T> &output) override
    {
        size_t height = input.shape[0];
        size_t width = input.shape[1];
        size_t output_height = height - kernel_size + 1;
        size_t output_width = width - kernel_size + 1;
        output = Tensor<T>({output_height, output_width, output_channels});
        
        for (size_t c_out = 0; c_out < output_channels; ++c_out)
        {
            for (size_t y = 0; y < output_height; ++y)
            {
                for (size_t x = 0; x < output_width; ++x)
                {
                    T sum = 0;
                    for (size_t c_in = 0; c_in < input_channels; ++c_in)
                    {
                        for (size_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (size_t kx = 0; kx < kernel_size; ++kx)
                            {
                                sum += input(y + ky, x + kx, c_in) * this->weights(c_in, c_out, ky * kernel_size + kx);
                            }
                        }
                    }
                    sum += this->bias(c_out, 0, 0);
                    output(y, x, c_out) = sum;
                }
            }
        }
        activation_function.activation(output);
        std::cout << "Convolution output: " << std::endl;
        output.print();
    }

    void backward(Tensor<T> &input, Tensor<T> &output_gradient, Tensor<T> &input_gradient) override
    {

        activation_function.activation_derivative(output_gradient);
        size_t height = input.shape[0];
        size_t width = input.shape[1];
        size_t output_height = output_gradient.shape[0];
        size_t output_width = output_gradient.shape[1];

        // Compute input gradient
        input_gradient = Tensor<T>(input.shape);
        for (size_t c_in = 0; c_in < input_channels; ++c_in)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x)
                {
                    T sum = 0;
                    for (size_t c_out = 0; c_out < output_channels; ++c_out)
                    {
                        for (size_t ky = 0; ky < kernel_size; ++ky)
                        {
                            for (size_t kx = 0; kx < kernel_size; ++kx)
                            {
                                if (y >= ky && x >= kx && y - ky < output_height && x - kx < output_width)
                                {
                                    sum += this->weights(c_in, c_out, ky * kernel_size + kx) * output_gradient(y - ky, x - kx, c_out);
                                }
                            }
                        }
                    }
                    input_gradient(y, x, c_in) = sum;
                }
            }
        }

        // Compute gradients for weights and biases
        this->dw = Tensor<T>(Layer<T>::weights.shape);
        this->db = Tensor<T>(Layer<T>::bias.shape);
        for (size_t c_out = 0; c_out < output_channels; ++c_out)
        {
            for (size_t c_in = 0; c_in < input_channels; ++c_in)
            {
                for (size_t ky = 0; ky < kernel_size; ++ky)
                {
                    for (size_t kx = 0; kx < kernel_size; ++kx)
                    {
                        T weight_gradient_sum = 0;
                        for (size_t y = 0; y < output_height; ++y)
                        {
                            for (size_t x = 0; x < output_width; ++x)
                            {
                                weight_gradient_sum += input(y + ky, x + kx, c_in) * output_gradient(y, x, c_out);
                            }
                        }
                        this->dw(c_in, c_out, ky * kernel_size + kx) = weight_gradient_sum;
                    }
                }
            }

            // Compute bias gradient
            T bias_gradient_sum = 0;
            for (size_t y = 0; y < output_height; ++y)
            {
                for (size_t x = 0; x < output_width; ++x)
                {
                    bias_gradient_sum += output_gradient(y, x, c_out);
                }
            }
            this->db(c_out, 0, 0) = bias_gradient_sum;
        }
    }

    void update_weights(float learning_rate) override
    {
        size_t num_weights = Layer<T>::weights.shape[0] * Layer<T>::weights.shape[1] * Layer<T>::weights.shape[2];
        for (size_t i = 0; i < num_weights; ++i)
        {
            this->weights.data[i] -= learning_rate * Layer<T>::dw.data[i];
        }
    }

    void update_bias(float learning_rate) override
    {
        size_t num_biases = Layer<T>::bias.shape[0];
        for (size_t i = 0; i < num_biases; ++i)
        {
            this->bias.data[i] -= learning_rate * Layer<T>::db.data[i];
        }
    }


private:
    size_t input_channels;
    size_t output_channels;
    size_t kernel_size;
    Activation activation_function;
};
#endif

// template class Convolution<float,relu<float>>;