#ifndef flatten_h
#define flatten_h

#include "layer.h"

template <typename T>
class Flatten : public Layer<T>
{
public:
    void forward(const Tensor<T> &input, Tensor<T> &output) override
    {
        size_t input_dims = input.shape[0] * input.shape[1] * input.shape[2];

        output = Tensor<T>(vector<size_t>({input_dims,1,1}));
        size_t counter = 0;

        for (size_t c = 0; c < input.shape[2]; ++c)
        {
            for (size_t y = 0; y < input.shape[1]; ++y)
            {
                for (size_t x = 0; x < input.shape[0]; ++x)
                {
                    output(counter, 0, 0) = input(x, y, c);
                    ++counter;
                }
            }
        }
    }

    void backward(Tensor<T> &input, Tensor<T> &output_gradient, Tensor<T> &input_gradient) override
    {
        // Reshape the output_gradient to match the input shape
        input_gradient= Tensor<T>(input.shape);
        size_t counter = 0;

        for (size_t c = 0; c < input.shape[2]; ++c)
        {
            for (size_t y = 0; y < input.shape[1]; ++y)
            {
                for (size_t x = 0; x < input.shape[0]; ++x)
                {
                    input_gradient(x, y, c) = output_gradient(counter, 0, 0);
                    ++counter;
                }
            }
        }
    }

    void update_weights(float learning_rate) override
    {
        // Flatten layer does not have any learnable parameters, so no need to update weights
    }
    void update_bias(float learning_rate) override
    {
        // Flatten layer does not have any learnable parameters, so no need to update weights
    }
};

#endif
