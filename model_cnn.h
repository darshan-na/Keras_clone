#ifndef model_cnn_h
#define model_cnn_h
#include "model_interface.h"
#include "cnn.h"
#include "dense.h"
#include "flatten.h"
using namespace std;
template <typename T, typename Activation, typename Activation_Dense>
class Model_CNN : public Model<T>
{
public:
    void add_layer(Layer<T> *layer) override
    {
        layers.push_back(layer);
    }
    void compile(const string &loss) override
    {
        loss_fn = LossFunction<T>::create_loss_function(loss);
    }
    void fit(const Tensor<T> &input, const Tensor<T> &target, size_t batch_size, size_t epochs, float learning_rate) override
    {
        
        for (size_t i = 0; i < layers.size(); i++)
        {
            if (typeid(*layers[i]) == typeid(Convolution<T, Activation>))
            {
                conv_layers.push_back(dynamic_cast<Convolution<T, Activation> *>(layers[i]));
            }
            else if (typeid(*layers[i]) == typeid(Flatten<T>))
            {
                flat_layer = dynamic_cast<Flatten<T> *>(layers[i]);
            }
            else
            {
                dense_layers.push_back(layers[i]);
            }
        }
        for (size_t epoch = 1; epoch <= epochs; epoch++)
        {
            size_t count = 0;
            size_t batch_count = (size_t)ceil(input.shape[2] / batch_size);
            size_t j = 0;
            while (count < batch_count)
            {
                // preparing the batch input
                cnn_layer_inputs.clear();
                dense_layer_inputs.clear();
                Tensor<T> flat_layer_input;
                Tensor<T> batch_input(vector<size_t>({input.shape[0], input.shape[1], batch_size}));
                Tensor<T> batch_target(vector<size_t>({target.shape[0], batch_size, 1}));
                for (size_t k = j; k < (j + batch_size) && k < input.shape[2]; k++)
                {
                    for (size_t height = 0; height < input.shape[0]; height++)
                    {
                        for (size_t width = 0; width < input.shape[1]; width++)
                        {
                            batch_input(height, width, k - j) = input(height, width, k);
                        }
                    }
                }
                for (size_t k = j; k < (j + batch_size) && k < input.shape[2]; k++)
                {
                    for (size_t target_dim = 0; target_dim < target.shape[0]; target_dim++)
                    {
                        batch_target(target_dim, k - j, 0) = target(target_dim, k,0);
                    }
                }

                // forward propagation

                Tensor<T>layer_output;
                for (size_t training_ex = 0; training_ex < batch_input.shape[2]; training_ex++)
                {
                    Tensor<T> current_input(vector<size_t>({input.shape[0], input.shape[1], 1}));
                    for(size_t height = 0; height<input.shape[0];height++)
                    {
                        for(size_t width = 0; width<input.shape[0];width++)
                        {
                            current_input(height,width,0) = batch_input(height,width,training_ex);
                        }
                    }
                    for (size_t i = 0; i < conv_layers.size(); i++)
                    {
                        Convolution<T, Activation> *layer = conv_layers[i];
                        cnn_layer_inputs.push_back(current_input);
                        Tensor<T> current_output;
                        layer->forward(current_input, current_output);
                        current_input = current_output;
                    }
                    Tensor<T> flattened_output;
                    flat_layer_input = current_input;
                    flat_layer->forward(current_input, flattened_output);
                    Tensor<T>layer_input(flattened_output);
                    Tensor<T>layer_output= Tensor<T>(vector<size_t>({dense_layers[0]->output_size, batch_size, 1}));
                    for (size_t x = 0; x < dense_layers.size(); x++)
                    {
                        Layer<T> *layer = dense_layers[x];
                        dense_layer_inputs.push_back(layer_input);
                        layer->forward(layer_input, layer_output);
                        layer_input = layer_output;
                        if (x == dense_layers.size() - 1)
                        {
                            break;
                        }
                        else
                        {
                            layer_output = Tensor<T>(vector<size_t>({dense_layers[x + 1]->output_size, batch_size, 1}));
                        }
                    }
                }

                //loss computation
                T loss = loss_fn->compute(layer_output, batch_target);
                Accuracy<float> a;
                cout << "Epoch : " << epoch << "/" << epochs << ".........."
                     << "loss : " << loss <<".......Accuracy : "<<a.compute(layer_output,batch_target)<<endl;

                Tensor<T> gradient(vector<size_t>({layer_output.shape[0], layer_output.shape[1], 1}));
                loss_fn->compute_gradient(layer_output, batch_target, gradient); // dy_hat calculation or da_last_layer
                
                //backprop thro dense layers
                Tensor<T> layer_input;
                for (size_t y = dense_layers.size() - 1; y != 0; y--)
                {
                    Layer<T> *layer = dense_layers[y];
                    layer_input = dense_layer_inputs[y];
                    if (y == 0)
                    {
                        break;
                    }
                    else
                    {
                        input_gradient = Tensor<T>(vector<size_t>({dense_layers[y - 1]->output_size, batch_size, 1}));
                    }
                    layer->backward(layer_input, gradient, input_gradient);
                    // weight updation
                    layer->update_weights(learning_rate);
                    layer->update_bias(learning_rate);
                    gradient = input_gradient;
                }
                Layer<T> *layer = dense_layers[0];
                layer_input = dense_layer_inputs[0];
                input_gradient = Tensor<T>(vector<size_t>({layer_input.shape[0], batch_size, 1}));
                layer->backward(layer_input, gradient, input_gradient);
                layer->update_weights(learning_rate);
                layer->update_bias(learning_rate);
                gradient = input_gradient;
                Tensor<T> conv_out_grad;
                flat_layer->backward(flat_layer_input,gradient,conv_out_grad);
                gradient = conv_out_grad;
                // back prop thro conv layers
                for(int y = conv_layers.size()-1;y>=0;y--)
                {
                    Convolution<T, Activation> * layer = conv_layers[y];
                    Tensor<T> input_gradient;
                    Tensor<T> curr_layer_input = cnn_layer_inputs[y];
                    layer->backward(curr_layer_input,gradient,input_gradient);
                    gradient = input_gradient;
                    layer->update_weights(learning_rate);
                    layer->update_bias(learning_rate);
                }
                j = j + batch_size;
                count++;
            }
        }
    }
     void predict(const Tensor<T> &input, Tensor<T> &output) override
     {

     }

public:
    Flatten<T> *flat_layer;
    vector<Convolution<T, Activation> *> conv_layers;
    
    // vector<Dense<T, Activation_Dense> *> dense_layers;
    vector<Layer<T> *> dense_layers;
    vector<Layer<T> *> layers;
    Tensor<T> input_gradient;
    vector<Tensor<T>> cnn_layer_inputs;
    vector<Tensor<T>> dense_layer_inputs;
    unique_ptr<LossFunction<T>> loss_fn;
};
#endif
template class Model_CNN<float, relu<float>, sigmoid<float>>;