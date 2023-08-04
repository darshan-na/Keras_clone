#ifndef model_dnn_h
#define model_dnn_h
#include "model_interface.h"
using namespace std;
template <typename T>
class Model_DNN : public Model<T>
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
        Tensor<T> batch_input(vector<size_t>({input.shape[0], batch_size, 1}));
        // cout<<input.shape[0]<<' '<<input.shape[1]<<'  '<<endl;
        for (size_t i = 1; i <= epochs; i++)
        {
            // batch calculations
            size_t count = 0;
            size_t batch_count = ceil(input.shape[1] / batch_size);
            size_t j = 0;
            while (count < batch_count)
            {
                all_layer_inputs.clear();
                // current batch formation
                for (size_t k = j; k < (j + batch_size) && k < input.shape[1]; k++)
                {
                    for (size_t l = 0; l < input.shape[0]; l++)
                    {
                        batch_input(l, k -j, 0) = input(l, k, 0);
                    }
                }

                // forward
                Tensor<T>layer_input(batch_input);
                Tensor<T>layer_output(vector<size_t>({layers[0]->output_size, batch_size, 1}));
                for (size_t x = 0; x < layers.size(); x++)
                {
                    // cout<<layer_input.shape[0]<<' '<<layer_input.shape[1]<<'  '<<layer_input.shape[2]<<endl;
                    Layer<T> *layer = layers[x];
                    all_layer_inputs.push_back(layer_input);
                    layer->forward(layer_input, layer_output);
                    layer_input = layer_output;
                    if (x == layers.size() - 1)
                    {
                        break;
                    }
                    else
                    {
                        layer_output = Tensor<T>(vector<size_t>({layers[x + 1]->output_size, batch_size, 1}));
                    }
                }
                
                // error calculation and accuracy computation
                T loss = loss_fn->compute(layer_output, target);
                Accuracy<float> a;
                cout << "Epoch : " << i << "/" << epochs << ".........."
                     << "loss : " << loss <<".......Accuracy : "<<a.compute(layer_output,target)<<endl;

                // cout << "Epoch : " << i << "/" << epochs << ".........."
                //      << "loss : " << loss <<endl;

                Tensor<T> gradient(vector<size_t>({layer_output.shape[0], layer_output.shape[1], 1}));
                loss_fn->compute_gradient(layer_output, target, gradient); // dy_hat calculation or da_last_layer

                // back propagation
                for (size_t y = layers.size() - 1; y != 0; y--)
                {
                    Layer<T> *layer = layers[y];
                    layer_input = all_layer_inputs[y];
                    if (y == 0)
                    {
                        break;
                    }
                    else
                    {
                        input_gradient = Tensor<T>(vector<size_t>({layers[y - 1]->output_size, batch_size, 1}));
                    }
                    layer->backward(layer_input, gradient, input_gradient);
                    // weight updation
                    layer->update_weights(learning_rate);
                    layer->update_bias(learning_rate);
                    gradient = input_gradient;
                }
                Layer<T> *layer = layers[0];
                layer_input = all_layer_inputs[0];
                input_gradient = Tensor<T>(vector<size_t>({layer_input.shape[0], batch_size, 1}));
                layer->backward(layer_input, gradient, input_gradient);
                layer->update_weights(learning_rate);
                layer->update_bias(learning_rate);
                j = j + batch_size;
                count++;
            }
        }
    }
    void predict(const Tensor<T> &input, Tensor<T> &output) override
    {
        Tensor<T> layer_output_test(vector<size_t>({layers[0]->output_size, input.shape[1], 1}));
        Tensor<T> layer_input_test = input;
        for (size_t x = 0; x < layers.size(); x++)
        {
            Layer<T> *layer = layers[x];
            layer->forward(layer_input_test, layer_output_test);
            layer_input_test = layer_output_test;
            if (x == layers.size() - 1)
            {
                break;
            }
            else
            {
                layer_output_test = Tensor<T>(vector<size_t>({layers[x + 1]->output_size, input.shape[1], 1}));
            }
        }
        output = layer_output_test;
    }

public:
    vector<Layer<T> *> layers;
    Tensor<T> input_gradient;
    vector<Tensor<T>> all_layer_inputs;
    unique_ptr<LossFunction<T>> loss_fn;
};
#endif