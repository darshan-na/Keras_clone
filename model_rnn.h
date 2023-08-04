#ifndef model_rnn_h
#define model_rnn_h
#include "model_interface.h"
#include "rnn_interface.h"
using namespace std;
template <typename T>
class Model_RNN : public Model<T>
{
public:
    // void words(vector<Tensor<T>> curr_layer_x_output)
    // {
    //     srand(time(0));
    //     int size = curr_layer_x_output[0].shape[1];
    //     size_t ran = (size_t) rand()%size;
    //     for(size_t i=0;i<curr_layer_x_output.size();i++)
    //     {
    //         Tensor<T> cur_time_tensor = curr_layer_x_output[i];
    //         Tensor<T> tensor_copy = cur_time_tensor;
    //         transform(tensor_copy.data.begin(), tensor_copy.data.end(), tensor_copy.data.begin(), [](T x) -> T { return round(x); });
    //         for(int )
    //     }
    // }    
    void fetch_rnn_cell_input(Tensor<T> &one_time_step_input, const Tensor<T> &batch_input, size_t time_step)
    {
        for (size_t i = 0; i < batch_input.shape[0]; i++)
        {
            for (size_t j = 0; j < batch_input.shape[1]; j++)
            {
                one_time_step_input(i, j, 0) = batch_input(i, j, time_step);
            }
        }
    }
    void add_layer(Layer<T> *layer) override
    {
        Rnn<T> *rnn_layer = dynamic_cast<Rnn<T> *>(layer);
        if (rnn_layer)
        {
            layers.push_back(rnn_layer);
        }
        else
        {
            throw runtime_error("RNN model cannot have a non Rnn layer");
        }
    }
    void compile(const string &loss) override
    {
        loss_fn = LossFunction<T>::create_loss_function(loss);
    }
    void fit(const Tensor<T> &input, const Tensor<T> &target, size_t batch_size, size_t epochs, float learning_rate) override
    {
        for (size_t i = 1; i <= epochs; i++)
        {
            
            size_t count = 0;
            // cout<<input.shape[1]<<endl;
            size_t batch_count = (size_t)ceil(input.shape[1] / batch_size);
            
            size_t j = 0;
            losses.clear();
            while (count < 50)
            {
                // cout<<"hi"<<endl;
                Tensor<T> batch_input(vector<size_t>({input.shape[0], batch_size, input.shape[2]}));
                Tensor<T> batch_target(vector<size_t>({target.shape[0], batch_size, target.shape[2]}));
                batch_targets_vector.clear();
                all_layer_x_inputs.clear();
                all_layer_a_prev_inputs.clear();
                all_layer_curr_a_inputs.clear();
                curr_layer_x_inputs.clear();
                // fetch the current batch
                for (size_t k = j; k < (j + batch_size) && k < input.shape[1]; k++)
                {
                    for (size_t l = 0; l < input.shape[0]; l++)
                    {
                        for (size_t m = 0; m < input.shape[2]; m++)
                        {
                            batch_input(l, k-j, m) = input(l, k, m);
                            batch_target(l, k - j, m) = target(l, k, m);
                        }
                    }
                }

                for (size_t time_step = 0; time_step < input.shape[2]; time_step++)
                {
                    Tensor<T> one_time_step_input(vector<size_t>({batch_input.shape[0], batch_input.shape[1], 1}));
                    fetch_rnn_cell_input(one_time_step_input, batch_input, time_step);
                    curr_layer_x_inputs.push_back(one_time_step_input);
                }

                // forward
                for (size_t curr_layer = 0; curr_layer < layers.size(); curr_layer++)
                {
                    // for every layer in the model
                    all_layer_x_inputs.push_back(curr_layer_x_inputs);
                    Rnn<T> *layer = layers[curr_layer];
                    curr_layer_a_prev_inputs.clear();
                    curr_layer_curr_a_inputs.clear();
                    Tensor<T> a_prev(vector<size_t>({layer->n_a, batch_size, 1}));

                    for (size_t time_step = 0; time_step < input.shape[2]; time_step++)
                    {
                        // for every time step in a layer
                        Tensor<T> curr_layer_curr_timestep = curr_layer_x_inputs[time_step];
                        Tensor<T> yt_out(vector<size_t>({layer->n_y, batch_size, 1}));
                        Tensor<T> a_next(vector<size_t>({layer->n_a, batch_size, 1}));
                        layer->forward(curr_layer_curr_timestep, a_prev, a_next, yt_out);
                        curr_layer_a_prev_inputs.push_back(a_prev);
                        curr_layer_curr_a_inputs.push_back(a_next);
                        a_prev = a_next;
                        curr_layer_x_output.push_back(yt_out);
                    }
                    all_layer_x_outputs.push_back(curr_layer_x_output);
                    curr_layer_x_inputs = curr_layer_x_output;
                    all_layer_a_prev_inputs.push_back(curr_layer_a_prev_inputs);
                    all_layer_curr_a_inputs.push_back(curr_layer_curr_a_inputs);
                }
                // words(curr_layer_x_output);

                // loss calculation
                T loss = 0;
                for (size_t time_step = 0; time_step < input.shape[2]; time_step++)
                {
                    Tensor<T> curr_time_output = curr_layer_x_output[time_step];
                    Tensor<T> curr_time_target(vector<size_t>({batch_target.shape[0], batch_target.shape[1], 1}));
                    fetch_rnn_cell_input(curr_time_target, batch_target, time_step);
                    batch_targets_vector.push_back(curr_time_target);
                    loss += loss_fn->compute(curr_time_output, curr_time_target);
                }
                loss /= input.shape[2];
                losses.push_back(loss);

                

                // loss gradient calculation
                for (size_t time_step = 0; time_step < input.shape[2]; time_step++)
                {
                    Tensor<T> curr_time_output = curr_layer_x_output[time_step];
                    Tensor<T> curr_time_target = batch_targets_vector[time_step];
                    Tensor<T> curr_time_gradient(vector<size_t>({curr_time_output.shape[0], curr_time_output.shape[1], 1}));
                    loss_fn->compute_gradient(curr_time_output,curr_time_target,curr_time_gradient);
                    layer_output_gradients.push_back(curr_time_gradient);
                    // vector<T> temp = curr_time_output.data;
                    // vector<T> temp1 = curr_time_target.data;
                    // for(int i=0;i<temp.size();i++)
                    // {
                    //     curr_time_gradient.data[i] = (temp[i] - temp1[i]);
                    // }
                    // 
                }

                // backprop implementation
                for (int curr_layer = (int)layers.size() - 1; curr_layer >= 0; curr_layer--)
                {
                    // for each layer
                    size_t index = (size_t)curr_layer;
                    Rnn<T> *layer = layers[index];
                    curr_layer_a_prev_inputs = all_layer_a_prev_inputs[index];
                    curr_layer_curr_a_inputs = all_layer_curr_a_inputs[index];
                    curr_layer_x_inputs = all_layer_x_inputs[index];

                    // cout<<"hello"<<endl;

                    Tensor<T> a_next_grad(vector<size_t>({layer->n_a, batch_size, 1}));
                    for (size_t time_step = 0; time_step < input.shape[2]; time_step++)
                    {
                        
                        Tensor<T> a_prev_grad(vector<size_t>({layer->n_a, batch_size, 1}));
                        Tensor<T> curr_time_input_grad(vector<size_t>({layer->n_x, batch_size, 1}));
                        Tensor<T> curr_time_input = curr_layer_x_inputs[time_step];
                        Tensor<T> curr_time_aprev = curr_layer_a_prev_inputs[time_step];
                        Tensor<T> curr_time_curra = curr_layer_curr_a_inputs[time_step];
                        Tensor<T> curr_time_output_grad = layer_output_gradients[time_step];
                        layer->backward(curr_time_input, curr_time_aprev, curr_time_curra, curr_time_output_grad, a_next_grad, curr_time_input_grad, a_prev_grad);
                        a_next_grad = a_prev_grad;
                        layer_input_gradients.push_back(curr_time_input_grad);
                        layer->update_weights(learning_rate);
                        layer->update_bias(learning_rate);
                    }
                    layer_output_gradients = layer_input_gradients;
                }

                
                count++;
                // cout<<count<<endl;
                j = j+batch_size;
            }
            T temp = std::accumulate(losses.begin(), losses.end(), 0) / losses.size();
            cout << "Epoch : " << i << "/" << epochs << ".........."<< "loss : " << temp <<endl;
            
        }
    }
    void predict(const Tensor<T> &input, Tensor<T> &output) override
    {
        for (size_t time_step = 0; time_step < input.shape[2]; time_step++)
        {
            Tensor<T> one_time_step_input(vector<size_t>({input.shape[0], input.shape[1], 1}));
            fetch_rnn_cell_input(one_time_step_input, input, time_step);
            curr_layer_x_inputs.push_back(one_time_step_input);
        }
        for (size_t curr_layer = 0; curr_layer < layers.size(); curr_layer++)
        {
            // for every layer in the model
            Rnn<T> *layer = layers[curr_layer];
            curr_layer_x_output.clear();
            Tensor<T> a_prev(vector<size_t>({layer->n_a, input.shape[1], 1}));
            for (size_t time_step = 0; time_step < input.shape[2]; time_step++)
            {
                // for every time step in a layer
                Tensor<T> curr_layer_curr_timestep = curr_layer_x_inputs[time_step];
                Tensor<T> yt_out(vector<size_t>({layer->n_y, input.shape[1], 1}));
                Tensor<T> a_next(vector<size_t>({layer->n_a, input.shape[1], 1}));
                layer->forward(curr_layer_curr_timestep, a_prev, a_next, yt_out);
                a_prev = a_next;
                curr_layer_x_output.push_back(yt_out);
            }
            curr_layer_x_inputs = curr_layer_x_output;
        }
    }

public:
    vector<Rnn<T> *> layers;
    vector<T>losses;
    vector<Tensor<T>> batch_targets_vector;
    vector<vector<Tensor<T>>> all_layer_x_inputs;
    vector<vector<Tensor<T>>> all_layer_x_outputs;
    vector<vector<Tensor<T>>> all_layer_a_prev_inputs;
    vector<vector<Tensor<T>>> all_layer_curr_a_inputs;
    vector<Tensor<T>> curr_layer_x_inputs;
    vector<Tensor<T>> curr_layer_x_output;
    vector<Tensor<T>> curr_layer_a_prev_inputs;
    vector<Tensor<T>> curr_layer_curr_a_inputs;
    vector<Tensor<T>> layer_output_gradients;
    vector<Tensor<T>> layer_input_gradients;
    unique_ptr<LossFunction<T>> loss_fn;
};
#endif