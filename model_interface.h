#ifndef model_interface
#define model_interface
#include "layer.h"
#include "tensor.h"
#include "loss.h"
#include "metrics.h"
using namespace std;
template<typename T>
class Model
{
public:

    virtual void add_layer(Layer<T> *layer) = 0;
    virtual void compile(const string &loss) = 0;
    virtual void fit(const Tensor<T> &input, const Tensor<T> &target, size_t batch_size, size_t epochs, float learning_rate) = 0;
    virtual void predict(const Tensor<T> &input, Tensor<T> &output) = 0;
};
#endif