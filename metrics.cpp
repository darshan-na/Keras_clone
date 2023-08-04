#include "metrics.h"
template <typename T>
T Accuracy<T>::compute(const Tensor<T> &y_pred, const Tensor<T> &y_true)
{
    Tensor<T> y_pred_copy = y_pred;
    transform(y_pred_copy.data.begin(), y_pred_copy.data.end(), y_pred_copy.data.begin(), [](T x) -> T
              { return round(x); });
    T num_correct = 0;
    for (size_t i = 0; i < y_pred_copy.shape[1]; i++)
    {
        bool correct = true;
        for (size_t j = 0; j < y_pred_copy.shape[0]; j++)
        {
            if (y_pred_copy(j, i, 0) != y_true(j, i, 0))
            {
                correct = false;
                break;
            }
        }
        if (correct)
        {
            num_correct++;
        }
    }
    T accuracy = num_correct / y_pred_copy.shape[1];
    return accuracy;
}

template class Accuracy<float>;
