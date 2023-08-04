#ifndef tensor_h
#define tensor_h
#include<bits/stdc++.h>
using namespace std;
template<typename T>
class Tensor {
public:
    Tensor(std::vector<size_t> shape) : shape(shape) , data(size(shape),T()) {
        // cout<<shape[0]<<' '<<shape[1]<<' '<<shape[2]<<endl; 
    }
    Tensor() : shape({1, 1, 1}), data(1, T())
    {
    }
    T& operator()(size_t i, size_t j, size_t k)
    {
        size_t index = get_index(i,j,k);
        return data[index];
    }
    const T& operator()(size_t i, size_t j, size_t k) const
    {
        size_t index = get_index(i,j,k);
        return data[index];
    }

     Tensor(const Tensor& other) 
     {
        this->shape = other.shape;
        this->data = other.data;
     }
     void operator =(const Tensor& other) 
     {
        this->shape = other.shape;
        this->data = other.data;
     }
     Tensor<T> dot(const Tensor<T>& second_matrix)
     {
        Tensor<T> temp(vector<size_t>({this->shape[0],second_matrix.shape[1],1}));
        size_t r_1 = this->shape[0];
        size_t c_1 = this->shape[1];
        size_t r_2 = second_matrix.shape[0];
        size_t c_2 = second_matrix.shape[1];
        if(r_2!=c_1)
        {
            throw runtime_error("Invalid Dimensions cannot perform the dot product");
        }
        for(size_t i=0;i<r_1;i++)
        {
            for(size_t j=0;j<c_2;j++)
            {
                T t = 0;
                for(size_t k=0;k<r_2;k++)
                {
                    t+=(*this)(i,k,0) * second_matrix(k,j,0);
                }
                temp(i,j,0) = t;
            }
        }
        return temp;
     }
     Tensor<T> operator +(Tensor<T>& second)
     {
        if((second.shape[0]!=this->shape[0]) || (second.shape[1]!=this->shape[1]))
        {
            throw runtime_error("The tensors are not of the same size");
        }
        Tensor<T>res(vector<size_t>({this->shape[0],this->shape[1],1}));
        for(size_t i=0;i<this->shape[0];i++)
        {
            for(size_t j=0;j<this->shape[1];j++)
            {
                res(i,j,0) = (*this)(i,j,0) + second(i,j,0);
            }
        }
        return res;
     }
     Tensor<T> sum()
     {
        Tensor<T> temp(vector<size_t>({this->shape[0],1,1}));
        for(size_t i=0;i<this->shape[0];i++)
        {
            T t=0;
            for(size_t j=0;j<this->shape[1];j++)
            {
                t+=(*this)(i,j,0);
            }
            temp(i,0,0) = t;
        }
        return temp;
     }
     Tensor<T> transpose()
     {
        Tensor<T> temp(vector<size_t>({this->shape[1],this->shape[0],1}));
        for(size_t i=0; i<this->shape[1];i++)
        {
            for(size_t j=0;j<this->shape[0];j++)
            {
                temp(i,j,0) = (*this)(j,i,0);
            }
        }
        return temp;
     }
     void print()
     {
        for(int i=0;i<this->data.size();i++)
        {
            cout<<this->data[i]<< ' ';
        }
        cout<<endl;
     }
    
public:
    vector<size_t>shape;
    vector<T> data;

    size_t size(vector<size_t>& shape) const
    {
        size_t s = 1;
        for(size_t i : shape)
        {
            s*=i;
        }
        return s;
    }
    size_t get_index(size_t i,size_t j, size_t k) const
    {
        if(shape.size()!=3)
        {
            throw std::runtime_error("Incorrect dimesions!! Required 3");
            return 0;
        }
        else if(i>=shape[0] || j>=shape[1] || k>=shape[2])
        {
            throw std::out_of_range("Index out of bounds");
            return 0;
        }
        else
        {
            return i*shape[1] + j + k*shape[0]*shape[1];
        }
    }
};
#endif