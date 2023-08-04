#include "gml.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;
#include "cnpy/cnpy.h"

// void read_mnist_images(const string& file_name, Tensor<float>& images, int num_images, int num_rows, int num_cols) {
//     ifstream file(file_name, ios::binary);

//     if (!file.is_open()) {
//         cerr << "Error opening file: " << file_name << endl;
//         exit(-1);
//     }

//     int magic_number = 0;
//     file.read((char*)&magic_number, sizeof(magic_number));
//     magic_number = __builtin_bswap32(magic_number);

//     if (magic_number != 2051) {
//         cerr << "Invalid MNIST image file: " << file_name << endl;
//         exit(-1);
//     }

//     file.read((char*)&num_images, sizeof(num_images));
//     num_images = __builtin_bswap32(num_images);

//     file.read((char*)&num_rows, sizeof(num_rows));
//     num_rows = __builtin_bswap32(num_rows);

//     file.read((char*)&num_cols, sizeof(num_cols));
//     num_cols = __builtin_bswap32(num_cols);

//     images = Tensor<float>({num_rows, num_cols, num_images});

//     for (int i = 0; i < num_images; i++) {
//         for (int r = 0; r < num_rows; r++) {
//             for (int c = 0; c < num_cols; c++) {
//                 unsigned char pixel = 0;
//                 file.read((char*)&pixel, sizeof(pixel));
//                 images(r, c, i) = static_cast<float>(pixel) / 255.0f;
//             }
//         }
//     }

//     file.close();
// }

// void read_mnist_labels(const string& file_name, Tensor<float>& labels, int num_labels) {
//     ifstream file(file_name, ios::binary);

//     if (!file.is_open()) {
//         cerr << "Error opening file: " << file_name << endl;
//         exit(-1);
//     }

//     int magic_number = 0;
//     file.read((char*)&magic_number, sizeof(magic_number));
//     magic_number = __builtin_bswap32(magic_number);

//     if (magic_number != 2049) {
//         cerr << "Invalid MNIST label file: " << file_name << endl;
//         exit(-1);
//     }

//     file.read((char*)&num_labels, sizeof(num_labels));
//     num_labels = __builtin_bswap32(num_labels);

//     labels = Tensor<float>({10, num_labels});

//     for (int i = 0; i < num_labels; i++) {
//         unsigned char label = 0;
//         file.read((char*)&label, sizeof(label));
//         for (int j = 0; j < 10; j++) {
//             labels(j, i) = (label == j) ? 1.0f : 0.0f;
//         }
//     }

//     file.close();
// }

// void load_mnist_data(Tensor<float> &train_images, Tensor<float> &train_labels, Tensor<float> &test_images, Tensor<float> &test_labels) {
//     int num_train_images = 0;
//     int num_train_labels = 0;
//     int num_test_images = 0;
//     int num_test_labels = 0;
//     int num_rows = 0;
//     int num_cols = 0;

//     string train_images_file = "train-images-idx3-ubyte";
//     string train_labels_file = "train-labels-idx1-ubyte";
//     string test_images_file = "t10k-images-idx3-ubyte";
//     string test_labels_file = "t10k-labels-idx1-ubyte";

//     read_mnist_images(train_images_file, train_images, num_train_images, num_rows, num_cols);
//     read_mnist_labels(train_labels_file, train_labels, num_train_labels);

//     if (num_train_images != num_train_labels) {
//         cerr << "Number of train images and train labels do not match." << endl;
//         exit(-1);
//     }

//     read_mnist_images(test_images_file, test_images, num_test_images, num_rows, num_cols);
//     read_mnist_labels(test_labels_file, test_labels, num_test_labels);

//     if (num_test_images != num_test_labels) {
//         cerr << "Number of test images and test labels do not match." << endl;
//         exit(-1);
//     }
// }
vector<vector<float>> read_csv(string filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }
    vector<vector<float>> data;
    if (file)
    {
        string line;
        while (getline(file, line))
        {
            vector<float> row;
            stringstream ss(line);
            string cell;

            while (getline(ss, cell, ','))
            {
                // cout<<(cell)<<endl;
                row.push_back(stof(cell));
            }

            data.push_back(row);
        }
    }

    return data;
}

vector<string> load_dinosaur_names(const string &filename)
{
    vector<string> names;
    ifstream file(filename);
    string line;
    while (getline(file, line))
    {
        names.push_back(line);
    }
    return names;
}

set<char> get_unique_characters(const vector<string> &names)
{
    set<char> unique_chars;
    for (const auto &name : names)
    {
        for (const char &c : name)
            unique_chars.insert(c);
    }
    unique_chars.insert('\n');
    return unique_chars;
}

Tensor<float> one_hot_encode(const string &name, map<char, size_t> &char_to_index, size_t num_chars, size_t maxi)
{
    Tensor<float> one_hot_tensor(vector<size_t>({num_chars, maxi + 1, 1}));

    for (size_t i = 0; i < maxi; i++)
    {
        char ch = name[i];
        // cout<<name[i]<<endl;
        size_t char_index = char_to_index[ch];
        one_hot_tensor(char_index, i + 1, 0) = 1;
    }

    return one_hot_tensor;
}

Tensor<float> create_target_tensor(const Tensor<float> &input_tensor, map<char, size_t> &char_to_index)
{
    size_t num_chars = input_tensor.shape[0];
    size_t name_length = input_tensor.shape[1];

    Tensor<float> target_tensor(vector<size_t>({num_chars, name_length, 1}));

    for (size_t i = 0; i < num_chars; i++)
    {
        for (size_t j = 0; j < name_length - 1; j++)
        {
            target_tensor(i, j, 0) = input_tensor(i, j + 1, 0);
        }
    }

    size_t newline_index = char_to_index.at('\n');
    target_tensor(newline_index, name_length - 1, 0) = 1;

    return target_tensor;
}

pair<Tensor<float>, Tensor<float>> preprocess_data(const vector<string> &dinosaur_names, map<char, size_t> &char_to_index, size_t maxi)
{
    size_t num_chars = char_to_index.size();
    vector<Tensor<float>> X_data;
    vector<Tensor<float>> Y_data;

    for (const auto &name : dinosaur_names)
    {
        Tensor<float> one_hot_tensor = one_hot_encode(name, char_to_index, num_chars, maxi);
        X_data.push_back(one_hot_tensor);
        Y_data.push_back(create_target_tensor(one_hot_tensor, char_to_index));
    }

    size_t num_names = dinosaur_names.size();
    Tensor<float> X(vector<size_t>({num_chars, num_names, maxi}));
    Tensor<float> Y(vector<size_t>({num_chars, num_names, maxi}));

    for (size_t i = 0; i < num_names; i++)
    {
        for (size_t j = 0; j < num_chars; j++)
        {
            for (size_t k = 0; k < maxi; k++)
            {
                X(j, i, k) = X_data[i](j, k, 0);
                Y(j, i, k) = Y_data[i](j, k, 0);
            }
        }
    }
    cout << X.shape[1] << endl;
    return make_pair(X, Y);
}
// int main()
// {
//     // Load dinosaur names
//     size_t maxi = 0;
//     vector<string> dinosaur_names = load_dinosaur_names("dinos.txt");
//     std::transform(dinosaur_names.begin(), dinosaur_names.end(), dinosaur_names.begin(),
//                [](std::string str) {
//                    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
//                    return str;
//                });
//     for (const auto &name : dinosaur_names)
//     {
//         // cout<<name<<endl;
//         maxi = max(maxi, name.size());
//     }
//     // Get unique characters
//     set<char> unique_chars = get_unique_characters(dinosaur_names);

//     map<char, size_t> char_to_index;
//     map<size_t, char> index_to_char;
//     size_t index = 0;
//     for (const char &ch : unique_chars)
//     {
//         char_to_index[ch] = index;
//         index_to_char[index] = ch;
//         index++;
//     }
//     cout<<char_to_index.size()<< ' '<<maxi<<endl;
//     // Preprocess the dataset
//     pair<Tensor<float>, Tensor<float>> x_y= preprocess_data(dinosaur_names, char_to_index,maxi);
//     Tensor<float> X = x_y.first;
//     Tensor<float> Y = x_y.second;
//     // cout<<X.shape[1]<<endl;
//     // Create the RNN model
//     Model_RNN<float> model;

//     // Add RNN layers
//     // (Adjust the layer parameters as needed)
//     model.add_layer(new vanilla_Rnn<float,tanh_<float>,softmax<float>>( /*hidden_units=*/50, /*input_dim=*/unique_chars.size(),/*output_dim=*/unique_chars.size(),1));

//     // Compile the model
//     model.compile("CategoricalCrossentropy");

//     // Train the model
//     // cout<<"hello"<<endl;
//     model.fit(X, Y, /*batch_size=*/1, /*epochs=*/100, /*learning_rate=*/0.1);
//     // cout<<"hello"<<endl;

//     return 0;
// }

// int main() {
//     // Load the data
//     vector<vector<float>> data = read_csv("breast_cancer.csv");

//     // Split the data into training and testing sets
//     vector<vector<float>> X_train, y_train;
//     size_t n_rows = data.size();
//     size_t n_cols = data[0].size()-1;
//     for (size_t i = 0; i < n_rows; i++) {
//             X_train.push_back(vector<float>(data[i].begin(), data[i].begin()+9));
//             y_train.push_back(vector<float>(data[i].begin()+9, data[i].end()));
//         }
//     // Create a model
//     Model_DNN<float> model;

//     // Add layers to the model
//     model.add_layer(new Dense<float,relu<float>>(n_cols, 16, 683));
//     model.add_layer(new Dense<float,relu<float>>(16, 16, 683));
//     model.add_layer(new Dense<float,sigmoid<float>>(16, 1, 683));

//     // Compile the model with a loss function
//     string model_loss = "BinaryCrossentropy";
//     model.compile(model_loss);

//     // Train the model
//     Tensor<float> input_data({n_cols, X_train.size(), 1});
//     Tensor<float> target_data({1, y_train.size(), 1});
//     for (size_t i = 0; i < X_train.size(); i++) {
//         for (size_t j = 0; j < n_cols; j++) {
//             input_data(j, i, 0) = X_train[i][j];
//         }
//         target_data(0, i, 0) = y_train[i][0];
//     }
//     model.fit(input_data, target_data, 683, 150, 0.003);

//     Make predictions
//     vector<float> temp({5,4,1,5,2,10,3,2,1});
//     Tensor<float>input_test(vector<size_t>({9,1,1}));
//     Tensor<float>output_test(vector<size_t>({1,1,1}));
//     for(size_t i=0;i<9;i++)
//     {
//         input_test(i,0,0) = temp[i];
//     }
//     model.predict(input_test,output_test);
//     cout<<"prediction : "<<output_test(0,0,0)<<endl;
//     return 0;
// }

template <typename T>
Tensor<T> load_npy_data(const std::string &file_path)
{
    cnpy::NpyArray npy_data = cnpy::npy_load(file_path);

    if (npy_data.shape.size() != 3)
    {
        throw std::runtime_error("Expected 3-dimensional data");
    }

    size_t height = npy_data.shape[1];
    size_t width = npy_data.shape[2];
    size_t training_example = npy_data.shape[0];

    Tensor<T> tensor({height, width, training_example});

    T *npy_data_ptr = npy_data.data<T>();

    for (size_t k = 0; k < training_example; ++k)
    {
        for (size_t i = 0; i < height; ++i)
        {
            for (size_t j = 0; j < width; ++j)
            {
                tensor(i, j, k) = npy_data_ptr[k * height * width + i * width + j];
            }
        }
    }

    return tensor;
}

template <typename T>
Tensor<T> load_npy_data_(const std::string &file_path)
{
    cnpy::NpyArray npy_data = cnpy::npy_load(file_path);

    if (npy_data.shape.size() != 3)
    {
        throw std::runtime_error("Expected 3-dimensional data");
    }

    size_t input_size = npy_data.shape[0];
    size_t training_example = npy_data.shape[1];

    Tensor<T> tensor({input_size,training_example,1});

    T *npy_data_ptr = npy_data.data<T>();

    for (size_t k = 0; k < training_example; ++k)
    {
        for (size_t i = 0; i < input_size; ++i)
        {
            // i*shape[1] + j + k*shape[0]*shape[1]
            tensor(i, k, 0) = npy_data_ptr[i*training_example+k];
        }
        
    }

    return tensor;
}

template <typename T>
Tensor<T> load_npy_labels(const std::string &file_path)
{
    cnpy::NpyArray npy_data = cnpy::npy_load(file_path);
    T *npy_data_ptr = npy_data.data<T>();
    if (npy_data.shape.size() != 3)
    {
        throw std::runtime_error("Expected 3-dimensional data");
    }
    Tensor<T> tensor(vector<size_t>({1, npy_data.shape[1], 1}));
    for(size_t i=0;i<npy_data.shape[1];i++)
    {   
        tensor(0, i, 0) = npy_data_ptr[i];
    }
    return tensor;
}

int main()
{


    vector<vector<float>> data = read_csv("breast_cancer.csv");

    // Split the data into training and testing sets
    vector<vector<float>> X_train, y_train;
    size_t n_rows = data.size();
    size_t n_cols = data[0].size()-1;
    for (size_t i = 0; i < n_rows; i++) {
            X_train.push_back(vector<float>(data[i].begin(), data[i].begin()+9));
            y_train.push_back(vector<float>(data[i].begin()+9, data[i].end()));
        }
    // Create a model
    Model_DNN<float> model;

    // Add layers to the model
    model.add_layer(new Dense<float,relu<float>>(n_cols, 16, 683));
    model.add_layer(new Dense<float,relu<float>>(16, 16, 683));
    model.add_layer(new Dense<float,sigmoid<float>>(16, 1, 683));

    // Compile the model with a loss function
    string model_loss = "BinaryCrossentropy";
    model.compile(model_loss);

    // Train the model
    vector<size_t>shape= {n_cols, X_train.size(), 1};
    Tensor<float> input_data(shape);
    cout<<input_data.shape[0]<<' '<<input_data.shape[1]<<' '<<input_data.shape[2]<<endl;
    Tensor<float> target_data({1, y_train.size(), 1});
    for (size_t i = 0; i < X_train.size(); i++) {
        for (size_t j = 0; j < n_cols; j++) {
            input_data(j, i, 0) = X_train[i][j];
        }
        target_data(0, i, 0) = y_train[i][0];
    }
    cout<<"in Main"<<endl;
    cout<<input_data.shape[0]<<' '<<input_data.shape[1]<<' '<<input_data.shape[2]<<endl;
    model.fit(input_data, target_data, 683, 150, 0.002);

    for (size_t i = 0; i < X_train.size(); i++) 
    {
        for (size_t j = 0; j < n_cols; j++) {
            input_data(j, i, 0) = rand()%100;
        }
        target_data(0, i, 0) = rand()%1;
    }

    Model_DNN<float> model_1;
    // Add layers to the model
    Tensor<float> train_data = load_npy_data_<float>("train_images.npy");
    Tensor<float> train_labels = load_npy_labels<float>("train_labels.npy");
    // static_assert(train_data.shape[0]==1024);
    // cout<<train_data.shape[0]<<' '<<train_data.shape[1]<<' '<<train_data.shape[2]<<endl;
    // cout<<train_labels.shape[0]<<' '<<train_labels.shape[1]<<' '<<train_labels.shape[2]<<endl;
    model_1.add_layer(new Dense<float,relu<float>>(n_cols, 16, 683));
    model_1.add_layer(new Dense<float,relu<float>>(16, 16, 683));
    model_1.add_layer(new Dense<float,sigmoid<float>>(16, 1, 683));
    model_1.compile(model_loss);











    // Load training data
    
    Tensor<float> train_data_ = load_npy_data_<float>("train_images.npy");
    Tensor<float> train_labels_ = load_npy_labels<float>("train_labels.npy");
    // Load validation data
    // Tensor<float> val_data = load_npy_data<float>("val_data.npy");
    // Tensor<float> val_labels = load_npy_data<float>("val_labels.npy");

    

    // Build model
    Model_CNN<float, relu<float>, sigmoid<float>> model_;
    model_.add_layer(new Convolution<float, relu<float>>(1, 16, 3));
    model_.add_layer(new Convolution<float, relu<float>>(16, 32, 3));
    model_.add_layer(new Flatten<float>());
    model_.add_layer(new Dense<float, sigmoid<float>>(32 * 28 * 28, 64, 1));
    model_.add_layer(new Dense<float, sigmoid<float>>(64, 1, 1));
    model_.compile("BinaryCrossentropy");

    cout<<"Convolution Operation"<<endl;

    // Train model
    model_1.fit(input_data, target_data, 683, 150, 0.001);

    // model_1.fit(train_data, train_labels, 2059, 5, 0.0001);

    // Evaluate on validation data
    // Tensor<float> val_preds;
    // cnn_model.predict(val_data, val_preds);
    // Accuracy<float> acc;
    // cout << "Validation Accuracy: " << acc.compute(val_preds, val_labels) << endl;

    return 0;
}
