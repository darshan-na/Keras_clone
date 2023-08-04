# CXX = g++
# CXXFLAGS = -Wall -Wextra -Wpedantic -std=c++17

# # List of object files to be built
# OBJS = loss.o dense.o activations.o metrics.o main.o vanilla_rnn.o

# # List of header files
# HDRS = model_interface.h loss.h layer.h dense.h activations.h activation_interface.h tensor.h gml.h metrics.h metrics_interface.h model_dnn.h model_rnn.h rnn_interface.h vanilla_rnn.h

# # List of source files
# SRCS = loss.cpp dense.cpp activations.cpp metrics.cpp main.cpp vanilla_rnn.cpp

# # Name of executable file
# EXEC = my_gml

# # Compile the executable
# $(EXEC): $(OBJS)
# 	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS)

# # Compile source files to object files

# loss.o: loss.cpp $(HDRS)
# 	$(CXX) $(CXXFLAGS) -c loss.cpp

# dense.o: dense.cpp $(HDRS)
# 	$(CXX) $(CXXFLAGS) -c dense.cpp

# activations.o: activations.cpp $(HDRS)
# 	$(CXX) $(CXXFLAGS) -c activations.cpp

# # Add dependencies for tensor.h and activation_interface.h header files

# metrics.o: metrics.cpp $(HDRS)
# 	$(CXX) $(CXXFLAGS) -c metrics.cpp

# main.o: main.cpp $(HDRS)
# 	$(CXX) $(CXXFLAGS) -c main.cpp

# vanilla_rnn.o: vanilla_rnn.cpp $(HDRS)
# 	$(CXX) $(CXXFLAGS) -c vanilla_rnn.cpp
# # Clean up object files and the executable
# clear:
# 	rm -f $(OBJS) $(EXEC)

CXX = g++
CXXFLAGS = -Wall -Wextra -Wpedantic -std=c++17 -I./cnpy -lz

# List of object files to be built
OBJS = loss.o dense.o activations.o metrics.o main.o vanilla_rnn.o cnpy.o 
LIBS = -lz
# List of header files
HDRS = model_interface.h loss.h layer.h dense.h activations.h activation_interface.h tensor.h gml.h metrics.h metrics_interface.h model_dnn.h model_rnn.h rnn_interface.h vanilla_rnn.h cnpy/cnpy.h cnn.h model_cnn.h

# List of source files
SRCS = loss.cpp dense.cpp activations.cpp metrics.cpp main.cpp vanilla_rnn.cpp cnpy/cnpy.cpp

# Name of executable file
EXEC = my_gml

# Compile the executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

# Compile source files to object files

loss.o: loss.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c loss.cpp

dense.o: dense.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c dense.cpp

activations.o: activations.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c activations.cpp

metrics.o: metrics.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c metrics.cpp

main.o: main.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c main.cpp

vanilla_rnn.o: vanilla_rnn.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c vanilla_rnn.cpp

cnpy.o: cnpy/cnpy.cpp cnpy/cnpy.h
	$(CXX) $(CXXFLAGS) -c cnpy/cnpy.cpp

# Clean up object files and the executable
clear:
	rm -f $(OBJS) $(EXEC)
