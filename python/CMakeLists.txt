cmake_minimum_required(VERSION 3.10)

# Set the project name
project(GPU_Python_interop LANGUAGES C CUDA)

# Specify the C++ standard
#set(CMAKE_CXX_STANDARD 13)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Specify the compilers
# CUDA identification by CMake doesn't take this params into account.
# Run cmake with CUDAHOSTCXX=g++-13 cmake ..
#set(CMAKE_CXX_COMPILER g++-13)
#set(CMAKE_CUDA_HOST_COMPILER g++-13)

# Include the directory where NumPy headers are located
include_directories(/usr/lib/x86_64-linux-gnu/python3-numpy/numpy/_core/include/)
include_directories(/usr/include/python3.13/)

add_library(complex_operation SHARED complex_operation_fp32.cu)

# Request that the lib be built with -std=c++11
# As this is a public compile feature anything that links to 
# it will also build with -std=c++11
#target_compile_features(complex_operation PUBLIC cxx_std_17)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( complex_operation
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set the CUDA target architecture
set_target_properties(complex_operation
                       PROPERTIES CUDA_ARCHITECTURES native
)
