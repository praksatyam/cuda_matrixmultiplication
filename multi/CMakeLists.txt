cmake_minimum_required(VERSION 3.10)
project(multi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
include_directories(include)
add_executable(multi src/main.cu src/utils.cpp)

find_package(CUDA REQUIRED)
target_link_libraries(multi PRIVATE cublas)
