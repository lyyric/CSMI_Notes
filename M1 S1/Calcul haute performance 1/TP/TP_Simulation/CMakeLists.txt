cmake_minimum_required(VERSION 3.15)
project(wave_simulation)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)

# 查找 MPI 和 HDF5
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)

# 添加源文件
add_executable(wave_simulation src/wave-parallel.cpp)

target_include_directories(wave_simulation PRIVATE ${CMAKE_SOURCE_DIR}/src)

# 链接库
target_link_libraries(wave_simulation PRIVATE MPI::MPI_C HDF5::HDF5)