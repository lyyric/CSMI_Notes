cmake_minimum_required(VERSION 3.10)

# 项目名称
project(WaveEquation2D LANGUAGES C CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到 MPI 库
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# 找到 HDF5 库（优先使用并行版本）
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

# 添加源文件
add_executable(wave-parallel src/wave-parallel.cpp)

# 链接 MPI 和 HDF5 库
target_link_libraries(wave-parallel PRIVATE MPI::MPI_CXX hdf5::hdf5)