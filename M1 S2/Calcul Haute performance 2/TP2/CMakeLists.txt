cmake_minimum_required(VERSION 3.10)
project(kfrog)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
endif()

# Find Kokkos
set(Kokkos_DIR "/home/helluy/local/lib/cmake/Kokkos")
find_package(Kokkos 4.2 REQUIRED CONFIG)

if(Kokkos_FOUND)
    message(STATUS "Kokkos_DIR: ${Kokkos_DIR}")
    message(STATUS "Kokkos_INCLUDE_DIRS: ${Kokkos_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Kokkos not found")
endif()

# List source files
file(GLOB SOURCES "src/*.cpp")

# ✅ First, define the executable
add_executable(kfrog ${SOURCES})

# ✅ Then, link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(kfrog OpenMP::OpenMP_CXX)
endif()

target_link_libraries(kfrog Kokkos::kokkos)

# ✅ Ensure Kokkos include directories are used
target_include_directories(kfrog PRIVATE ${Kokkos_INCLUDE_DIRS})

# Debugging output
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
