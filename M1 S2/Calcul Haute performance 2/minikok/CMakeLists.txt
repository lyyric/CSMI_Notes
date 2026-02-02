cmake_minimum_required(VERSION 3.10)
project(minikok)

set(CMAKE_CXX_EXTENSIONS OFF)
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
set(Kokkos_DIR "/home/u1/helluy/local/lib/cmake/Kokkos")
find_package(Kokkos REQUIRED CONFIG)

if(Kokkos_FOUND)
    message(STATUS "Kokkos_DIR: ${Kokkos_DIR}")
    message(STATUS "Kokkos_INCLUDE_DIRS: ${Kokkos_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Kokkos not found")
endif()

# List source files
set(SOURCES 
    src/minikok.cpp
)

# --- Executable 1: minikok_cuda ---
add_executable(minikok_cuda ${SOURCES})


find_package(CUDA QUIET)
find_package(CUDA QUIET)
if(CUDA_FOUND)
	message(STATUS "CUDA found: ${CUDA_VERSION}")
	#target_include_directories(minikok_cuda PRIVATE ${CUDA_INCLUDE_DIRS})
else()
	message(WARNING "CUDA not found. minikok_cuda may fail to compile.")
endif()
# Link Kokkos (CUDA activé implicitement si disponible dans votre installation Kokkos)
target_link_libraries(minikok_cuda PRIVATE Kokkos::kokkos)
target_include_directories(minikok_cuda PRIVATE ${Kokkos_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
# Find CUDA


# Ajouter des définitions spécifiques pour CUDA (pas de OPMP ici)
target_compile_definitions(minikok_cuda PRIVATE KOKKOS_ENABLE_CUDA)

# --- Executable 2: minikok_openmp ---
add_executable(minikok_openmp ${SOURCES})

# Link Kokkos et OpenMP
target_link_libraries(minikok_openmp PRIVATE Kokkos::kokkos)
if(OpenMP_CXX_FOUND)
    target_link_libraries(minikok_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(minikok_openmp PRIVATE ${Kokkos_INCLUDE_DIRS})
target_compile_options(minikok_openmp PRIVATE -O3)


# Ajouter des définitions spécifiques pour OpenMP et activer OPMP
target_compile_definitions(minikok_openmp PRIVATE KOKKOS_ENABLE_OPENMP OPMP)

# Debugging output
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
