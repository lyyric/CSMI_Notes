cmake_minimum_required (VERSION 3.21)
# set the project name
project( MonProjet )

if(NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choix du type de build" FORCE)
endif ()

# specify the C++ standard
set( CMAKE_CXX_STANDARD 17)
set( CMAKE_CXX_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED)

add_executable(wave-parallel wave-parallel.cpp)
target_link_libraries(wave-parallel PUBLIC MPI::MPI_CXX hdf5::hdf5)
target_include_directories(wave-parallel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})