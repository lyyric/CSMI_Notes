cmake_minimum_required(VERSION 2.8)

project(cylinderExample)

set(CMAKE_BUILD_TYPE "Release")

# Under Windows, un-comment the following line and set the `VTK_DIR' variable to
# the path where the file `VTKConfig.cmake' is located (in your installed VTK 
# source tree).

#set(VTK_DIR "C:\\Program Files\\VTK\\lib\\vtk-5.10")

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

include_directories(".")

add_executable(cylinderExample 
  CylinderExample.cpp)

target_link_libraries(cylinderExample
  ${VTK_LIBRARIES})

install(TARGETS cylinderExample RUNTIME DESTINATION bin)
