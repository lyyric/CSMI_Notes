cmake_minimum_required(VERSION 2.8)

project(isosurface)

set(CMAKE_BUILD_TYPE "Release")

# Under Windows, un-comment the following line and set the `VTK_DIR' variable to
# the path where the file `VTKConfig.cmake' is located (in your installed VTK 
# source tree).

#set(VTK_DIR "C:\\Program Files\\VTK\\lib\\vtk-5.10")

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

include_directories(".")

add_executable(isosurface 
  IsoSurfacer.cpp
  UserInterface.cpp 
  Editor.cpp 
  main.cpp)

target_link_libraries(isosurface
  ${VTK_LIBRARIES})

install(TARGETS isosurface RUNTIME DESTINATION bin)
