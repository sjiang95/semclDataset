cmake_minimum_required(VERSION 3.16) # default for Ubuntu20.04

project(dataset_conv)
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")

add_executable(dataset_conv main.cpp)

if(MINGW OR MSVC) # on windows
    # suppose environmant variable `OPENCV_ROOT` points to the installation folder of opencv, which contains `OpenCVConfig.cmake`
    set(OpenCV_DIR $ENV{OPENCV_ROOT})
    if(MINGW)
        set(OpenCV_LIBRARY_DIRS $ENV{OPENCV_ROOT}/x64/mingw/lib)
    else()# MSVC
        set(OpenCV_LIBRARY_DIRS $ENV{OPENCV_ROOT}/x64/vc17/lib)# vc17==vs2022
    endif()
endif()
find_package( OpenCV REQUIRED COMPONENTS highgui imgcodecs imgproc)
message(STATUS "OpenCV include path: " ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV library path: " ${OpenCV_LIBRARY_DIRS})

target_link_libraries(dataset_conv ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})