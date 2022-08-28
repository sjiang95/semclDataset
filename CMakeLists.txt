cmake_minimum_required(VERSION 3.16) # default for Ubuntu20.04

project(dataset_conv)
set(CMAKE_CXX_STANDARD 20) 
find_package( OpenCV REQUIRED )
find_package(Threads)
# include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(dataset_conv dataset_conv.cpp)

if(MINGW OR MSVC) # on windows
    # suppose environmant variable `OPENCV_ROOT` points to the installation folder of opencv, which contains `OpenCVConfig.cmake`
    set(OpenCV_DIR $ENV{OPENCV_ROOT})
    find_package(OpenCV REQUIRED)
    # suppose environmant variable `OPENCV_ROOT` points to the root directory of Eigen
    set(Eigen_DIR $ENV{EIGEN_ROOT})
    message(STATUS "Eigen include path: " ${Eigen_DIR})
    include_directories(${OpenCV_INCLUDE_DIRS} ${Eigen_DIR})
    if(MINGW)
        set(OpenCV_LIBRARY_DIRS $ENV{OPENCV_ROOT}/x64/mingw/lib)
    else()# MSVC
        set(OpenCV_LIBRARY_DIRS $ENV{OPENCV_ROOT}/x64/vc17/lib)# vc17==vs2022
    endif()
    link_directories(${OpenCV_LIBRARY_DIRS})
    message(STATUS "OpenCV include path: " ${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library path: " ${OpenCV_LIBRARY_DIRS})
else() # gcc on linux
    find_package( OpenCV REQUIRED )
    find_package(Threads)
    # include_directories( ${OpenCV_INCLUDE_DIRS} )
endif()

# directory of opencv library
# link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(dataset_conv ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})