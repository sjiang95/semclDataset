cmake_minimum_required(VERSION 3.20)

project(dataset_conv)
set(CMAKE_CXX_STANDARD 20) 
find_package( OpenCV REQUIRED )
find_package(Threads)
# include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(dataset_conv dataset_conv.cpp)

# directory of opencv library
# link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(dataset_conv ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})