cmake_minimum_required(VERSION 3.20)

project(voc_conv)
set(CMAKE_CXX_STANDARD 20) 
find_package( OpenCV REQUIRED )
find_package(Threads)
# include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(voc_conv voc_conv.cpp)

# directory of opencv library
# link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(voc_conv ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})