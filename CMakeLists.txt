cmake_minimum_required(VERSION 3.2)
project(SIFT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

include_directories(
      include
      ${OpenCV_INCLUDE_DIRS}
)

FIND_PACKAGE(OpenCV REQUIRED) # require opencv 3.2

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


set(HEADER_FILES SIFT.h)
set(SOURCE_FILES SIFT.cpp main.cpp  )
add_executable(SIFT ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(SIFT vigraimpex ${OpenCV_LIBS})