project(Lanny)
cmake_minimum_required(VERSION 3.5)

set (CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS "-std=c++0x -fopenmp -msse4.1 -mavx2")

find_package(OpenCV 3.4 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )


set(OPENNI2_INCLUDE_DIRS "/home/doctorant/libs/OpenNI2/Include")
set(OPENNI2_LIBRARIES "/home/doctorant/libs/OpenNI2/Bin/x64-Release")

find_package(PCL 1.3 REQUIRED COMPONENTS common io visualization)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

file(GLOB MY_SRC
    "*.h"
    "*.cpp"
)

add_executable(${PROJECT_NAME} ${MY_SRC})

target_link_libraries(Lanny ${OpenCV_LIBS} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} pcl_filters)

install(TARGETS Lanny DESTINATION bin)
