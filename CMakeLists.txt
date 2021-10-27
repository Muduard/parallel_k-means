cmake_minimum_required(VERSION 3.10)
project(parziale_programmazione_parallela)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
add_library(datalib STATIC dataManipulator.cpp)
target_include_directories(datalib PUBLIC include)
add_executable(parziale main.cpp )
target_link_libraries(datalib PUBLIC ${Boost_LIBRARIES})
target_link_libraries(parziale PUBLIC datalib)