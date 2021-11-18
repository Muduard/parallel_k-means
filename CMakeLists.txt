cmake_minimum_required(VERSION 3.10)
project(parziale_programmazione_parallela)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
FIND_PACKAGE( Boost 1.40 COMPONENTS program_options REQUIRED )

include(FetchContent) # If not included already

FetchContent_Declare(sciplot_content
  GIT_REPOSITORY https://github.com/sciplot/sciplot.git
  GIT_TAG master)

FetchContent_GetProperties(sciplot_content)
if(NOT sciplot_content_POPULATED)
  FetchContent_Populate(sciplot_content)
endif()

include_directories(${sciplot_content_SOURCE_DIR})
include_directories(${sciplot_content_SOURCE_DIR})
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
add_library(datalib STATIC src/dataManipulator.cpp)
add_library(kmeans STATIC src/kmeans.cpp)
target_include_directories(datalib PUBLIC include)
target_include_directories(kmeans PUBLIC include)
add_executable(parziale src/main.cpp src/dataManipulator.cpp src/kmeans.cpp)
target_link_libraries(datalib PUBLIC ${Boost_LIBRARIES})
target_link_libraries(parziale PUBLIC datalib)
target_link_libraries(parziale PUBLIC kmeans)