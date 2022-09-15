cmake_minimum_required(VERSION 3.10)
project(parziale_programmazione_parallela LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
if(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp")
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
endif()
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
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
add_library(datalib STATIC src/dataManipulator.cpp)
add_library(kmeans STATIC src/kmeans.cpp src/kmeansCuda.cu src/point.cpp )
add_library(utils STATIC src/ioUtils.cpp src/plotUtils.cpp)
target_include_directories(datalib PUBLIC include)
target_include_directories(kmeans PUBLIC include)
target_include_directories(utils PUBLIC include)
if(WIN32)
  target_include_directories(utils PUBLIC win_include)
endif()

add_executable(parziale src/benchmark.cpp src/dataManipulator.cpp src/kmeans.cpp src/kmeansCuda.cu)
target_link_libraries(datalib PUBLIC ${Boost_LIBRARIES})
target_link_libraries(parziale PUBLIC datalib)
target_link_libraries(parziale PUBLIC kmeans)
target_link_libraries(parziale PUBLIC utils)