cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

project(ml)

include(FetchContent)
FetchContent_Declare(mdspan
  GIT_REPOSITORY git@github.com:kokkos/mdspan.git
  GIT_TAG mdspan-0.5.0
)
FetchContent_MakeAvailable(mdspan)

find_package(X11)

add_library(io src/io.hpp src/io.cpp)
target_link_libraries(io mdspan)

add_library(model src/model.hpp src/model.cpp)
target_link_libraries(model mdspan)

add_executable(handwriting src/handwriting.cpp)
target_link_libraries(handwriting io mdspan model X11)
