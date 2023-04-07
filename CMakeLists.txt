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

# -ffast-math literally doubles the training speed.
add_compile_options(-ffast-math)

find_package(X11)

add_library(io src/io.hpp src/io.cpp)
target_link_libraries(io mdspan)

add_library(model src/model.hpp src/model.cpp)
target_link_libraries(model mdspan)

add_library(load_digit src/load_digit.cpp src/load_digit.hpp)
target_link_libraries(load_digit mdspan X11)

add_executable(handwriting src/handwriting.cpp)
target_link_libraries(handwriting io mdspan model load_digit)
