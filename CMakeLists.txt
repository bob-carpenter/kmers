cmake_minimum_required(VERSION 3.0)
project(
  libkmers
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)

file(GLOB KMERS_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(kmers SHARED ${KMERS_SOURCES})
target_compile_options(kmers PRIVATE)
target_link_libraries(kmers PRIVATE)

install(TARGETS kmers)
