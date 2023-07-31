cmake_minimum_required(VERSION 3.1)
project(
  libkmers
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)

find_package(ZLIB REQUIRED)

set(KMERS_SOURCES "${PROJECT_SOURCE_DIR}/src/libkmers.cpp")
add_library(kmers SHARED ${KMERS_SOURCES})
target_compile_options(kmers PRIVATE)
target_link_libraries(kmers PRIVATE ${ZLIB_LIBRARIES})

install(TARGETS kmers)
