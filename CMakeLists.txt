cmake_minimum_required(VERSION 3.24)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(repro)


option(
  USE_EXTERNAL_PYBIND11
  "Use external build of pybind11, only applicable when building python bindings"
  ON)
if(USE_EXTERNAL_PYBIND11)
    find_package(pybind11 REQUIRED CONFIG)
    message(STATUS "Using external pybind11")
else()
  message(STATUS "Fetching pybind11")
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.12.0)
  FetchContent_MakeAvailable(pybind11)
endif()

find_package(Kokkos REQUIRED)
message(STATUS "Using external installation of kokkos")
set(kokkos_libs "Kokkos::kokkos")
if("${ENABLE_KOKKOS_BACKEND}" STREQUAL "CUDA")
  project(SYNERGIA3 LANGUAGES CXX C CUDA)
endif()

add_subdirectory(src)
