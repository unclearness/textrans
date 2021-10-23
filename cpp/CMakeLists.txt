cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME textrans)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "textrans cpp")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

option(TEXTRANS_USE_OPENCV "Use OpenCV" ON)

if (TEXTRANS_USE_OPENCV)
  find_package(OpenCV REQUIRED)
  set(UGU_USE_OPENCV ON CACHE BOOL "")
endif()
add_subdirectory(third_party/ugu)
if (TEXTRANS_USE_OPENCV)
  add_definitions(-DUGU_USE_OPENCV)
endif()

add_definitions(-DUGU_USE_TINYOBJLOADER)
add_definitions(-DUGU_USE_LODEPNG)
add_definitions(-DUGU_USE_TINYCOLORMAP)
add_definitions(-DUGU_USE_NANORT)
add_definitions(-DUGU_USE_JSON)
add_definitions(-DUGU_USE_NANOFLANN)
add_definitions(-DUGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION)
add_definitions(-DUGU_USE_OPENMP)

add_executable(textrans app.cc)
target_include_directories(textrans PRIVATE ${Ugu_INCLUDE_DIRS} third_party/cxxopts/include)
target_link_libraries(textrans ${Ugu_LIBS})

