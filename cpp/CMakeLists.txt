cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME textrans)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "textrans cpp")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 


set(Textrans_LIBS)

if (WIN32)
# option for Visual Studio
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "/std:c++17 /W4 /EHsc /MP")


else()
# g++ option for *nix

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #Set this to ON if you want to use GNU++
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O2")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(Textrans_LIBS ${Textrans_LIBS} Threads::Threads)

endif()

option(TEXTRANS_USE_OPENCV "Use OpenCV" ON)

if (TEXTRANS_USE_OPENCV)
  find_package(OpenCV REQUIRED)
  set(UGU_USE_OPENCV ON CACHE BOOL "")
else()
  set(UGU_USE_OPENCV OFF CACHE BOOL "")
  set(UGU_USE_STB ON CACHE BOOL "")
  set(UGU_USE_LODEPNG ON CACHE BOOL "")
endif()

option(TEXTRANS_USE_OPENMP "Use OpenMP" ON)
if(APPLE)
  set(TEXTRANS_USE_OPENMP OFF CACHE BOOL "")
  set(UGU_USE_OPENMP OFF CACHE BOOL "")
endif()


add_subdirectory(third_party/ugu)


if (TEXTRANS_USE_OPENCV)
  add_definitions(-DUGU_USE_OPENCV)
else()
  add_definitions(-DUGU_USE_LODEPNG)
  add_definitions(-DUGU_USE_STB)
endif()

add_definitions(-DUGU_USE_TINYOBJLOADER)
add_definitions(-DUGU_USE_TINYCOLORMAP)
add_definitions(-DUGU_USE_NANORT)
add_definitions(-DUGU_USE_JSON)
add_definitions(-DUGU_USE_NANOFLANN)
add_definitions(-DUGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION)

# For OpenMP
if(TEXTRANS_USE_OPENMP)
  add_definitions(-DUGU_USE_OPENMP)
  if(UGU_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
  endif()
endif()


add_executable(textrans app.cc)
target_include_directories(textrans PRIVATE ${Ugu_INCLUDE_DIRS} third_party/cxxopts/include)
target_link_libraries(textrans ${Ugu_LIBS} ${Textrans_LIBS})

