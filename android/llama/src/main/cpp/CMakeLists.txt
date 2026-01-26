cmake_minimum_required(VERSION 3.22.1)
project(llama-android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to llama.cpp source
set(LLAMA_CPP_DIR ${CMAKE_SOURCE_DIR}/../../../../llama.cpp)

# Build llama.cpp as static library
set(BUILD_SHARED_LIBS OFF)
set(GGML_STATIC ON)
set(LLAMA_STATIC ON)

# Enable common library build
set(LLAMA_BUILD_COMMON ON)

# Android-specific settings
set(GGML_OPENMP OFF)
set(GGML_LLAMAFILE OFF)
set(GGML_NATIVE OFF)

# Add llama.cpp subdirectory
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp.build)

# Create the Android JNI library
add_library(llama-android SHARED
    llama-android.cpp
)

target_include_directories(llama-android PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/ggml/include
    ${LLAMA_CPP_DIR}/common
)

target_link_libraries(llama-android
    common
    llama
    ggml
    android
    log
)

# Strip debug symbols in release
set_target_properties(llama-android PROPERTIES
    LINK_FLAGS_RELEASE "-s"
)

