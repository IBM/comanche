cmake_minimum_required(VERSION 3.10)

project(writearrow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Arrow package
find_package(Arrow REQUIRED)

# Add your source code file
add_executable(writearrow writearrow.cpp)

# Link the Arrow libraries
target_link_libraries(writearrow Arrow::arrow_shared parquet)

# Set compiler flags (if needed)
target_compile_options(writearrow
    PRIVATE
    -DUTF8PROC_EXPORTS
    -DCARES_STATICLIB
    -I/opt/mellanox/grpc/include
    -I/usr/include/x86_64-linux-gnu
)
