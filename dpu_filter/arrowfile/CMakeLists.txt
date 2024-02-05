cmake_minimum_required(VERSION 3.10)

project(writearrow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Arrow package
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(ArrowDataset REQUIRED)

# Add your source code files
add_executable(writearrow writearrow.cpp)
add_executable(readparquet read_parquet.cpp)
add_executable(processfile processFile.cc)
add_executable(processcolumn processColumn.cpp)
add_executable(read_dataset read_dataset.cpp)  # Add the new source file here

# Link the Arrow and Parquet libraries
target_link_libraries(writearrow Arrow::arrow_shared Parquet::parquet_shared)
target_link_libraries(readparquet Arrow::arrow_shared Parquet::parquet_shared)
target_link_libraries(processfile Arrow::arrow_shared Parquet::parquet_shared ArrowDataset::arrow_dataset_shared)
target_link_libraries(read_dataset Arrow::arrow_shared Parquet::parquet_shared ArrowDataset::arrow_dataset_shared)  # Link with ArrowDataset library
target_link_libraries(processcolumn Arrow::arrow_shared Parquet::parquet_shared ArrowDataset::arrow_dataset_shared)  # Link with ArrowDataset library


# Set compiler flags to enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set compiler flags (if needed)
target_compile_options(readparquet
    PRIVATE
    -DUTF8PROC_EXPORTS
    -DCARES_STATICLIB
    -I/opt/mellanox/grpc/include
    -I/usr/include/x86_64-linux-gnu
)

# Set compiler flags (if needed)
target_compile_options(writearrow
    PRIVATE
    -DUTF8PROC_EXPORTS
    -DCARES_STATICLIB
    -I/opt/mellanox/grpc/include
    -I/usr/include/x86_64-linux-gnu
)

# Set compiler flags (if needed)
target_compile_options(processfile
    PRIVATE
    -DUTF8PROC_EXPORTS
    -DCARES_STATICLIB
    -I/opt/mellanox/grpc/include
    -I/usr/include/x86_64-linux-gnu
)


# Set compiler flags (if needed)
target_compile_options(processcolumn
    PRIVATE
    -DUTF8PROC_EXPORTS
    -DCARES_STATICLIB
    -I/opt/mellanox/grpc/include
    -I/usr/include/x86_64-linux-gnu
)
