cmake_minimum_required(VERSION 3.10)
project(dpucpp)

set(SERVICE_COMPONENTS s3)

# Find Arrow
find_package(Arrow REQUIRED)


# Find AWS SDK
find_package(AWSSDK REQUIRED COMPONENTS ${SERVICE_COMPONENTS})

# Add your source files here
set(SOURCE_FILES
    localfilter.cpp
)

find_package(PkgConfig)
pkg_check_modules(Pistache REQUIRED IMPORTED_TARGET libpistache)

# Add your executable target
add_executable(localfilter ${SOURCE_FILES})

# Link Arrow, Pistache, and AWS SDK libraries
target_link_libraries(localfilter Arrow::arrow_shared
                                                 PkgConfig::Pistache aws-cpp-sdk-s3)
