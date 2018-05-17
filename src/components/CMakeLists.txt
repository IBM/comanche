cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_subdirectory (block)
add_subdirectory (partition)
add_subdirectory (pmem)
add_subdirectory (fs)
add_subdirectory (allocator)
add_subdirectory (store)
add_subdirectory (metadata)
add_subdirectory (net)
add_subdirectory (examples)
#add_subdirectory (cluster)

install(DIRECTORY "api" DESTINATION include FILES_MATCHING PATTERN "*.h*")
      
