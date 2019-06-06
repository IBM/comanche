
cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_subdirectory (append)
add_subdirectory (log)
#add_subdirectory (rockstore)
add_subdirectory (filestore)
#add_subdirectory(hstore)
add_subdirectory(nvmestore)
add_subdirectory(mapstore)
add_subdirectory(pmstore)
add_subdirectory(dummystore)
