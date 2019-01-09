# Additional target to perform clang-format/clang-tidy run
# Requires clang-format and clang-tidy

# export C_INCLUDE_PATH=/usr/local/include:/usr/include/x86_64-linux-gnu:/usr/include  
#export CPLUS_INCLUDE_PATH=/usr/local/include:/usr/include/x86_64-linux-gnu:/usr/include 


# Get all project files
file(GLOB_RECURSE ALL_SOURCE_FILES *.c *.cc *.h *.cpp)

# This will search the tree upwards for .clang-format
add_custom_target(
  ${PROJECT_NAME}-format
  COMMAND clang-format
  -style=file
  -i
  ${ALL_SOURCE_FILES}
  )
      
      
#get_property(incdirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#set(sysincludes "-I/usr/include/c++/5 -I/usr/include/x86_64-linux-gnu/c++/5/ -I/usr/include/linux")
# add -I prefix
#string(REGEX REPLACE "([^;]+)" "-I\\1" istring "${incdirs}")

# add_custom_target(
#         tidy
#         COMMAND /usr/bin/clang-tidy -header-filter=.* ${ALL_SOURCE_FILES} -- -std=c++11 ${istring} 
# )

# add_custom_target(
#         check
#         COMMAND scan-build
#         make
# )


