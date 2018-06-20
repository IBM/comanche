
cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

CHECK_INCLUDE_FILES(infiniband/verbs_exp.h HAVE_INFINIBAND_HEADERS)
if(HAVE_INFINIBAND_HEADERS)
  add_subdirectory (rdma)
endif()
# add_subdirectory (fabric)
