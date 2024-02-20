project('dpucpp_stat', 'cpp', default_options : ['cpp_std=c++17'])

# Define components
service_components = ['s3']

# Find Arrow
arrow_dep = dependency('arrow')

# Find AWS SDK
aws_sdk_dep = dependency('aws-cpp-sdk-s3')

# Define your source files
source_files = files('localfilter.cpp')

# Define Pistache dependency using pkg-config
pistache_dep = dependency('libpistache')

sample_link_args = [
    '-larrow',     # Link with Arrow library
    '-lparquet',   # Link with Parquet library
    '-larrow_dataset',
    '-larrow_acero',
    '-Wl,-rpath=/usr/local/lib',  # Add rpath to the library directory
]

sample_inc_dirs  = []
# Common DOCA library logic
sample_inc_dirs += include_directories('/home/ubuntu/json/include')



# Add executable target
executable('localfilter', source_files,
           dependencies: [arrow_dep, pistache_dep, aws_sdk_dep],
           include_directories: sample_inc_dirs,
           c_args: ['-D_FILE_OFFSET_BITS=64',
                    '-Wall',
                    '-Winvalid-pch',
                    '-Wnon-virtual-dtor'],
           cpp_args: ['-std=c++17',
                      '-fexceptions',
                      '-O3'],  # Add -O3 optimization flag
           link_args: sample_link_args)
