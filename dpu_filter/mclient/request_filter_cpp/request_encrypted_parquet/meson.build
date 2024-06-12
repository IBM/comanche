project('dpucpp', 'cpp', default_options : ['cpp_std=c++17'])

# Define components
service_components = ['s3']

# Find dependencies
arrow_dep = dependency('arrow')
aws_sdk_dep = dependency('aws-cpp-sdk-s3')
openssl_dep = dependency('openssl')

# Define your source files
source_files = files(['localfilter.cpp'])

sample_link_args = [
    '-larrow',     # Link with Arrow library
    '-lparquet',   # Link with Parquet library
    '-larrow_dataset',
    '-larrow_acero',
    '-lcurl',
    '-lssl',       # Link with OpenSSL library
    '-lcrypto',    # Link with OpenSSL crypto library
]

sample_inc_dirs = []
# Common DOCA library logic
# Define include directories for the SQL parser
sample_inc_dirs += include_directories('..')

# Add executable target
executable('localfilter', source_files,
           dependencies: [arrow_dep, aws_sdk_dep, openssl_dep],
           include_directories: sample_inc_dirs,
           c_args: ['-D_FILE_OFFSET_BITS=64',
                    '-Wall',
                    '-Winvalid-pch',
                    '-Wnon-virtual-dtor'],
           cpp_args: ['-std=c++17',
                      '-fexceptions',
                      '-O3'],  # Add -O3 optimization flag
           link_args: sample_link_args)
