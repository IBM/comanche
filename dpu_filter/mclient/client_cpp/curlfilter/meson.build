project('DOCA_SAMPLE', 'c', 'cpp',
    version: run_command(find_program('cat'),
    files('/opt/mellanox/doca/applications/VERSION'), check: true).stdout().strip(),
    license: 'Proprietary',
    default_options: ['cpp_std=c++17', 'buildtype=release'],
    meson_version: '>= 0.61.2'
)

# Dependencies from both projects
dependencies = [
    dependency('arrow'),
    dependency('libcurl', version: '>= 7.x', fallback: ['curl', 'libcurl_dep'])
]

# Source files from both projects
source_files = files([
    'localfilter.cpp',
    'SQLParser.cpp',
])

# Include directories from both projects
include_dirs = include_directories(
    '..',
    '../..',
    '/home/ubuntu/json/include',
)

# Compiler arguments for C and CPP files
add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp'])
add_project_arguments(['-D_FILE_OFFSET_BITS=64', '-Wall', '-Winvalid-pch'], language: 'c')
add_project_arguments(['-D_FILE_OFFSET_BITS=64', '-Wall', '-Winvalid-pch', '-Wnon-virtual-dtor', '-fexceptions'], language: 'cpp')

# Link arguments for both projects
link_args = [
    '-larrow',
    '-lparquet',
    '-larrow_dataset',
    '-larrow_acero',
    '-lcurl', # Ensure libcurl is linked properly
]

executable('localfilter', source_files,
    dependencies: dependencies,
    include_directories: include_dirs,
    c_args: ['-Wno-missing-braces', '-O3'],
    cpp_args: ['-std=c++17', '-fexceptions', '-O3'],
    link_args: link_args,
    install: false
)
