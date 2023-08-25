project('SAMPLE_PI', 'C', 'CPP',
    # Get version number from file.
    version: run_command(find_program('cat'),
        files('/opt/mellanox/doca/applications/VERSION'), check: true).stdout().strip(),
    license: 'Proprietary',
    default_options: ['buildtype=debug'],
    meson_version: '>= 0.61.2'
)

SAMPLE_NAME = 'ips'

# Comment this line to restore warnings of experimental DOCA features
add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp'])

sample_dependencies = []
# Required for all DOCA programs
sample_dependencies += dependency('doca')
# 3rd Party dependencies
sample_dependencies += dependency('libdpdk')

# Comment this line to restore warnings of experimental DOCA features
add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: ['c', 'cpp'])

common_path = '/opt/mellanox/doca/applications/common/src'
common_dir_path = common_path
samples_dir_path = '/opt/mellanox/doca/samples'

sample_srcs = [
    # The sample itself
    SAMPLE_NAME + '_core.c',
    # Main function for the sample's executable
    SAMPLE_NAME + '.c',
    # Common code for all DOCA applications
    'dpdk_utils.c',
    '/opt/mellanox/doca/applications/common/src/offload_rules.c',
    # Common code for all DOCA applications
    '/opt/mellanox/doca/applications/common/src/doca_utils.c',
    'dpdk_utils.c',
    'dpi_worker.c',
    common_dir_path + '/offload_rules.c',
    common_dir_path + '/sig_db.c',
    common_dir_path + '/telemetry.c',
    common_dir_path + '/utils.c',
    common_dir_path + '/doca_utils.c',
    samples_dir_path + '/common.c',
    'process_data.cc',  # Include process_data.cpp here
]

sample_inc_dirs  = []
# Common DOCA library logic
sample_inc_dirs += include_directories('..')
# Common DOCA logic (applications)
sample_inc_dirs += include_directories('/opt/mellanox/doca/applications/common/src')

sample_link_args = [
    '-larrow',     # Link with Arrow library
    '-lparquet',   # Link with Parquet library
	'-Wl,-rpath=/usr/local/lib',  # Add rpath to the library directory
]

executable('doca_' + SAMPLE_NAME, sample_srcs,
    c_args : '-Wno-missing-braces',
    dependencies : sample_dependencies,
    include_directories: sample_inc_dirs,
    install: false,
    link_args: sample_link_args  # Specify the link arguments for libraries
)
