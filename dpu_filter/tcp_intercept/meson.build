#Nara version 2023.7.3
# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
#
# This software product is a proprietary product of NVIDIA CORPORATION &
# AFFILIATES (the "Company") and all right, title, and interest in and to the
# software product, including all associated intellectual property rights, are
# and shall remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
#

project('DOCA_APPLICATIONS', 'C', 'CPP',
	# Get version number from file.
	version: run_command(find_program('cat'), files('/opt/mellanox/doca/applications/VERSION'), check: true).stdout().strip(),
	license: 'Proprietary',
	default_options: ['buildtype=debug'],
	meson_version: '>= 0.61.2'
)

DOCA_PREFIX = 'doca_'
GPU_SUFFIX = '_gpu'

# With enabling GPU support, languages variable will contain CUDA language
languages = ['c', 'cpp']
if get_option('enable_gpu_support')
	# Enables ifdef for GPU code
	gpu_compile_flags = ['-DGPU_SUPPORT']

	add_languages('CUDA')
	languages += ['cuda']
	cuda = import('unstable-cuda')
	nvcc = meson.get_compiler('cuda')

	nvcc_flags = []
	nvcc_flags += ['-gencode', 'arch=compute_80,code=sm_80']
	add_project_arguments(nvcc_flags, language: 'cuda')

	# Refer to https://mesonbuild.com/Cuda-module.html
	add_project_arguments('-forward-unknown-to-host-compiler', language: 'cuda')
	if get_option('enable_cross_compilation_to_dpu')
		add_project_link_arguments(['-forward-unknown-to-host-linker', '--allow-shlib-undefined'], language : 'cuda')
	endif
	add_project_arguments('-rdc=true', language: 'cuda')

	gpu_dependencies = []
	gpu_dependencies += dependency('cuda', version: '>= 11-8', modules: ['cudart', 'cuda'])

	nvcc_flags_link = []
	nvcc_flags_link += ['-gencode=arch=compute_80,code=sm_80']
	add_project_link_arguments(nvcc_flags_link, language: 'cuda')
endif

install_apps = false
bin_install_dir = ''

base_c_args = [
	'-Wno-missing-braces',
	'-Wno-missing-field-initializers',
]

base_cpp_args = [
	'-std=c++11',
	'-Wno-missing-field-initializers',
]

grpc_c_args = base_c_args + [
	'-Wno-null-dereference',
]

grpc_cpp_args = base_cpp_args + [
	'-Wno-null-dereference',
]

if get_option('enable_gpu_support')
	gpu_c_args = base_c_args + gpu_compile_flags
	gpu_cpp_args = base_cpp_args + gpu_compile_flags
	gpu_cuda_args = gpu_compile_flags
endif

# Comment this line to restore warnings of experimental DOCA features
add_project_arguments('-D DOCA_ALLOW_EXPERIMENTAL_API', language: languages)

if get_option('enable_developer_log')
	add_project_arguments('-D DOCA_LOGGING_ALLOW_DLOG', language: languages)
endif

flag_enable_driver_flexio = get_option('enable_driver_flexio')

if not flag_enable_driver_flexio
	set_variable('flexio_disable_reason', 'Disabled by user')
endif

if get_option('enable_cross_compilation_to_dpu')
	# Please update this folder if the base cross-compilation folder is located elsewhere
	add_project_arguments('-I/root/doca-cross/usr/include', language: languages)
	# DPA Compiler doesn't fully support cross compilation yet
	if flag_enable_driver_flexio
		flag_enable_driver_flexio = false
		set_variable('flexio_disable_reason', 'Cross compilation is not supported for FlexIO applications')
	endif
endif

# Resolve irrelevant compiler warnings
add_project_arguments('-Wno-format-zero-length', language: languages)

base_app_dependencies = []
base_app_dependencies += dependency('threads')
base_app_dependencies += dependency('json-c')
dependency_libbsd = dependency('libbsd', required: false)
if dependency_libbsd.found()
	base_app_dependencies += dependency_libbsd
	add_project_arguments('-D DOCA_USE_LIBBSD', language: languages)
	# Ensure mlnx-dpdk will manage to find our libbsd if exists
	add_project_arguments('-D RTE_USE_LIBBSD', language: languages)
endif

if get_option('enable_grpc_support')
	dependency_protobuf = dependency('protobuf')
	dependency_grpc = dependency('grpc++')

	protoc = find_program('protoc', version: '3.15.8')
	grpc_cpp_plugin = find_program('grpc_cpp_plugin')
	grpc_py_plugin = find_program('grpc_python_plugin')
	# Hack to force the generation of non-executable generated files
	python_generator = find_program('cp')

	grpc_dependencies = [dependency_protobuf, dependency_grpc]
endif

common_path = '/opt/mellanox/doca/applications/common/src'
common_dir_path = common_path
common_grpc_dir_path = common_dir_path + '/grpc'
base_app_inc_dirs = [
	include_directories(common_path),
	include_directories('..'),
]
samples_dir_path = '/opt/mellanox/doca/samples'

# Add the relevant GPU stuff to have a successful compilation with GPU support.
apps_gpu_common_src = files([
	common_path + '/gpu_init.c',
])

source_path = 'src'
bin_path = 'bin'

app_list = [
	    'ips',
]

fs = import('fs')
if fs.is_dir('l2_reflector')
	app_list += ['l2_reflector']
endif

if fs.is_dir('dpa_all_to_all')
	app_list += ['dpa_all_to_all']
endif

if fs.is_dir('gpu_packet_processing')
	app_list += ['gpu_packet_processing']
endif

if fs.is_dir('yara_inspection')
	app_list += ['yara_inspection']
endif

grpc_extended_apps = [
	'application_recognition',
	'dns_filter',
	'ips',
	'url_filter',
]

foreach APP_NAME : app_list
	if (not get_option('enable_' + APP_NAME) and not get_option('enable_all_applications'))
		warning('Skipping compilation of DOCA Application - @0@.'.format(APP_NAME))
		continue
	endif

	# gRPC preparation (just in case)
	app_supports_grpc = grpc_extended_apps.contains(APP_NAME) and get_option('enable_grpc_support')
	if app_supports_grpc
		common_grpc = '/'.join(['common', source_path, 'grpc'])
		common_proto = '/'.join([common_grpc, 'common.proto'])
		app_proto = APP_NAME + '.proto'

		# Create absolute grpc paths
		grpc_base_path   = '/'.join([APP_NAME, source_path, 'grpc'])
		grpc_server_path = '/'.join([grpc_base_path, 'server'])
		grpc_client_path = '/'.join([grpc_base_path, 'client'])

		bin_install_grpc_base_dir   = '/'.join([APP_NAME, bin_path, 'grpc'])
		bin_install_grpc_server_dir = '/'.join([bin_install_grpc_base_dir, 'server'])
		bin_install_grpc_client_dir = '/'.join([bin_install_grpc_base_dir, 'client'])

		# Generate the gRPC Server
		common_gen_cc = generator(protoc, \
					  output    : ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
					  arguments : ['--proto_path=@SOURCE_ROOT@/' + common_grpc,
						       '--cpp_out=@BUILD_DIR@',
						       '@INPUT@'])

		orchestration_gen_cc = generator(protoc, \
						 output    : ['@BASENAME@.grpc.pb.cc', '@BASENAME@.grpc.pb.h'],
						 arguments : ['--proto_path=@SOURCE_ROOT@/' + common_grpc,
							      '--grpc_out=@BUILD_DIR@',
							      '--plugin=protoc-gen-grpc=' + grpc_cpp_plugin.full_path(),
							      '@INPUT@'])

		proto_gen_cc = generator(protoc, \
					 output    : ['@BASENAME@.pb.cc', '@BASENAME@.pb.h'],
					 arguments : ['--proto_path=@SOURCE_ROOT@/' + grpc_base_path,
						      '--proto_path=@SOURCE_ROOT@/' + common_grpc,
						      '--cpp_out=@BUILD_DIR@',
						      '@INPUT@'])

		grpc_gen_cc = generator(protoc, \
					output    : ['@BASENAME@.grpc.pb.cc', '@BASENAME@.grpc.pb.h'],
					arguments : ['--proto_path=@SOURCE_ROOT@/' + grpc_base_path,
						     '--proto_path=@SOURCE_ROOT@/' + common_grpc,
						     '--grpc_out=@BUILD_DIR@',
						     '--plugin=protoc-gen-grpc=' + grpc_cpp_plugin.full_path(),
						     '@INPUT@'])

		gen_common_cc = common_gen_cc.process(common_proto)
		gen_orchestration_cc = orchestration_gen_cc.process(common_proto)
		gen_proto_cc = proto_gen_cc.process(grpc_base_path + '/' + app_proto)
		gen_grpc_cc = grpc_gen_cc.process(grpc_base_path + '/' + app_proto)

		# Generate the gRPC Client
		common_gen_py = generator(protoc, \
					  output    : ['@BASENAME@_pb2.py'],
					  arguments : ['--proto_path=@SOURCE_ROOT@/' + common_grpc,
						       '--python_out=@BUILD_DIR@',
						       '@INPUT@'])

		proto_gen_py = generator(protoc, \
					 output    : ['@BASENAME@_pb2.py'],
					 arguments : ['--proto_path=@SOURCE_ROOT@/' + grpc_base_path,
						      '--proto_path=@SOURCE_ROOT@/' + common_grpc,
						      '--python_out=@BUILD_DIR@',
						      '@INPUT@'])

		grpc_gen_py = generator(protoc, \
					output    : ['@BASENAME@_pb2_grpc.py'],
					arguments : ['--proto_path=@SOURCE_ROOT@/' + grpc_base_path,
						     '--proto_path=@SOURCE_ROOT@/' + common_grpc,
						     '--grpc_out=@BUILD_DIR@',
						     '--plugin=protoc-gen-grpc=' + grpc_py_plugin.full_path(),
						     '@INPUT@'])

		gen_common_py = common_gen_py.process(common_proto)
		gen_proto_py = proto_gen_py.process(grpc_base_path + '/' + app_proto)
		gen_grpc_py = grpc_gen_py.process(grpc_base_path + '/' + app_proto)
	endif

	# Basic app definitions
	app_dependencies = base_app_dependencies
	app_inc_dirs = base_app_inc_dirs
	app_libs = []
	app_srcs = []

	# Enter the application's directory
	subdir(APP_NAME + '/' + source_path)

endforeach
