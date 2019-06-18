# template file for setup.py
# see https://docs.python.org/2/distutils/apiref.html
# use CC="ccache gcc"
# use CXX=""

import os
from distutils.core import setup, Extension
import distutils.sysconfig

# remove the pesky -Wstrict-prototypes flag!
#
cfg_vars = distutils.sysconfig.get_config_vars()
include_dir = '/usr/include/python3.6'

for key, value in cfg_vars.items():
    if(key == 'OPT'):
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")
        cfg_vars[key] = value.replace("-O0", "-O2")
        cfg_vars[key] += " -Wno-write-strings"  # add option to disable write strings warning
    elif(key == 'Py_DEBUG'):
        if(value == 1):
            include_dir = '/usr/include/python3.6dm'


module_core = Extension('dawn',
                        define_macros = [('MAJOR_VERSION', '1'),
                                         ('CXX',''),
                                         ('MINOR_VERSION', '0')],
                        include_dirs = [include_dir,
                                        '/usr/local/lib/python3.6/dist-packages/numpy/core/include',
                                        '/home/dnb/proj/comanche/src/python/dawn-api/../../../src/components',
                                        '/home/dnb/proj/comanche/src/python/dawn-api/../../../src/lib/common/include',
                                        '/home/dnb/proj/comanche/src/python/dawn-api/../../../src/lib/core/include',
                                        '/home/dnb/proj/comanche/src/python/dawn-api/src'
                        ],
                        library_dirs = ['/usr/lib/python3.6m/',
                                        '/home/dnb/proj/comanche/src/python/dawn-api/../../lib/common',
                                        '/home/dnb/proj/comanche/src/python/dawn-api/../../lib/core',
                                        '/home/dnb/proj/comanche/src/python/dawn-api/dist/lib',
                                        '/home/dnb/proj/comanche/src/python/dawn-api/dist/lib64',
                                        '/home/dnb/proj/comanche/build/dist/lib',
                                        '/usr/local/lib',
                        ],
                        libraries = ['common','comanche-core','numa','pthread','z','rt','dl',
                                     'boost_system',
                        ],
                        sources = ['/home/dnb/proj/comanche/src/python/dawn-api/src/dawn_api_module.cc',
                                   '/home/dnb/proj/comanche/src/python/dawn-api/src/string_type.cc',
                                   '/home/dnb/proj/comanche/src/python/dawn-api/src/session_type.cc',
                                   '/home/dnb/proj/comanche/src/python/dawn-api/src/pool_type.cc',
                        ],
                        extra_compile_args = ['-fPIC','-std=c++11','-DCONFIG_DEBUG','-g','-O2'],
                        runtime_library_dirs = ['/usr/lib/python3.6m/',
                                                '/home/dnb/proj/comanche/src/python/dawn-api/dist/lib',
                                                '/home/dnb/proj/comanche/src/python/dawn-api/dist/lib64',
                                                '/usr/local/lib',
                                                '-Wl,--rpath=/home/dnb/proj/comanche/src/python/dawn-api/dist/lib'
                        ]
)


setup (name = 'DawnApiPackage',
       version = '0.1',
       author = 'Daniel Waddington',
       author_email = 'daniel.waddington@ibm.com',
       description = 'Dawn-KV API library',
       package_dir={ '': '/home/dnb/proj/comanche/src/python/dawn-api' },
       ext_modules = [module_core]
)



