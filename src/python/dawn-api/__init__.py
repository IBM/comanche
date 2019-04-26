#import kivati_core

# from __future__ import division, absolute_import, print_function

# import sys
# import warnings

# from kivati import *

# # Disallow reloading. Doing that does nothing to change previously
# # loaded modules, which would need to be reloaded separately, but it does
# # change the identity of the warnings and sentinal classes defined below
# # with dire consequences when checking for identity.
# if '_is_loaded' in globals():
#     raise RuntimeError('Reloading Kivati is not supported')
# _is_loaded = True




# class VisibleDeprecationWarning(UserWarning):
#     """Visible deprecation warning.

#     By default, python will not show deprecation warnings, so this class
#     can be used when a very visible warning is helpful, for example because
#     the usage is most likely a user bug.

#     """
#     pass


# class _NoValue:
#     """Special keyword value.

#     This class may be used as the default value assigned to a deprecated
#     keyword in order to check if it has been given a user defined value.
#     """
#     pass


# # oldnumeric and numarray were removed in 1.9. In case some packages import
# # but do not use them, we define them here for backward compatibility.
# oldnumeric = 'removed'
# numarray = 'removed'


# # We first need to detect if we're being called as part of the numpy setup
# # procedure itself in a reliable manner.
# try:
#     __NUMPY_SETUP__
# except NameError:
#     __NUMPY_SETUP__ = False

# if __NUMPY_SETUP__:
#     sys.stderr.write('Running from numpy source directory.\n')
# else:
#     try:
#         from numpy.__config__ import show as show_config
#     except ImportError:
#         msg = """Error importing numpy: you should not try to import numpy from
#         its source directory; please exit the numpy source tree, and relaunch
#         your python interpreter from there."""
#         raise ImportError(msg)
#     from .version import git_revision as __git_revision__
#     from .version import version as __version__

#     from ._import_tools import PackageLoader

#     def pkgload(*packages, **options):
#         loader = PackageLoader(infunc=True)
#         return loader(*packages, **options)

#     from . import add_newdocs
#     __all__ = ['add_newdocs',
#                'ModuleDeprecationWarning',
#                'VisibleDeprecationWarning']

#     pkgload.__doc__ = PackageLoader.__call__.__doc__

#     # We don't actually use this ourselves anymore, but I'm not 100% sure that
#     # no-one else in the world is using it (though I hope not)
#     from .testing import Tester
#     test = testing.nosetester._numpy_tester().test
#     bench = testing.nosetester._numpy_tester().bench

#     # Allow distributors to run custom init code
#     from . import _distributor_init

#     from . import core
#     from .core import *
#     from . import compat
#     from . import lib
#     from .lib import *
#     from . import linalg
#     from . import fft
#     from . import polynomial
#     from . import random
#     from . import ctypeslib
#     from . import ma
#     from . import matrixlib as _mat
#     from .matrixlib import *
#     from .compat import long

#     # Make these accessible from numpy name-space
#     # but not imported in from numpy import *
#     if sys.version_info[0] >= 3:
#         from builtins import bool, int, float, complex, object, str
#         unicode = str
#     else:
#         from __builtin__ import bool, int, float, complex, object, unicode, str

#     from .core import round, abs, max, min

#     __all__.extend(['__version__', 'pkgload', 'PackageLoader',
#                'show_config'])
#     __all__.extend(core.__all__)
#     __all__.extend(_mat.__all__)
#     __all__.extend(lib.__all__)
#     __all__.extend(['linalg', 'fft', 'random', 'ctypeslib', 'ma'])


#     # Filter annoying Cython warnings that serve no good purpose.
#     warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#     warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#     warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
