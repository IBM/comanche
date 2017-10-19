AC_INIT([comanche],[0.1])
AC_LANG(C++)

AC_CHECK_LIB([numa],[tolower],[],AC_MSG_ERROR([libnuma-dev not installed]))

AC_CONFIG_FILES([
  Makefile.gmake
  unit_test/Makefile.gmake
])

AC_ARG_ENABLE(mlnx, [AS_HELP_STRING([--enable-mlnx],[enable Mellanox driver integration])],
[enable_mlnx=true],[enable_mlnx=false])


AC_SUBST(use_rdma, $enable_rdma)
AC_SUBST(use_mlnx, $enable_mlnx)  

AC_OUTPUT
