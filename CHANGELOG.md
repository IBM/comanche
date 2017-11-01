# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

Daniel Waddington (daniel.waddington@ibm.com)

## [v0.2.0]
## 2017-10
### Changed
- Default read/write sync operations on IBlock_device now use a semaphore and thus do a sleep-based wait.
- Added second call back argument to IBlock_device asynchronous methods.  This allows this pointer passing.
- Rework of CMakeLists.txt for components to improve common.cmake and use COMANCHE_HOME env variable
- Break out components into family directories
### Added
- Initial RAID0 component. Untested.
- Zero-copy read iterator to Append-Store
- Support for callbacks to async_read/write methods on IBlock_device
- Sample component
## 2017-9
### Changed
- Major overhaul of the build/linking arrangement.  SPDK and DPDK are now compiled into the comanche-core dynamic library.  All components needing access to DPDK/SPDK should link to this library.
- Updated to DPDK 17.08 and SPDK (tag fca11f1)
- Changed libcommon.a to dynamic library so that its license (LGPL) can be kept separate.
- Block-nvme component; major rework around out-of-order operation fixes.
- Unified cmake build system.  CMakeLists.txt should include common.cmake, e.g.,include(../../../mk/common.cmake)
- Debug, Release or ASAN (Address Sanitizer) builds configured through: cmake -DCMAKE_BUILD_TYPE=<DEBUG|ASAN|RELEASE> .
- Cmake should be run for root to ensure consistency
### Added
- Placeholder for shared work hooks in IBlock_device interface
- Jenkins continuous integration support for build testing
- Address santizer (ASAN) build support; used for memory leak debugging. Note: ASAN may not work with components t that perform memory tricks.
- Support for DPDK memory in block-posix block device.  This allows it to be used with the pmem-paged-posix component.
- Patch for DPDK 17.05.1 to support XMS-module based physical address resolution.
- Updates to fetch-deps script to perform DPDK patch.
- Removed rapidjson from the fetch-deps into a install-apts (i.e. package). Make sure you delete /usr/local/include/rapidjson from previous installations.
- Improved load component to search current path and COMANCHE_HOME path.
## 2017-8-30
### Changed
- The pmem-paged-posix uses POSIX APIs and XMS module (to get virtual-physical address mappings.  Only the posix-nvme block device current works with the pager-simple. Do NOT attempt to use the uNVMe based block device - this will result in memory corruption.




