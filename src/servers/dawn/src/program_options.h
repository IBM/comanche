#ifndef __DAWN_PROGRAM_OPTIONS_H__
#define __DAWN_PROGRAM_OPTIONS_H__

struct Program_options {
  std::string config_file;
  std::string device;
  std::string backend;
  std::string pci_addr;
  unsigned debug_level;
  bool forced_exit;
};

#endif  // __DAWN_PROGRAM_OPTIONS_H__
