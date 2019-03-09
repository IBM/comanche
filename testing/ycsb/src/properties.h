#ifndef __YCSB_PROPERTIES_H__
#define __YCSB_PROPERTIES_H__

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>

using namespace std;

namespace ycsbutils
{
class Properties {
 public:
  std::string getProperty(
      const std::string &key,
      const std::string &default_value = std::string()) const;
  const std::string &operator[](const std::string &key) const;
  const std::map<std::string, std::string> &properties() const;

  void setProperty(const std::string &key, const std::string &value);
  bool load(std::ifstream &input);

 private:
  std::map<std::string, std::string> properties_;
  std::string                        trim(const std::string &str);
};

inline std::string Properties::getProperty(
    const std::string &key,
    const std::string &default_value) const
{
  std::map<std::string, std::string>::const_iterator it = properties_.find(key);
  if (properties_.end() == it) {
    return default_value;
  }
  else
    return it->second;
}

inline const std::string &Properties::operator[](const std::string &key) const
{
  return properties_.at(key);
}

inline const std::map<std::string, std::string> &Properties::properties() const
{
  return properties_;
}

inline void Properties::setProperty(const std::string &key,
                                    const std::string &value)
{
  properties_[key] = value;
}

inline bool Properties::load(std::ifstream &input)
{
  if (!input.is_open()) throw "File not open!";

  while (!input.eof() && !input.bad()) {
    std::string line;
    std::getline(input, line);
    if (line[0] == '#') continue;
    size_t pos = line.find_first_of('=');
    if (pos == std::string::npos) continue;
    setProperty(trim(line.substr(0, pos)), trim(line.substr(pos + 1)));
  }
  return true;
}

inline std::string Properties::trim(const std::string &str)
{
  auto front = std::find_if_not(str.begin(), str.end(),
                                [](int c) { return std::isspace(c); });
  return std::string(
      front,
      std::find_if_not(str.rbegin(), std::string::const_reverse_iterator(front),
                       [](int c) { return std::isspace(c); })
          .base());
}
const uint64_t kFNVOffsetBasis64 = 0xCBF29CE484222325;
const uint64_t kFNVPrime64       = 1099511628211;

inline uint64_t FNVHash64(uint64_t val)
{
  uint64_t hash = kFNVOffsetBasis64;

  for (int i = 0; i < 8; i++) {
    uint64_t octet = val & 0x00ff;
    val            = val >> 8;

    hash = hash ^ octet;
    hash = hash * kFNVPrime64;
  }
  return hash;
}

inline uint64_t Hash(uint64_t val) { return FNVHash64(val); }

inline double RandomDouble(double min = 0.0, double max = 1.0)
{
  static std::default_random_engine             generator;
  static std::uniform_real_distribution<double> uniform(min, max);
  return uniform(generator);
}

///
/// Returns an ASCII code that can be printed to desplay
///
inline char RandomPrintChar() { return rand() % 94 + 33; }
}  // namespace ycsbutils

#endif


