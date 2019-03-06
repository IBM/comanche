#ifndef __YCSB_PROPERTIES_H_
#define __YCSB_PROPERTIES_H_

#include <string>
#include <map>
#include <fstream>
#include <cassert>

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
  if (!input.is_open()) throw utils::Exception("File not open!");

  while (!input.eof() && !input.bad()) {
    std::string line;
    std::getline(input, line);
    if (line[0] == '#') continue;
    size_t pos = line.find_first_of('=');
    if (pos == std::string::npos) continue;
    SetProperty(Trim(line.substr(0, pos)), Trim(line.substr(pos + 1)));
  }
  return true;
}
}  // namespace ycsbutils

#endif


