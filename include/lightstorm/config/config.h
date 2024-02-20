#pragma once

#include <string>

namespace lightstorm {

struct LightstormConfig {
  bool verbose = false;
  std::string entry;
  std::string runtime_header_location;
};

} // namespace lightstorm
