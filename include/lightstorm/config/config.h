#pragma once

#include <string>

namespace lightstorm {

struct LightstormConfig {
  bool verbose = false;
  std::string entry;
  std::string runtime_source_location;
};

} // namespace lightstorm
