#pragma once

#include <filesystem>

namespace lightstorm {

class Compiler {
public:
  void compileSourceFile(const std::filesystem::path &file_path);

private:
};

} // namespace lightstorm
