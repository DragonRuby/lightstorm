#pragma once

namespace llvm {
class raw_ostream;
}

namespace lightstorm {
void print_version_information(llvm::raw_ostream &out);
} // namespace lightstorm
