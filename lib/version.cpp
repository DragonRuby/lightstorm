#include "lightstorm/version.h"

#include <llvm/Support/raw_ostream.h>

namespace lightstorm {

static const char *version_string() {
  return "@PROJECT_VERSION@";
}
static const char *commit_string() {
  return "@GIT_COMMIT@";
}
static const char *build_date_string() {
  return "@BUILD_DATE@";
}
static const char *description_string() {
  return "@PROJECT_DESCRIPTION@";
}
static const char *homepage_string() {
  return "@PROJECT_HOMEPAGE_URL@";
}
static const char *llvm_version_string() {
  return "@LLVM_VERSION@";
}
static const char *mruby_version_string() {
  return "@MRUBY_VERSION@";
}

void print_version_information(llvm::raw_ostream &out) {
  out << "Lightstorm: " << description_string() << "\n";
  out << "Home: " << homepage_string() << "\n";
  out << "Version: " << version_string() << " (LLVM " << llvm_version_string() << ", mruby "
      << mruby_version_string() << ")" << "\n";
  out << "Commit: " << commit_string() << "\n";
  out << "Date: " << build_date_string() << "\n";
}

} // namespace lightstorm
