include(ExternalProject)
set (MRUBY_DIR "${CMAKE_SOURCE_DIR}/third_party/mruby")
set (MRUBY_BINARY "${MRUBY_DIR}/bin/mruby")
set (MRUBY_STATIC "${MRUBY_DIR}/build/host/lib/libmruby.a")

ExternalProject_Add(mruby
  SOURCE_DIR ${MRUBY_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${CMAKE_COMMAND} -E env CFLAGS="-flto" rake all --verbose
  BUILD_IN_SOURCE ON
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${MRUBY_STATIC} ${MRUBY_BINARY}
  BUILD_ALWAYS
)

add_executable(mruby_binary IMPORTED GLOBAL)
set_property(TARGET mruby_binary PROPERTY IMPORTED_LOCATION ${MRUBY_BINARY})
add_dependencies(mruby_binary mruby)

add_library(mruby_static STATIC IMPORTED GLOBAL)
set_property(TARGET mruby_static PROPERTY IMPORTED_LOCATION ${MRUBY_STATIC})
add_dependencies(mruby_static mruby)
