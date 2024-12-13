include(ExternalProject)
set(MRUBY_DIR "${CMAKE_SOURCE_DIR}/third_party/mruby")
set(MRUBY_BINARY "${MRUBY_DIR}/bin/mruby")
set(MRBC_BINARY "${MRUBY_DIR}/bin/mrbc")
set(MRUBY_STATIC "${MRUBY_DIR}/build/host/lib/libmruby.a")

file(READ ${MRUBY_DIR}/include/mruby/version.h mruby_version_h)

string(REGEX MATCH "MRUBY_RELEASE_MAJOR ([0-9]*)" _ ${mruby_version_h})
set(MRUBY_RELEASE_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "MRUBY_RELEASE_MINOR ([0-9]*)" _ ${mruby_version_h})
set(MRUBY_RELEASE_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "MRUBY_RELEASE_TEENY ([0-9]*)" _ ${mruby_version_h})
set(MRUBY_RELEASE_TEENY ${CMAKE_MATCH_1})

ExternalProject_Add(
  mruby
  SOURCE_DIR ${MRUBY_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CPP=${CMAKE_CXX_COMPILER}
    CFLAGS="${LIGHTSTORM_CFLAGS}" LDFLAGS="${LIGHTSTORM_CFLAGS}" rake all
    --verbose
  BUILD_IN_SOURCE ON
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${MRUBY_STATIC} ${MRUBY_BINARY} ${MRBC_BINARY}
  BUILD_ALWAYS)

add_executable(mruby_binary IMPORTED GLOBAL)
set_property(TARGET mruby_binary PROPERTY IMPORTED_LOCATION ${MRUBY_BINARY})
add_dependencies(mruby_binary mruby)
install(
  FILES $<TARGET_FILE:mruby_binary>
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  PERMISSIONS
    OWNER_EXECUTE
    OWNER_WRITE
    OWNER_READ
    GROUP_EXECUTE
    GROUP_READ
    WORLD_EXECUTE
    WORLD_READ
  RENAME lightstorm-mruby)

add_executable(mrbc_binary IMPORTED GLOBAL)
set_property(TARGET mrbc_binary PROPERTY IMPORTED_LOCATION ${MRBC_BINARY})
add_dependencies(mrbc_binary mruby)
install(
  FILES $<TARGET_FILE:mrbc_binary>
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  PERMISSIONS
    OWNER_EXECUTE
    OWNER_WRITE
    OWNER_READ
    GROUP_EXECUTE
    GROUP_READ
    WORLD_EXECUTE
    WORLD_READ
  RENAME lightstorm-mrbc)

add_library(mruby_static STATIC IMPORTED GLOBAL)
set_property(TARGET mruby_static PROPERTY IMPORTED_LOCATION ${MRUBY_STATIC})
add_dependencies(mruby_static mruby)
install(
  FILES $<TARGET_FILE:mruby_static>
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RENAME liblightstorm_mruby.a)
