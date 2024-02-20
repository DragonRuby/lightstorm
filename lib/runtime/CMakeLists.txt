add_library(lightstorm_runtime STATIC
  lightstorm_runtime.c
)
target_include_directories(lightstorm_runtime
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/lightstorm/runtime
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include
)
target_compile_options(lightstorm_runtime PRIVATE ${LIGHTSTORM_CFLAGS})
target_link_options(lightstorm_runtime PRIVATE ${LIGHTSTORM_CFLAGS})
if (LIGHTSTORM_LTO)
  target_compile_definitions(lightstorm_runtime PRIVATE LS_NO_CATCH)
endif()
add_dependencies(lightstorm_runtime mruby_static)

add_library(lightstorm_runtime_main STATIC
  lightstorm_runtime_main.c
)
target_include_directories(lightstorm_runtime_main
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/include
  ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include
)
target_compile_options(lightstorm_runtime_main PRIVATE ${LIGHTSTORM_CFLAGS})
target_link_options(lightstorm_runtime_main PRIVATE ${LIGHTSTORM_CFLAGS})
add_dependencies(lightstorm_runtime_main mruby_static)
