add_library(lightstorm_runtime_main STATIC lightstorm_runtime_main.c
                                           ${CMAKE_SOURCE_DIR}/simple_kpc.c)
target_include_directories(
  lightstorm_runtime_main
  PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}
         ${CMAKE_SOURCE_DIR}/third_party/mruby/include
         ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include)
target_compile_options(lightstorm_runtime_main PRIVATE ${LIGHTSTORM_CFLAGS})
target_link_options(lightstorm_runtime_main PRIVATE ${LIGHTSTORM_CFLAGS})
target_link_libraries(lightstorm_runtime_main PUBLIC m)
add_dependencies(lightstorm_runtime_main mruby_static)
install(TARGETS lightstorm_runtime_main)
install(FILES lightstorm_runtime.c DESTINATION ${CMAKE_INSTALL_LIBDIR})
