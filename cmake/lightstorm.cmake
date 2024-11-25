function(add_lightstorm_executable ruby)
  set(in_ruby ${CMAKE_CURRENT_LIST_DIR}/${ruby})
  set(out_c ${CMAKE_CURRENT_BINARY_DIR}/${ruby}.c)
  set(target_name ${ruby}.exe)
  add_custom_command(
    OUTPUT ${out_c}
    COMMAND $<TARGET_FILE:lightstorm> --no-opt ${in_ruby} -o ${out_c}
    DEPENDS ${in_ruby} lightstorm)
  add_executable(${target_name} ${out_c})
  target_compile_options(${target_name} PRIVATE ${LIGHTSTORM_CFLAGS})
  target_link_options(${target_name} PRIVATE ${LIGHTSTORM_CFLAGS})
  target_link_libraries(${target_name} PRIVATE mruby_static
                                               lightstorm_runtime_main)
  target_include_directories(
    ${target_name}
    PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/third_party/mruby/include
            ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include)
  add_dependencies(${target_name} mruby_static)
endfunction()
