function(add_end2end_test dir)
  set(host_ruby ${CMAKE_CURRENT_LIST_DIR}/${dir}/host.rb)
  set(lightstorm_ruby ${CMAKE_CURRENT_LIST_DIR}/${dir}/lightstorm.rb)
  set(host_c ${CMAKE_CURRENT_BINARY_DIR}/${dir}/host.c)
  set(lightstorm_c ${CMAKE_CURRENT_BINARY_DIR}/${dir}/lightstorm.c)
  set(target_name end2end_test_${dir}.exe)
  add_custom_command(
    OUTPUT ${lightstorm_c}
    COMMAND $<TARGET_FILE:lightstorm> ${lightstorm_ruby} --runtime-dir
            ${CMAKE_SOURCE_DIR}/lib/runtime -o ${lightstorm_c}
    DEPENDS ${lightstorm_ruby} lightstorm)
  add_custom_command(
    OUTPUT ${host_c}
    COMMAND $<TARGET_FILE:mrbc_binary> -Blightstorm_host -o ${host_c}
            ${host_ruby}
    DEPENDS ${host_ruby})
  add_executable(${target_name} end2end_test_main.c ${host_c} ${lightstorm_c})
  target_include_directories(
    ${target_name}
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/mruby/include
            ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include)
  target_compile_options(${target_name} PRIVATE -g)
  target_link_libraries(${target_name} PRIVATE mruby_static
                                               lightstorm_runtime_main)
  add_dependencies(${target_name} mruby_static)
endfunction()

add_end2end_test(00)
add_end2end_test(geometry_quadtree)
