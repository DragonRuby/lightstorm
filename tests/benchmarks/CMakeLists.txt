function(add_bench_executable ruby)
  set(full_ruby_path ${CMAKE_CURRENT_LIST_DIR}/${ruby})
  set(host_c ${CMAKE_CURRENT_BINARY_DIR}/bytecode_${ruby}.c)
  set(target_name bench_${ruby}.exe)
  add_custom_command(
    OUTPUT ${host_c}
    COMMAND $<TARGET_FILE:mrbc_binary> -Blightstorm_bench -o ${host_c}
            ${full_ruby_path}
    DEPENDS ${full_ruby_path})
  add_executable(${target_name} benchmarks_bytecode_main.c ${host_c}
                                ${CMAKE_SOURCE_DIR}/simple_kpc.c)
  target_include_directories(
    ${target_name}
    PRIVATE ${CMAKE_SOURCE_DIR}/third_party/mruby/include ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/third_party/mruby/build/host/include)
  target_compile_options(${target_name} PRIVATE -g)
  target_link_libraries(${target_name} PRIVATE mruby_static m)
  add_dependencies(${target_name} mruby_static)
endfunction()

file(
  GLOB files
  RELATIVE ${CMAKE_CURRENT_LIST_DIR}
  "${CMAKE_CURRENT_LIST_DIR}/*.rb")

foreach(file ${files})
  add_lightstorm_executable(${file})
  add_bench_executable(${file})
  set(bench_targets ${bench_targets} $<TARGET_FILE:${file}.exe>
                    $<TARGET_FILE:bench_${file}.exe>)
endforeach()

add_custom_target(
  run-benchmarks COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/bench.py
                         ${bench_targets})
