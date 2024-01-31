function(add_benchmark_test ruby)
  set (in_ruby ${CMAKE_CURRENT_LIST_DIR}/${ruby})
  set (out_c ${CMAKE_CURRENT_BINARY_DIR}/${ruby}.c)
  set (target_name ${ruby}.exe)
  add_custom_command(OUTPUT ${out_c}
    COMMAND $<TARGET_FILE:lightstorm> ${in_ruby} -o ${out_c}
    DEPENDS ${in_ruby} lightstorm
  )
  add_executable(${target_name} ${out_c})
  target_compile_options(${target_name} PRIVATE -g)
  target_link_options(${target_name} PRIVATE -g)
  target_link_libraries(${target_name} PRIVATE mruby_static lightstorm_runtime lightstorm_runtime_main)
  add_dependencies(${target_name} mruby_static)
  add_custom_target(benchhmark-${ruby}
    COMMAND hyperfine --warmup 1
      --command-name "mruby ${ruby}" '$<TARGET_FILE:mruby_binary> ${in_ruby}'
      --command-name "lightstorm ${ruby}" $<TARGET_FILE:${target_name}>
  )
  set (test_targets ${test_targets} $<TARGET_FILE:${target_name}> ${in_ruby} PARENT_SCOPE)
#  set (test_targets $<TARGET_FILE:${target_name}> ${in_ruby} PARENT_SCOPE)
#  set (test_targets )
endfunction()

file(GLOB files RELATIVE ${CMAKE_CURRENT_LIST_DIR} "${CMAKE_CURRENT_LIST_DIR}/*.rb")
foreach(file ${files})
  add_benchmark_test(${file})
endforeach ()
add_custom_target(run-benchmarks
  COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/bench.py $<TARGET_FILE:mruby_binary> ${test_targets}
)
