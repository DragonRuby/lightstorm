function(add_end2end_test ruby)
  set (in_ruby ${CMAKE_CURRENT_LIST_DIR}/${ruby})
  set (out_c ${CMAKE_CURRENT_BINARY_DIR}/${ruby}.c)
  set (target_name ${ruby}.exe)
  add_custom_command(OUTPUT ${out_c}
    COMMAND $<TARGET_FILE:lightstorm> ${in_ruby} -o ${out_c}
    DEPENDS ${in_ruby} lightstorm
  )
  add_executable(${target_name} ${out_c})
  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Output
    # tmp is needed by LIT
    OUTPUT_NAME ${ruby}.tmp.exe
  )
  target_compile_options(${target_name} PRIVATE -flto -O3)
  target_link_libraries(${target_name} PRIVATE mruby_static lightstorm_runtime lightstorm_runtime_main)
  add_dependencies(${target_name} mruby_static)
  set (test_target_names ${test_target_names} "$<TARGET_FILE:${target_name}>" PARENT_SCOPE)
endfunction()

add_end2end_test(loads.rb)
add_end2end_test(arith.rb)
add_end2end_test(array.rb)
add_end2end_test(cmp.rb)
add_end2end_test(jumps.rb)
add_end2end_test(long_loop.rb)
add_end2end_test(calls.rb)
add_end2end_test(hash.rb)

message("...${test_target_names}...")
add_custom_target(run-end2end-tests
  COMMAND env MRUBY_BINARY=$<TARGET_FILE:mruby_binary> FILECHECK_BINARY=filecheck lit -vv ${CMAKE_CURRENT_LIST_DIR}
  DEPENDS ${test_target_names}
)
