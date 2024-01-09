function(add_end2end_test ruby)
  set (in_ruby ${CMAKE_CURRENT_LIST_DIR}/${ruby})
  set (out_c ${CMAKE_CURRENT_BINARY_DIR}/${ruby}.c)
  set (target_name ${ruby}.exe)
  add_custom_command(OUTPUT ${out_c}
    COMMAND $<TARGET_FILE:lightstorm> ${in_ruby} -o ${out_c}
    DEPENDS ${in_ruby} lightstorm
  )
  add_executable(${target_name} ${out_c})
  target_link_libraries(${target_name} PRIVATE mruby_static lightstorm_runtime)
endfunction()
add_end2end_test(00.rb)
