file(
  GLOB files
  RELATIVE ${CMAKE_CURRENT_LIST_DIR}
  "${CMAKE_CURRENT_LIST_DIR}/*.rb")
foreach(file ${files})
  add_lightstorm_executable(${file})
  set(bench_targets ${bench_targets} $<TARGET_FILE:${file}.exe>
                    ${CMAKE_CURRENT_LIST_DIR}/${file})
endforeach()
add_custom_target(
  run-benchmarks COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/bench.py
                         $<TARGET_FILE:mruby_binary> ${bench_targets})
