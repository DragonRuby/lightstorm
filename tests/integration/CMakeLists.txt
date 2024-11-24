function(add_integration_test ruby)
  add_lightstorm_executable(${ruby})
  set(target_name ${ruby}.exe)
  set_target_properties(
    ${target_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/Output
               # tmp is needed by LIT
               OUTPUT_NAME ${ruby}.tmp.exe)
  set(test_target_names
      ${test_target_names} "$<TARGET_FILE:${target_name}>"
      PARENT_SCOPE)
endfunction()

file(
  GLOB files
  RELATIVE ${CMAKE_CURRENT_LIST_DIR}
  "${CMAKE_CURRENT_LIST_DIR}/*.rb")
foreach(file ${files})
  add_integration_test(${file})
endforeach()

add_custom_target(
  run-integration-tests
  COMMAND
    env MRUBY_BINARY=$<TARGET_FILE:mruby_binary> FILECHECK_BINARY=filecheck
    LIGHTSTORM_BINARY=$<TARGET_FILE:lightstorm> lit -vv
    ${CMAKE_CURRENT_LIST_DIR}
  DEPENDS ${test_target_names} $<TARGET_FILE:lightstorm>)
