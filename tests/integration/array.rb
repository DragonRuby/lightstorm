puts [1, 2, 3]
puts [1, 2]

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:[1, 2, 3]
# CHECK-NEXT:[1, 2]
