puts [1, 2, 3]
puts [1, 2]

*c = [[]]
p c

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:[1, 2, 3]
# CHECK-NEXT:[1, 2]
# CHECK-NEXT:[[]]
