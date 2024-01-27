puts (1..10).to_a
puts (1...10).to_a

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# CHECK-NEXT:[1, 2, 3, 4, 5, 6, 7, 8, 9]
