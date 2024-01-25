puts [1, 2, 3]
puts [1, 2]

*c = [[]]
p c

a = [1, 2, 3, 4, 5, 6, 7]
b, c, *d, e, f = a

puts b
puts c
puts d
puts e
puts f

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:[1, 2, 3]
# CHECK-NEXT:[1, 2]
# CHECK-NEXT:[[]]
# CHECK-NEXT:1
# CHECK-NEXT:2
# CHECK-NEXT:[3, 4, 5]
# CHECK-NEXT:6
# CHECK-NEXT:7
