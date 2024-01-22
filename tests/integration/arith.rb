i = 42
puts i + 2
puts i + i
puts i - 1
puts i - i
puts i * 1
puts i * i
puts 2 / i
puts i / i

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:44
# CHECK-NEXT:84
# CHECK-NEXT:41
# CHECK-NEXT:0
# CHECK-NEXT:42
# CHECK-NEXT:1764
# CHECK-NEXT:0
# CHECK-NEXT:1
