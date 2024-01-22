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
# CHECK:84
# CHECK:41
# CHECK:0
# CHECK:42
# CHECK:1764
# CHECK:0
# CHECK:1
