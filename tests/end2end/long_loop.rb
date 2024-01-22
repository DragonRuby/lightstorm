i = 0
while i < 1_000_000
  i += 1
end
puts i

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:1000000
