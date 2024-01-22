if 5 < 10
  puts 42
end

if 5.nil?
  puts 5
else
  puts !5
end

i = 0
while i < 10
  i += 1
end
puts i

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:42
# CHECK:false
# CHECK:10
