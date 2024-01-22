puts 5 > 10
puts 5 < 10
puts 5 >= 10
puts 5 <= 10
puts 5 == 10

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:false
# CHECK:true
# CHECK:false
# CHECK:true
# CHECK:false
