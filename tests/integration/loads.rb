puts 1
puts (-1)
puts 42
puts (-42)
puts 1_000
puts (-1_000)
puts 1_000_000
puts (-1_000_000)
puts nil
#puts 42.0
#puts "42"
puts true
puts false
puts self
puts :hello


# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:1
# CHECK:-1
# CHECK:42
# CHECK:-42
# CHECK:1000
# CHECK:-1000
# CHECK:1000000
# CHECK:-1000000
# CHECK:
# CHECK:true
# CHECK:false
# CHECK:main
# CHECK:hello
