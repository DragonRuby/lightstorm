puts 1
puts (-1)
puts 42
puts (-42)
puts 1_000
puts (-1_000)
puts 1_000_000
puts (-1_000_000)
puts nil
puts 42.0
puts "a string #{42}"
puts true
puts false
puts self
puts :hello
puts %i(a_sym)


# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:1
# CHECK-NEXT:-1
# CHECK-NEXT:42
# CHECK-NEXT:-42
# CHECK-NEXT:1000
# CHECK-NEXT:-1000
# CHECK-NEXT:1000000
# CHECK-NEXT:-1000000
# CHECK-NEXT:
# CHECK-NEXT:42.0
# CHECK-NEXT:a string 42
# CHECK-NEXT:true
# CHECK-NEXT:false
# CHECK-NEXT:main
# CHECK-NEXT:hello
# CHECK-NEXT:[:a_sym]
