x = { a: 1, b: 2, c: 3 }
puts x

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:{:a=>1, :b=>2, :c=>3}
