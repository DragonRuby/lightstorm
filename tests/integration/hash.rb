x = { a: 1, b: 2, c: 3 }
puts x

h1 = {:s => 0}
h2 = {:t => 1}
h3 = {**h1, **h2}
puts h3

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:{:a=>1, :b=>2, :c=>3}
# CHECK-NEXT:{:s=>0, :t=>1}
