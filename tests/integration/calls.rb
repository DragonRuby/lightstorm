def add x, y
  x + y
end

module M
  class << self
    def f
      42
    end
  end
end

puts add(2, M::f)

puts(
    1,2,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, \
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,333,0
)

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:44
# CHECK-NEXT:1
# CHECK-NEXT:2
# CHECK-NEXT:0
# CHECK-NEXT:0
# CHECK-NEXT:5
# CHECK-NEXT:0
# CHECK:333
# CHECK-NEXT:0
