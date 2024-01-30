class Adder
  def initialize(a, b)
    @a = a
    @b = b
  end
  def add
    @a + @b
  end

  alias sum add
  # throws an exception
  # undef sum
end

puts Adder.new(2, 42).add
puts Adder.new(15, 30).add

class A; class ::C; end end

puts C

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:44
# CHECK-NEXT:45
# CHECK-NEXT:C
