class A
  def initialize
  end
end

class B
  def initialize
  end
end

puts A
puts B

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

class C; class ::D; end end

puts C
puts D

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:A
# CHECK-NEXT:B
# CHECK-NEXT:44
# CHECK-NEXT:45
# CHECK-NEXT:C
# CHECK-NEXT:D
