class Adder
  def add(a, b)
    a + b
  end

  alias sum add
  # throws an exception
  # undef sum
end

adder = Adder.new
puts adder.add(2, 42)
puts adder.add(15, 30)

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:44
# CHECK-NEXT:45
