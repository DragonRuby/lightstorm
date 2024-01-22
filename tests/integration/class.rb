class Adder
  def add(a, b)
    a + b
  end
end

adder = Adder.new
puts adder.add(2, 42)

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:44
