class Computer
  def initialize(x, y, z)
    @x = x
    @y = y
    @z = z
  end

  def product
    @x * @y * @z
  end
end

m = Computer.new(2.0, 3.0, 4.0)
p m.product

# UN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:24.0
