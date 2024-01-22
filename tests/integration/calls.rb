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

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:44
