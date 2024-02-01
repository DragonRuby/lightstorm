module M
  CONST = 1 << 29
  def self.f
  end
end

p M::CONST

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:536870912
