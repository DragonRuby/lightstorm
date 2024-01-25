$global = "GLOBAL VAR"
puts $global

class ClassVar
  def a=(a)
    @a_var = a
  end

  def a
    @a_var
  end
end

cv = ClassVar.new
cv.a = "instance var"
puts cv.a

class A
  @@b = "A@@b"
  def self.b=(b)
    @@b = b
  end
  def self.b
    @@b
  end
end

puts A.b
A.b = "A@@b2"
puts A.b

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:GLOBAL VAR
# CHECK-NEXT:instance var
# CHECK-NEXT:A@@b
# CHECK-NEXT:A@@b2
