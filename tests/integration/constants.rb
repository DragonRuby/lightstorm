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

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:GLOBAL VAR
# CHECK-NEXT:instance var
