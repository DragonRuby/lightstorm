def a
  super.a
end

# RUN: %lightstorm %s > %t.log 2>&1; cat %t.log | %filecheck %s %fcheck_opts

# CHECK:{{.*}}.rb:2: Calling `super` is not supported