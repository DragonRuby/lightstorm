begin
rescue RuntimeError
end

# RUN: %lightstorm %s > %t.log 2>&1; cat %t.log | %filecheck %s %fcheck_opts

# CHECK:{{.*}}.rb:2: Exceptions are not supported
