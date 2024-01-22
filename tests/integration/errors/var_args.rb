
def a(*x, b)
end

# RUN: %lightstorm %s > %t.log 2>&1; cat %t.log | %filecheck %s %fcheck_opts

# CHECK:{{.*}}_args.rb:2: Variable arguments are not supported yet
