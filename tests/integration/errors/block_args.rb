def b(a, &block)
end

# RUN: %lightstorm %s > %t.log 2>&1; cat %t.log | %filecheck %s %fcheck_opts

# CHECK:{{.*}}_args.rb:1: Block arguments are not supported yet
