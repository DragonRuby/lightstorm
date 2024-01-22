

def b(key: "word")
end

# RUN: %lightstorm %s > %t.log 2>&1; cat %t.log | %filecheck %s %fcheck_opts

# CHECK:{{.*}}_args.rb:3: Keyword arguments are not supported yet
