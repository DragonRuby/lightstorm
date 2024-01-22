de s
end

# RUN: %mruby %s > %t.log 2>&1; cat %t.log | %filecheck %s %fcheck_opts
# RUN: %lightstorm %s > %t.log 2>&1; cat %t.log | %filecheck %s %fcheck_opts

# CHECK:{{.*}}syntax.rb:2:3: syntax error, unexpected keyword_end, expecting $end
