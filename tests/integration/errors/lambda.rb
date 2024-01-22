
my_lambda = -> { }

# RUN: %lightstorm %s > %t.log 2>&1; cat %t.log | %filecheck %s %fcheck_opts

# CHECK:{{.*}}.rb:2: Lambdas are not supported
