$global = "GLOBAL VAR"
puts $global

# RUN: %mruby %s | %filecheck %s %fcheck_opts
# RUN: %t.exe | %filecheck %s %fcheck_opts

# CHECK:GLOBAL VAR
