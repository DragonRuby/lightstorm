#include <mruby.h>
#include <mruby/irep.h>
#include <mruby/proc.h>

extern const uint8_t lightstorm_bench[];

int main() {
  mrb_state *mrb = mrb_open();
  mrb_load_irep(mrb, lightstorm_bench);
  mrb_close(mrb);
  return 0;
}
