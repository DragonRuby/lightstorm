#include "simple_kpc.h"

#include <assert.h>
#include <dlfcn.h>
#include <locale.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef uint16_t u16;
typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;
typedef size_t usize;

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof((x)[0]))

#define KPC_MAX_COUNTERS 32

typedef struct kpep_db kpep_db;
typedef struct kpep_config kpep_config;
typedef struct kpep_event kpep_event;

static int (*kpc_set_counting)(u32 classes);
static int (*kpc_set_thread_counting)(u32 classes);
static int (*kpc_set_config)(u32 classes, u64 *config);
static int (*kpc_get_thread_counters)(u32 tid, u32 buf_count, u64 *buf);
static int (*kpc_force_all_ctrs_set)(int val);
static int (*kpc_force_all_ctrs_get)(int *val_out);

static int (*kpep_config_create)(kpep_db *db, kpep_config **cfg_ptr);
static void (*kpep_config_free)(kpep_config *cfg);
static int (*kpep_config_add_event)(kpep_config *cfg, kpep_event **ev_ptr, u32 flag, u32 *err);
static int (*kpep_config_force_counters)(kpep_config *cfg);
static int (*kpep_config_kpc)(kpep_config *cfg, u64 *buf, usize buf_size);
static int (*kpep_config_kpc_classes)(kpep_config *cfg, u32 *classes_ptr);
static int (*kpep_config_kpc_map)(kpep_config *cfg, usize *buf, usize buf_size);

static int (*kpep_db_create)(const char *name, kpep_db **db_ptr);
static void (*kpep_db_free)(kpep_db *db);
static int (*kpep_db_event)(kpep_db *db, const char *name, kpep_event **ev_ptr);

typedef struct {
  const char *name;
  void **impl;
} symbol;

#define SYMBOL(n) { .name = #n, .impl = (void **)&n }

static const symbol KPERF_SYMBOLS[] = {
  SYMBOL(kpc_set_counting),        SYMBOL(kpc_set_thread_counting), SYMBOL(kpc_set_config),
  SYMBOL(kpc_get_thread_counters), SYMBOL(kpc_force_all_ctrs_set),  SYMBOL(kpc_force_all_ctrs_get),
};

static const symbol KPERFDATA_SYMBOLS[] = {
  SYMBOL(kpep_config_create),    SYMBOL(kpep_config_free),
  SYMBOL(kpep_config_add_event), SYMBOL(kpep_config_force_counters),
  SYMBOL(kpep_config_kpc),       SYMBOL(kpep_config_kpc_classes),
  SYMBOL(kpep_config_kpc_map),   SYMBOL(kpep_db_create),
  SYMBOL(kpep_db_free),          SYMBOL(kpep_db_event),
};

#define KPERF_PATH "/System/Library/PrivateFrameworks/kperf.framework/kperf"
#define KPERFDATA_PATH "/System/Library/PrivateFrameworks/kperfdata.framework/kperfdata"

static bool initialized = false;

void sk_init(void) {
  if (initialized)
    return;

  void *kperf = dlopen(KPERF_PATH, RTLD_LAZY);
  if (!kperf) {
    fprintf(stderr,
            "simple_kpc: failed to load kperf.framework, message: "
            "%s\n",
            dlerror());
    exit(1);
  }

  void *kperfdata = dlopen(KPERFDATA_PATH, RTLD_LAZY);
  if (!kperfdata) {
    fprintf(stderr,
            "simple_kpc: failed to load kperfdata.framework, "
            "message: %s\n",
            dlerror());
    exit(1);
  }

  for (usize i = 0; i < ARRAY_LENGTH(KPERF_SYMBOLS); i++) {
    const symbol *symbol = &KPERF_SYMBOLS[i];
    void *p = dlsym(kperf, symbol->name);
    if (!p) {
      fprintf(stderr,
              "simple_kpc: failed to load kperf function "
              "%s\n",
              symbol->name);
      exit(1);
    }
    *symbol->impl = p;
  }

  for (usize i = 0; i < ARRAY_LENGTH(KPERFDATA_SYMBOLS); i++) {
    const symbol *symbol = &KPERFDATA_SYMBOLS[i];
    void *p = dlsym(kperfdata, symbol->name);
    if (!p) {
      fprintf(stderr,
              "simple_kpc: failed to load kperfdata function "
              "%s\n",
              symbol->name);
      exit(1);
    }
    *symbol->impl = p;
  }

  if (kpc_force_all_ctrs_get(NULL) != 0) {
    fprintf(stderr,
            "simple_kpc: permission denied, xnu/kpc "
            "requires root privileges\n");
    exit(1);
  }

  initialized = true;
}

struct sk_events {
  const char **human_readable_names;
  const char **internal_names;
  usize count;
};

sk_events *sk_events_create(void) {
  sk_events *e = calloc(1, sizeof(sk_events));
  *e = (sk_events){
    .human_readable_names = calloc(KPC_MAX_COUNTERS, sizeof(const char *)),
    .internal_names = calloc(KPC_MAX_COUNTERS, sizeof(const char *)),
    .count = 0,
  };
  return e;
}

void sk_events_push(sk_events *e, const char *human_readable_name, const char *internal_name) {
  if (e->count >= KPC_MAX_COUNTERS) {
    fprintf(stderr, "Cannot push event anymore");
    abort();
  }
  e->human_readable_names[e->count] = human_readable_name;
  e->internal_names[e->count] = internal_name;
  e->count++;
}

void sk_events_destroy(sk_events *e) {
  free(e->human_readable_names);
  free(e->internal_names);
  free(e);
}

struct sk_in_progress_measurement {
  sk_events *events;
  u32 classes;
  usize counter_map[KPC_MAX_COUNTERS];
  u64 regs[KPC_MAX_COUNTERS];
  u64 counters[KPC_MAX_COUNTERS];
};

sk_in_progress_measurement *sk_start_measurement(sk_events *e) {
  assert(initialized);

  sk_in_progress_measurement *m = calloc(1, sizeof(sk_in_progress_measurement));
  *m = (sk_in_progress_measurement){
    .events = e,
    .classes = 0,
    .counter_map = { 0 },
    .counters = { 0 },
  };

  kpep_db *kpep_db = NULL;
  kpep_db_create(NULL, &kpep_db);

  kpep_config *kpep_config = NULL;
  kpep_config_create(kpep_db, &kpep_config);
  kpep_config_force_counters(kpep_config);

  for (usize i = 0; i < m->events->count; i++) {
    const char *internal_name = m->events->internal_names[i];
    kpep_event *event = NULL;
    kpep_db_event(kpep_db, internal_name, &event);

    if (event == NULL) {
      const char *human_readable_name = m->events->human_readable_names[i];
      fprintf(stderr, "Cannot find event for %s: “%s”.\n", human_readable_name, internal_name);
      exit(1);
    }

    kpep_config_add_event(kpep_config, &event, 0, NULL);
  }

  kpep_config_kpc_classes(kpep_config, &m->classes);
  kpep_config_kpc_map(kpep_config, m->counter_map, sizeof(m->counter_map));
  kpep_config_kpc(kpep_config, m->regs, sizeof(m->regs));

  kpep_config_free(kpep_config);
  kpep_db_free(kpep_db);

  kpc_force_all_ctrs_set(1);
  kpc_set_config(m->classes, m->regs);

  // Don’t put any library code below these kpc calls!
  kpc_set_counting(m->classes);
  kpc_set_thread_counting(m->classes);
  kpc_get_thread_counters(0, KPC_MAX_COUNTERS, m->counters);
  return m;
}

void sk_finish_measurement(sk_in_progress_measurement *m) {
  u64 counters_after[KPC_MAX_COUNTERS] = { 0 };

  // Don’t put any library code above these kpc calls!
  // We don’t want to execute anything until timing has stopped
  kpc_get_thread_counters(0, KPC_MAX_COUNTERS, counters_after);
  kpc_set_counting(0);
  kpc_force_all_ctrs_set(0);

  fprintf(stderr, "\033[1m=== simple-kpc report ===\033[m\n\n");
  setlocale(LC_NUMERIC, "");
  for (usize i = 0; i < m->events->count; i++) {
    const char *name = m->events->human_readable_names[i];
    usize idx = m->counter_map[i];
    u64 diff = counters_after[idx] - m->counters[idx];
    fprintf(stderr, "\033[32m%16llu \033[95m%s\033[m\n", diff, name);
  }

  free(m);
}
