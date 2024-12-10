typedef struct sk_events sk_events;
typedef struct sk_in_progress_measurement sk_in_progress_measurement;

void sk_init(void);

sk_events *sk_events_create(void);
void sk_events_push(sk_events *e, const char *human_readable_name, const char *internal_name);
void sk_events_destroy(sk_events *e);

sk_in_progress_measurement *sk_start_measurement(sk_events *e);
void sk_finish_measurement(sk_in_progress_measurement *m);
