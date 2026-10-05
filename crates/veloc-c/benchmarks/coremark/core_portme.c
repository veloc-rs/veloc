/* Shared OS adapter, compiled once with Clang and linked into both executables. */
#include "coremark.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

ee_u32 default_num_contexts = 1;
static struct timespec start, end;
int ee_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);
    int result = vprintf(format, args);
    va_end(args);
    return result;
}
void *portable_malloc(ee_size_t size) { return malloc(size); }
void portable_free(void *p) { free(p); }
void portable_init(core_portable *p, int *argc, char *argv[]) {
    (void)argc; (void)argv;
    p->portable_id = 1;
}
void portable_fini(core_portable *p) { p->portable_id = 0; }
void start_time(void) { clock_gettime(CLOCK_MONOTONIC, &start); }
void stop_time(void) { clock_gettime(CLOCK_MONOTONIC, &end); }
CORE_TICKS get_time(void) {
    return (CORE_TICKS)(end.tv_sec - start.tv_sec) * 1000000000UL
        + (CORE_TICKS)(end.tv_nsec - start.tv_nsec);
}
secs_ret time_in_secs(CORE_TICKS ticks) { return (double)ticks / 1000000000.0; }
