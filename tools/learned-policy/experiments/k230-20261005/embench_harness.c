/* Linux timing adapter. Compile separately, identically for every compiler. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void initialise_benchmark(void);
extern void warm_caches(int);
extern int benchmark(void);
extern int verify_benchmark(int);

static double seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int repeats = argc > 1 ? atoi(argv[1]) : 1;
    if (repeats < 1) return 2;
    initialise_benchmark();
#ifndef BEEBS
    warm_caches(1);
#endif
    uint32_t checksum = 0;
    int result = 0;
    double start = seconds();
    for (int i = 0; i < repeats; ++i) {
#ifdef BEEBS
        /* Several BEEBS kernels transform their input in place. */
        initialise_benchmark();
#endif
        result = benchmark();
        checksum = checksum * 33u + (uint32_t)result;
    }
    double elapsed = seconds() - start;
    int valid = verify_benchmark(result);
    printf("%.9f %u %d\n", elapsed, checksum, valid);
    return valid == 1 ? 0 : 1;
}
