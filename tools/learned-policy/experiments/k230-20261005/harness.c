/* Independent timing/checksum harness, shared by every compiled policy. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern uint32_t kernel(uint32_t *, uint32_t, uint32_t);
uint32_t external_mix(uint32_t x) { return (x * 1664525u + 1013904223u) ^ (x >> 13); }
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
int main(int argc, char **argv) {
    unsigned repeats = argc > 1 ? strtoul(argv[1], 0, 10) : 100;
    unsigned seed = argc > 2 ? strtoul(argv[2], 0, 10) : 17;
    uint32_t a[256], x = seed;
    for (unsigned i = 0; i < 256; ++i) { x = x * 1664525u + 1013904223u; a[i] = x; }
    double start = now();
    for (unsigned r = 0; r < repeats; ++r) x ^= kernel(a, 256, x + r);
    double elapsed = now() - start;
    printf("%.9f %08x\n", elapsed, x);
    return 0;
}
