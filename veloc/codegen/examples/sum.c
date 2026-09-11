#include <assert.h>
#include <stdint.h>
#include <stdio.h>

extern uint64_t sum(uint64_t n);
int main(void) {
    uint64_t result = sum(100);
    assert(result == 5050);
    printf("sum(100) = %llu\n", (unsigned long long)result);
    return 0;
}
