#define _GNU_SOURCE
#include <errno.h>
#include <setjmp.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <ucontext.h>
#include <unistd.h>

struct memory_range {
    uintptr_t start, accessible_end, reservation_end;
};
struct code_range {
    uintptr_t start, end;
};
struct scope {
    sigjmp_buf jump;
    struct scope *previous;
    const struct memory_range *memories;
    size_t memory_count;
    const struct code_range *code;
    size_t code_count;
    int memory_trap;
};
static _Thread_local struct scope *active;
static struct sigaction previous_segv, previous_bus;

/* No allocation, locks, Rust unwinding or host callbacks on this path. */
void veloc_native_handle_signal(int signal, siginfo_t *info, void *context) {
    (void)signal;
    struct scope *scope = active;
    if (!scope || info->si_code <= 0)
        return;
    ucontext_t *uc = context;
#if defined(__x86_64__)
    uintptr_t pc = uc->uc_mcontext.gregs[REG_RIP];
#elif defined(__riscv) && __riscv_xlen == 64
    uintptr_t pc = uc->uc_mcontext.__gregs[REG_PC];
#else
#error unsupported native trap context
#endif
    int native_pc = 0;
    for (size_t i = 0; i < scope->code_count; ++i)
        if (pc >= scope->code[i].start && pc < scope->code[i].end)
            native_pc = 1;
    if (!native_pc)
        return;
    uintptr_t fault = (uintptr_t)info->si_addr;
    for (size_t i = 0; i < scope->memory_count; ++i) {
        const struct memory_range *range = &scope->memories[i];
        if (fault >= range->start && fault >= range->accessible_end &&
            fault < range->reservation_end)
            siglongjmp(scope->jump, scope->memory_trap);
    }
}

static void handle_signal(int signal, siginfo_t *info, void *context) {
    veloc_native_handle_signal(signal, info, context);
    const struct sigaction *old = signal == SIGSEGV ? &previous_segv : &previous_bus;
    if (old->sa_handler == SIG_IGN && info->si_code <= 0)
        return;
    if (old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN) {
        if (old->sa_flags & SA_SIGINFO)
            old->sa_sigaction(signal, info, context);
        else
            old->sa_handler(signal);
        return;
    }
    sigaction(signal, old, NULL);
    raise(signal);
    _exit(128 + signal);
}

/* Serialized by the Rust installer. Publish old handlers before installing. */
int veloc_native_install(void) {
    struct sigaction action = {0};
    action.sa_sigaction = handle_signal;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGSEGV, NULL, &previous_segv) || sigaction(SIGBUS, NULL, &previous_bus))
        return errno;
    if (sigaction(SIGSEGV, &action, NULL))
        return errno;
    if (sigaction(SIGBUS, &action, NULL)) {
        int error = errno;
        sigaction(SIGSEGV, &previous_segv, NULL);
        return error;
    }
    return 0;
}

/* setjmp and its resumed continuation are in C, never in a Rust function.
   The actual guest call is direct: a fault crosses only generated frames.
   All Rust-owned arguments and diagnostics live above this boundary. */
int veloc_native_call(const void *entry, void *vmctx, const int64_t *args, int64_t *results,
                      int64_t *return_bits, int initialize, const struct memory_range *memories,
                      size_t memory_count, const struct code_range *code, size_t code_count,
                      int memory_trap) {
    struct scope scope = {.previous = active,
                          .memories = memories,
                          .memory_count = memory_count,
                          .code = code,
                          .code_count = code_count,
                          .memory_trap = memory_trap};
    active = &scope;
    int trap = sigsetjmp(scope.jump, 1);
    if (!trap) {
        if (initialize)
            ((void (*)(void *))entry)(vmctx);
        else
            *return_bits =
                ((int64_t (*)(void *, const int64_t *, int64_t *))entry)(vmctx, args, results);
    }
    active = scope.previous;
    return trap;
}

void veloc_native_raise(int code) {
    if (active)
        siglongjmp(active->jump, code);
}
void *veloc_native_suspend(void) {
    struct scope *old = active;
    active = NULL;
    return old;
}
void veloc_native_restore(void *scope) { active = scope; }
