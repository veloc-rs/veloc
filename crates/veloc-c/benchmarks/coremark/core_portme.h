/* Linux LP64D adapter declarations. Benchmark algorithm sources are unchanged. */
#ifndef VELOC_CORE_PORTME_H
#define VELOC_CORE_PORTME_H
#define HAS_FLOAT 1
#define HAS_TIME_H 0
#define HAS_STDIO 0
#define HAS_PRINTF 0
#define MULTITHREAD 1
#define MAIN_HAS_NOARGC 0
#define MAIN_HAS_NORETURN 0
#define SEED_METHOD SEED_ARG
#define MEM_METHOD MEM_MALLOC
#define MEM_LOCATION "heap"
#define NULL ((void *)0)
#ifndef COMPILER_VERSION
#define COMPILER_VERSION "unspecified"
#endif
#ifndef COMPILER_FLAGS
#define COMPILER_FLAGS "unspecified"
#endif
typedef signed short ee_s16;
typedef unsigned short ee_u16;
typedef signed int ee_s32;
typedef unsigned int ee_u32;
typedef unsigned char ee_u8;
typedef double ee_f32;
typedef unsigned long ee_ptr_int;
typedef unsigned long ee_size_t;
typedef unsigned long CORE_TICKS;
#define align_mem(x) (void *)(4 + (((ee_ptr_int)(x) - 1) & ~3UL))
typedef struct CORE_PORTABLE_S { ee_u8 portable_id; } core_portable;
extern ee_u32 default_num_contexts;
int ee_printf(const char *format, ...);
void portable_init(core_portable *p, int *argc, char *argv[]);
void portable_fini(core_portable *p);
#endif
