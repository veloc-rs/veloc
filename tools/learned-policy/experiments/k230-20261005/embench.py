#!/usr/bin/env python3
"""Build pinned, unmodified Embench programs with a common Linux adapter.

The small header facade exposes the LP64 libc declarations the programs use;
both compilers receive exactly the same preprocessed input. No LTO is used.
This is a compiler experiment, not an official Embench score submission.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REVISION = "09c2ed8c3b7008c95d08b038de4a3f6dc103ed70"
# Fixed before looking at performance. Whole programs stay in one partition.
PARTITIONS = {
    "train": "aha-mont64 crc32 edn huffbench matmult-int md5sum nettle-aes statemate ud",
    "validation": "nettle-sha256 slre tarfind",
    "test": "depthconv nsichneu picojpeg qrduino sglib-combined wikisort xgboost",
}
BEEBS_REVISION = "049ded9f3aeb5591f553879d3a0376b8614e9422"
BEEBS_PARTITIONS = {
    "train": "compress dijkstra fdct fir jfdctint levenshtein ludcmp mergesort minver nettle-arcfour nettle-des rijndael",
    "validation": "cubic expint matmult-float nettle-cast128 qurt st",
    "test": "bubblesort cnt duff fasta insertsort ns prime qsort select stb_perlin stringsearch1 strstr tarai",
}
HEADERS = {
    "stddef.h": """typedef unsigned long size_t;
typedef long ptrdiff_t;
#define NULL ((void *)0)
#define offsetof(t, m) __builtin_offsetof(t, m)
""",
    "stdint.h": """typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;
typedef long intptr_t;
typedef unsigned long uintptr_t;
#define UINT32_C(x) x##U
#define UINT64_C(x) x##UL
#define INT32_MAX 2147483647
#define UINT32_MAX 4294967295U
""",
    "stdbool.h": "#define bool _Bool\n#define true 1\n#define false 0\n",
    "stdlib.h": """#include <stddef.h>
void *malloc(size_t size); void *calloc(size_t n, size_t size); void *realloc(void *p, size_t size);
void free(void *p); void abort(void); void exit(int status); int abs(int x);
int rand(void); void srand(unsigned int seed); int atoi(const char *s);
void qsort(void *p, size_t n, size_t size, int (*cmp)(const void *a, const void *b));
#define RAND_MAX 2147483647
""",
    "string.h": """#include <stddef.h>
void *memcpy(void *d, const void *s, size_t n); void *memmove(void *d, const void *s, size_t n);
void *memset(void *d, int x, size_t n); int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s); int strcmp(const char *a, const char *b);
int strncmp(const char *a, const char *b, size_t n); char *strcpy(char *d, const char *s);
char *strncpy(char *d, const char *s, size_t n); char *strchr(const char *s, int x);
""",
    "stdio.h": """int printf(const char *fmt, ...); int puts(const char *s);
int sprintf(char *d, const char *fmt, ...); int putchar(int x);
""",
    "math.h": """double fabs(double); float fabsf(float); double sqrt(double);
double floor(double); double ceil(double); double pow(double, double);
double log(double); double exp(double); double sin(double); double cos(double);
float frexpf(float x, int *exp);
""",
    "ctype.h": "int isspace(int); int isdigit(int); int isalpha(int); int tolower(int);\n",
    "assert.h": "void abort(void);\n#define assert(x) ((x) ? (void)0 : abort())\n",
    "limits.h": """#define CHAR_BIT 8
#define INT_MAX 2147483647
#define INT_MIN (-INT_MAX - 1)
#define UINT_MAX 4294967295U
#define LONG_MAX 9223372036854775807L
#define ULONG_MAX 18446744073709551615UL
""",
    "stdarg.h": "/* No variadic definitions in the supported benchmark subset. */\n",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log=None):
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if log:
        log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-2500:])


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--suite", choices=["embench", "beebs"], default="embench")
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--compiler", type=Path, default=ROOT / "target/release/veloc-c"
    )
    parser.add_argument("--clang", default="/opt/homebrew/opt/llvm/bin/clang")
    parser.add_argument("--sysroot", type=Path, default=ROOT / "target/riscv64-sysroot")
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).with_name("models") / "c908.vlp"
    )
    parser.add_argument("--mode", default="learned")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--probes", action="store_true")
    parser.add_argument(
        "--probe-limit",
        type=int,
        default=64,
        help="contexts per program, chosen before timing",
    )
    return parser.parse_args()


def main():
    args = arguments()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    beebs = args.suite == "beebs"
    source = (
        args.source or ROOT / ("target/beebs" if beebs else "target/embench-iot")
    ).resolve()
    revision = BEEBS_REVISION if beebs else REVISION
    repository = (
        "https://github.com/mageec/beebs"
        if beebs
        else "https://github.com/embench/embench-iot"
    )
    partitions = BEEBS_PARTITIONS if beebs else PARTITIONS
    if not source.exists():
        run(["git", "clone", repository + ".git", str(source)])
        run(["git", "-C", str(source), "checkout", revision])
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision or subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"]
    ):
        raise RuntimeError(
            "Benchmark input must be the clean pinned revision " + revision
        )
    common = [
        "--target=riscv64-linux-gnu",
        "--sysroot=" + str(args.sysroot),
        "-march=rv64gc_zba_zbb",
        "-mabi=lp64d",
    ]
    link = [args.clang, *common, "-fuse-ld=lld", "-no-pie"]

    def compile_unit(case, unit, mode, policy=None):
        directory = root / "build" / case / mode
        directory.mkdir(parents=True, exist_ok=True)
        obj = directory / (unit + ".o")
        inp = root / "input" / case / (unit + ".i")
        if mode == "llvm":
            command = [
                args.clang,
                *common,
                "-O3",
                "-ffp-contract=off",
                "-c",
                str(inp),
                "-o",
                str(obj),
            ]
        else:
            command = [
                str(args.compiler),
                str(inp),
                "-O1",
                "--cpu",
                "c908",
                "--verify-ir",
                "--policy-trace",
                str(directory / (unit + ".jsonl")),
                "-o",
                str(obj),
            ]
            if policy:
                command += ["--policy", str(policy)]
        run(command, directory / (unit + ".log"))
        return obj

    def link_case(name, mode, objects):
        exe = root / "bin" / name / mode
        exe.parent.mkdir(parents=True, exist_ok=True)
        run([*link, *map(str, objects), str(root / "harness.o"), "-lm", "-o", str(exe)])
        return exe

    if args.probes:
        manifest = json.loads((root / "manifest.json").read_text())
        verified = {
            row["name"]
            for row in json.loads((root / "timings.json").read_text())["cases"]
            if "error" not in row
        }
        tasks = []
        for case in manifest["cases"]:
            if (
                case.get("error")
                or case["split"] == "test"
                or case["name"] not in verified
            ):
                continue
            contexts = []
            for unit in case["units"]:
                trace = root / "build" / case["name"] / "baseline" / (unit + ".jsonl")
                seen = set()
                for line in trace.read_text().splitlines()[1:]:
                    row = json.loads(line)
                    key = (row["decision"], tuple(row["features"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    if row["decision"] == "schedule" and (
                        row["features"][0] < 8 or row["features"][15] == 0
                    ):
                        continue
                    contexts.append(dict(row, unit=unit))
            # Stable hash sample, independent of timing or benefit.
            contexts.sort(
                key=lambda row: hashlib.sha256(
                    json.dumps(row, sort_keys=True).encode()
                ).digest()
            )
            for row in contexts[: args.probe_limit]:
                for action in range(1, 3 if row["decision"] == "inline" else 5):
                    tasks.append(
                        dict(
                            row,
                            action=action,
                            name=case["name"],
                            split=case["split"],
                            units=case["units"],
                        )
                    )
        out = root / "probes"
        out.mkdir(exist_ok=True)

        def probe(item):
            index, row = item
            tag = "%04d" % index
            directory = out / tag
            directory.mkdir(exist_ok=True)
            policy = directory / "policy.json"
            policy.write_text(
                json.dumps(
                    dict(
                        version=1,
                        target="riscv64/c908",
                        **{
                            row["decision"]: dict(
                                kind="probe",
                                cases=[
                                    dict(features=row["features"], action=row["action"])
                                ],
                            )
                        }
                    )
                )
            )
            obj = compile_unit(row["name"], row["unit"], "probe-" + tag, policy)
            baseline = root / "build" / row["name"] / "baseline"
            changed = obj.read_bytes() != (baseline / obj.name).read_bytes()
            sha = None
            if changed:
                objects = [
                    obj if u == row["unit"] else baseline / (u + ".o")
                    for u in row["units"]
                ]
                exe = link_case(row["name"], "probe-" + tag, objects)
                shutil.copy2(exe, directory / "run")
                sha = digest(exe)
            return dict(row, id=tag, changed=changed, sha256=sha)

        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(probe, enumerate(tasks)))
        canonical = {}
        for row in rows:
            if row["changed"]:
                row["measurement"] = canonical.setdefault(
                    (row["name"], row["sha256"]), row["id"]
                )
        (out / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            len(rows), "interventions,", len(canonical), "distinct changed executables"
        )
    elif args.evaluate:
        manifest = json.loads((root / "manifest.json").read_text())
        for case in manifest["cases"]:
            if case.get("error"):
                continue
            objects = [
                compile_unit(case["name"], u, args.mode, args.model)
                for u in case["units"]
            ]
            link_case(case["name"], args.mode, objects)
            print(case["name"], args.mode, flush=True)
        manifest["models"][args.mode] = digest(args.model)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        headers = root / "include"
        headers.mkdir(exist_ok=True)
        for name, text in HEADERS.items():
            (headers / name).write_text("#pragma once\n" + text)
        run(
            [
                args.clang,
                *common,
                *(["-DBEEBS"] if beebs else []),
                "-O2",
                "-c",
                str(Path(__file__).with_name("embench_harness.c")),
                "-o",
                str(root / "harness.o"),
            ]
        )
        manifest = dict(
            repository=repository,
            revision=revision,
            compiler=digest(args.compiler),
            clang=subprocess.check_output([args.clang, "--version"], text=True),
            models={"learned": digest(args.model)},
            cases=[],
        )
        for split, names in partitions.items():
            for name in names.split():
                row = dict(
                    name=name, split=split, units=[], source_sha256={}, input_sha256={}
                )
                try:
                    directory = root / "input" / name
                    directory.mkdir(parents=True, exist_ok=True)
                    sources = sorted((source / "src" / name).glob("*.c"))
                    if not beebs:
                        sources.append(source / "support/beebsc.c")
                    for path in sources:
                        unit = path.stem
                        row["units"].append(unit)
                        inp = directory / (unit + ".i")
                        definitions = (
                            ["-DMATMULT_FLOAT"]
                            if beebs and name == "matmult-float"
                            else []
                        )
                        run(
                            [
                                args.clang,
                                *common,
                                "-E",
                                "-P",
                                "-nostdinc",
                                "-I" + str(headers),
                                "-I" + str(source / "support"),
                                "-DGLOBAL_SCALE_FACTOR=1",
                                "-U__SIZEOF_INT128__",
                                "-D__attribute__(x)=",
                                "-D__attribute(x)=",
                                *definitions,
                                str(path),
                                "-o",
                                str(inp),
                            ]
                        )
                        row["source_sha256"][str(path.relative_to(source))] = digest(
                            path
                        )
                        row["input_sha256"][unit] = digest(inp)
                    for mode, policy in (
                        ("baseline", None),
                        ("learned", args.model),
                        ("llvm", None),
                    ):
                        objects = [
                            compile_unit(name, unit, mode, policy)
                            for unit in row["units"]
                        ]
                        link_case(name, mode, objects)
                except (RuntimeError, subprocess.TimeoutExpired) as e:
                    row["error"] = str(e)
                manifest["cases"].append(row)
                print(name, row.get("error", "built"), flush=True)
                (root / "manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["objects"] = {}
    for case in manifest["cases"]:
        if case.get("error"):
            continue
        modes = {}
        for directory in (root / "build" / case["name"]).iterdir():
            if directory.name.startswith("probe-"):
                continue
            objects = {u: directory / (u + ".o") for u in case["units"]}
            if all(path.exists() for path in objects.values()):
                modes[directory.name] = {u: digest(path) for u, path in objects.items()}
        manifest["objects"][case["name"]] = modes
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(root / "target.tar.gz", "w:gz") as archive:
        for directory in ("bin", "input", "probes"):
            if (root / directory).exists():
                for path in (root / directory).rglob("*"):
                    if directory == "bin" and path.name.startswith("probe-"):
                        continue
                    if path.is_file() and (
                        directory != "probes" or path.name in ("run", "manifest.json")
                    ):
                        archive.add(path, arcname=str(path.relative_to(root)))
        archive.add(root / "manifest.json", arcname="manifest.json")
        archive.add(
            Path(__file__).with_name("measure_embench.py"), arcname="measure_embench.py"
        )
        archive.add(
            Path(__file__).with_name("measure_compile.py"), arcname="measure_compile.py"
        )


if __name__ == "__main__":
    main()
