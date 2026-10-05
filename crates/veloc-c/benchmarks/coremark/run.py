#!/usr/bin/env python3
"""Reproducible native CoreMark comparison; never commits or pushes source.

Compilation timing covers all five unchanged translation units, excludes the
shared OS adapter and linker, and includes preprocessing and process startup.
Optional preprocessed timing isolates C-to-object compilation on identical input.
Warmups precede interleaved serial samples; profiling is disabled while timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
PORT = Path(__file__).resolve().parent
REVISION = "1f483d5b8316753a742cbf5590caf5bd0a4e4777"
UNITS = ("main", "list_join", "matrix", "state", "util")


def run(command, **kwargs):
    return subprocess.run(list(map(str, command)), check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "target/veloc-c-coremark/coremark")
    parser.add_argument("--veloc", type=Path, default=ROOT / "target/release/veloc-c")
    parser.add_argument("--clang", type=Path, default=Path("/opt/homebrew/opt/llvm/bin/clang"))
    parser.add_argument("--sysroot", type=Path, default=ROOT / "target/riscv64-sysroot")
    parser.add_argument("--out", type=Path, default=ROOT / "target/native-coremark")
    parser.add_argument("--host", default="root@192.168.2.19")
    parser.add_argument("--remote-dir", default="/root/veloc-riscv/native-coremark-20261004")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--compile-runs", type=int, default=7)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--preprocessed", action="store_true")
    parser.add_argument("--screen", action="store_true", help="Allow sub-10-second exploratory samples")
    args = parser.parse_args()
    if min(args.iterations, args.runs, args.compile_runs) < 1:
        parser.error("iteration and sample counts must be positive")
    for name in ("source", "veloc", "clang", "sysroot", "out"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.source.exists():
        run(["git", "clone", "https://github.com/eembc/coremark.git", args.source])
        run(["git", "-C", args.source, "checkout", REVISION])
    revision = run(["git", "-C", args.source, "rev-parse", "HEAD"]).stdout.strip()
    if revision != REVISION or run(["git", "-C", args.source, "status", "--porcelain", "--untracked-files=no"]).stdout:
        raise RuntimeError("CoreMark must be an unmodified checkout of " + REVISION)
    args.out.mkdir(parents=True, exist_ok=True)
    common = ["--target=riscv64-linux-gnu", "--sysroot=" + str(args.sysroot),
              "-march=rv64gc_zba_zbb", "-mabi=lp64d"]
    defines = ["-DPERFORMANCE_RUN=1", "-DITERATIONS=" + str(args.iterations),
               '-DCOMPILER_VERSION="measured"', '-DCOMPILER_FLAGS="see report.json"']
    includes = ["-I", PORT, "-I", args.source]
    commands = {}
    for compiler in ("veloc", "llvm"):
        (args.out / compiler).mkdir(exist_ok=True)
        commands[compiler] = []
        for unit in UNITS:
            source = args.source / ("core_" + unit + ".c")
            if args.preprocessed:
                dest = args.out / (unit + ".i")
                if compiler == "veloc":
                    dest.write_text(run([args.clang, *common, "-E", "-P", *includes, *defines, source]).stdout)
                source = dest
            obj = args.out / compiler / (unit + ".o")
            preprocessing = [] if args.preprocessed else [*includes, *defines]
            if compiler == "veloc":
                command = [args.veloc, source, "-O1", "--cpu", "c908", "--cpp", args.clang,
                           "--sysroot", args.sysroot, *preprocessing, "-o", obj]
            else:
                command = [args.clang, *common, "-O3", *preprocessing, "-c", source, "-o", obj]
            commands[compiler].append(command)
    report = {
        "coremark_revision": revision,
        "llvm_version": run([args.clang, "--version"]).stdout,
        "compiler_sha256": digest(args.veloc),
        "veloc_commit": run(["git", "-C", ROOT, "rev-parse", "HEAD"]).stdout.strip(),
        "working_tree": run(["git", "-C", ROOT, "status", "--short"]).stdout,
        "preprocessed": args.preprocessed,
        "commands": {k: [[str(v) for v in cmd] for cmd in cmds] for k, cmds in commands.items()},
        "compile_seconds": {k: [] for k in commands}, "runs": [],
        "port_sha256": {p.name: digest(p) for p in (PORT / "core_portme.c", PORT / "core_portme.h")},
    }
    def save():
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Warm both compilers; report every measured sample, including outliers.
    for iteration in range(args.compile_runs + 1):
        for compiler in (("veloc", "llvm") if iteration % 2 == 0 else ("llvm", "veloc")):
            start = time.perf_counter()
            for command in commands[compiler]:
                run(command)
            duration = time.perf_counter() - start
            if iteration:
                report["compile_seconds"][compiler].append(duration)
    report["compile_median_seconds"] = {k: statistics.median(v) for k, v in report["compile_seconds"].items()}
    port = args.out / "port.o"
    run([args.clang, *common, "-O2", *includes, *defines, "-c", PORT / "core_portme.c", "-o", port])
    for compiler in commands:
        run([args.clang, *common, "-fuse-ld=lld", "-no-pie",
             *[args.out / compiler / (u + ".o") for u in UNITS], port,
             "-o", args.out / ("coremark-" + compiler)])
    report["binary_sha256"] = {k: digest(args.out / ("coremark-" + k)) for k in commands}
    save()
    print("compile medians:", report["compile_median_seconds"], flush=True)
    if args.compile_only:
        return
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host]
    report["board"] = run([*ssh, "uname -a; cat /proc/cpuinfo"]).stdout
    run([*ssh, "mkdir -p " + shlex.quote(args.remote_dir)])
    for compiler in commands:
        binary = args.out / ("coremark-" + compiler)
        run(["scp", binary, args.host + ":" + args.remote_dir + "/"])
        actual = run([*ssh, "sha256sum " + shlex.quote(args.remote_dir + "/" + binary.name)]).stdout.split()[0]
        if actual != report["binary_sha256"][compiler]:
            raise RuntimeError("uploaded binary hash differs")
    # Transfers are finished before timing on the board's one online Linux hart.
    for iteration in range(args.runs):
        for compiler in (("veloc", "llvm") if iteration % 2 == 0 else ("llvm", "veloc")):
            command = shlex.join(["taskset", "-c", "0", args.remote_dir + "/coremark-" + compiler,
                                  "0", "0", "0x66", str(args.iterations), "7", "1", "2000"])
            output = run([*ssh, command]).stdout
            (args.out / f"{compiler}-{iteration + 1}.log").write_text(output)
            score = float(re.search(r"Iterations/Sec\s*:\s*([\d.]+)", output)[1])
            seconds = float(re.search(r"Total time \(secs\)\s*:\s*([\d.]+)", output)[1])
            valid = "Correct operation validated" in output
            # CoreMark labels a correct short development run as invalid solely
            # for duration. In screen mode require all published CRCs explicitly.
            crc_ok = all(re.search(pattern, output) for pattern in (
                r"seedcrc\s*:\s*0xe9f5", r"crclist\s*:\s*0xe714",
                r"crcmatrix\s*:\s*0x1fd7", r"crcstate\s*:\s*0x8e3a"))
            unexpected_errors = [line for line in output.splitlines() if "ERROR!" in line and "Must execute for at least 10 secs" not in line]
            if not crc_ok or unexpected_errors or not (valid or (args.screen and seconds < 10)):
                raise RuntimeError("CoreMark validation failed: " + output)
            if seconds < 10 and not args.screen:
                raise RuntimeError("Formal samples must run at least ten seconds")
            crc_final = re.search(r"crcfinal\s*:\s*(0x[0-9a-fA-F]+)", output)[1].lower()
            if any(sample["crc_final"] != crc_final for sample in report["runs"]):
                raise RuntimeError("Final CRC differs between compilers or repeated runs")
            sample = dict(compiler=compiler, score=score, seconds=seconds,
                          iteration=iteration + 1, crc_final=crc_final)
            report["runs"].append(sample)
            save()
            print(sample, flush=True)
    report["screen"] = args.screen
    report["median"] = {k: statistics.median(r["score"] for r in report["runs"] if r["compiler"] == k) for k in commands}
    report["veloc_over_llvm"] = report["median"]["veloc"] / report["median"]["llvm"]
    save()
    print("runtime ratio:", report["veloc_over_llvm"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Command failed: {shlex.join(list(map(str, error.cmd)))}\n{error.stderr}")
