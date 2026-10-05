#!/usr/bin/env python3
"""Long CoreMark runs and native target compilation latency, without tracing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

UNITS = ("main", "list_join", "matrix", "state", "util")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--modes", nargs="+", default=["baseline", "learned"])
    parser.add_argument("--runtime-rounds", type=int, default=7)
    parser.add_argument("--compile-rounds", type=int, default=21)
    parser.add_argument("--policy-extension", default="json")
    parser.add_argument("--report", default="final-measurement.json")
    args = parser.parse_args()
    root = args.root.resolve()
    os.sched_setaffinity(0, {0})
    report = dict(
        board=platform.uname()._asdict(),
        cpuinfo=Path("/proc/cpuinfo").read_text(),
        load_before=os.getloadavg(),
        affinity=[0],
        runtime=[],
        compilation=[],
        compiler_sha256=digest(root / "veloc-c"),
        source_sha256={u: digest(root / "coremark/input" / (u + ".i")) for u in UNITS},
        executable_sha256={
            m: digest(root / "coremark" / ("coremark-" + m)) for m in args.modes
        },
        policy_sha256={},
    )

    def save():
        (root / args.report).write_text(json.dumps(report, indent=2) + "\n")

    for iteration in (range(-1, args.runtime_rounds) if args.runtime_rounds else []):
        order = args.modes if iteration % 2 == 0 else args.modes[::-1]
        for mode in order:
            iterations = 10000 if iteration < 0 else 100000
            command = [
                str(root / "coremark" / ("coremark-" + mode)),
                "0",
                "0",
                "0x66",
                str(iterations),
                "7",
                "1",
                "2000",
            ]
            output = subprocess.run(
                command, check=True, text=True, capture_output=True
            ).stdout
            (
                root / "coremark" / ("final-%s-%d.log" % (mode, iteration + 1))
            ).write_text(output)
            patterns = (
                r"seedcrc\s*:\s*0xe9f5",
                r"crclist\s*:\s*0xe714",
                r"crcmatrix\s*:\s*0x1fd7",
                r"crcstate\s*:\s*0x8e3a",
            )
            assert all(re.search(p, output) for p in patterns), output
            if iteration >= 0:
                assert (
                    "Correct operation validated" in output and "ERROR!" not in output
                ), output
                score = float(re.search(r"Iterations/Sec\s*:\s*([\d.]+)", output)[1])
                assert iterations / score >= 10
                report["runtime"].append(
                    dict(mode=mode, round=iteration + 1, score=score)
                )
                save()
                print("runtime", mode, iteration + 1, score, flush=True)

    # Warm process startup, file caches and compiler allocation paths equally.
    for iteration in range(-3, args.compile_rounds):
        order = args.modes if iteration % 2 == 0 else args.modes[::-1]
        for mode in order:
            directory = root / "native-build" / mode
            directory.mkdir(parents=True, exist_ok=True)
            policy = (
                None
                if mode == "baseline"
                else root / "policies" / (mode + "." + args.policy_extension)
            )
            if policy:
                report["policy_sha256"][mode] = digest(policy)
            units = []
            for unit in UNITS:
                obj = directory / (unit + ".o")
                command = [
                    str(root / "veloc-c"),
                    str(root / "coremark/input" / (unit + ".i")),
                    "-O1",
                    "--cpu",
                    "c908",
                    "-o",
                    str(obj),
                ]
                if policy:
                    command += ["--policy", str(policy)]
                start = time.perf_counter()
                with (directory / (unit + ".log")).open("w") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=log)
                    _, status, usage = os.wait4(child.pid, 0)
                    child.returncode = os.waitstatus_to_exitcode(status)
                elapsed = time.perf_counter() - start
                assert child.returncode == 0, (mode, unit, child.returncode)
                units.append(
                    dict(
                        unit=unit,
                        wall=elapsed,
                        cpu=usage.ru_utime + usage.ru_stime,
                        rss_kib=usage.ru_maxrss,
                        object_sha256=digest(obj),
                    )
                )
                expected = root / "coremark" / mode / (unit + ".o")
                if expected.exists():
                    assert digest(obj) == digest(expected), (
                        "host/board object mismatch",
                        mode,
                        unit,
                    )
            if iteration >= 0:
                report["compilation"].append(
                    dict(
                        mode=mode,
                        round=iteration + 1,
                        units=units,
                        wall=sum(u["wall"] for u in units),
                        cpu=sum(u["cpu"] for u in units),
                    )
                )
                save()
                print(
                    "compile",
                    mode,
                    iteration + 1,
                    report["compilation"][-1]["wall"],
                    flush=True,
                )
    report["runtime_median"] = {
        m: statistics.median(s["score"] for s in report["runtime"] if s["mode"] == m)
        for m in args.modes
        if report["runtime"]
    }
    report["compile_median"] = {
        m: statistics.median(s["wall"] for s in report["compilation"] if s["mode"] == m)
        for m in args.modes
    }
    report["load_after"] = os.getloadavg()
    save()
    print(report["runtime_median"], report["compile_median"], flush=True)


if __name__ == "__main__":
    main()
