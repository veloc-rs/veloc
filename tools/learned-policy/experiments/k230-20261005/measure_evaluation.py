#!/usr/bin/env python3
"""Pilot evaluation and ablations. Split/provenance determine independence."""
from pathlib import Path
import argparse
import json
import os
import re
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--skip-coremark", action="store_true")
    p.add_argument("--mode", default="learned", help="kernel candidate executable name")
    p.add_argument("--report", default="model-evaluation.json")
    args = p.parse_args()
    root = args.root.resolve()
    os.sched_setaffinity(0, {0})
    cases = json.loads((root / "corpus-times.json").read_text())
    results = []
    for case in cases:
        modes = ("baseline", args.mode)
        samples = {v: [] for v in modes}
        checks = {}
        for iteration in range(6):
            for mode in (modes if iteration % 2 == 0 else modes[::-1]):
                seed = (17, 9127)[iteration % 2]
                r = subprocess.run(
                    [
                        str(root / "bin" / case["name"] / mode),
                        str(case["repeats"]),
                        str(seed),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                elapsed, checksum = r.stdout.strip().split()
                if seed in checks:
                    assert checks[seed] == checksum, (case["name"], mode, seed)
                checks[seed] = checksum
                if iteration:
                    samples[mode].append(float(elapsed))
        row = dict(
            name=case["name"],
            split=case["split"],
            samples=samples,
            median={k: statistics.median(v) for k, v in samples.items()},
        )
        results.append(row)
        (root / args.report).write_text(json.dumps(results, indent=2) + "\n")
    print("corpus evaluation complete", flush=True)
    if args.skip_coremark:
        return
    cm = root / "coremark"
    runs = []
    modes = (
        "baseline",
        "learned",
        "learned-inline",
        "learned-schedule",
        "inline-expand",
        "schedule-pressure",
        "schedule-latency",
        "schedule-source",
        "schedule-fanout",
    )
    for iteration in range(3):
        for mode in (modes if iteration % 2 == 0 else modes[::-1]):
            r = subprocess.run(
                [
                    str(cm / ("coremark-" + mode)),
                    "0",
                    "0",
                    "0x66",
                    "10000",
                    "7",
                    "1",
                    "2000",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            (cm / (mode + "-" + str(iteration + 1) + ".log")).write_text(r.stdout)
            assert all(
                re.search(p, r.stdout)
                for p in (
                    r"seedcrc\s*:\s*0xe9f5",
                    r"crclist\s*:\s*0xe714",
                    r"crcmatrix\s*:\s*0x1fd7",
                    r"crcstate\s*:\s*0x8e3a",
                )
            ), r.stdout
            assert not [
                l
                for l in r.stdout.splitlines()
                if "ERROR!" in l and "Must execute for at least 10 secs" not in l
            ], r.stdout
            score = float(re.search(r"Iterations/Sec\s*:\s*([\d.]+)", r.stdout)[1])
            runs.append(dict(mode=mode, iteration=iteration + 1, score=score))
            (root / "coremark-pilot.json").write_text(json.dumps(runs, indent=2) + "\n")
            print("pilot", mode, iteration + 1, score, flush=True)
    print(
        "CoreMark pilot uses short runs; final acceptance requires >=10-second runs.",
        flush=True,
    )


if __name__ == "__main__":
    main()
