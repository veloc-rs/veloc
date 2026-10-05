#!/usr/bin/env python3
"""Paired native compilation of validated real programs, including model loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--mode", default="tuned")
    parser.add_argument("--verified", default="final.json")
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    root = args.root.resolve()
    os.sched_setaffinity(0, {0})
    manifest = json.loads((root / "manifest.json").read_text())
    verified = {
        row["name"]
        for row in json.loads((root / args.verified).read_text())["cases"]
        if "error" not in row
    }
    report = dict(
        compiler=sha256(args.compiler),
        policy=sha256(args.policy),
        rounds=args.rounds,
        affinity=[0],
        cases=[],
    )
    for case in manifest["cases"]:
        name = case["name"]
        if name not in verified:
            continue
        modes = ["baseline", args.mode]
        samples = {mode: [] for mode in modes}
        hashes = {}
        host_differences = {}
        for iteration in range(-2, args.rounds):
            for mode in modes if iteration % 2 == 0 else modes[::-1]:
                elapsed = 0.0
                for unit in case["units"]:
                    directory = root / "native-build" / mode / name
                    directory.mkdir(parents=True, exist_ok=True)
                    obj = directory / (unit + ".o")
                    command = [
                        str(args.compiler),
                        str(root / "input" / name / (unit + ".i")),
                        "-O1",
                        "--cpu",
                        "c908",
                        "-o",
                        str(obj),
                    ]
                    if mode != "baseline":
                        command += ["--policy", str(args.policy)]
                    start = time.perf_counter()
                    subprocess.run(command, check=True, capture_output=True)
                    elapsed += time.perf_counter() - start
                    key = mode + "/" + unit
                    digest = sha256(obj)
                    expected = manifest["objects"][name][mode][unit]
                    if digest != expected:
                        # Native compilation is a separate experiment. Preserve
                        # evidence when host-dependent optimization choices differ;
                        # these objects must be linked and validated separately.
                        host_differences[key] = dict(host=expected, native=digest)
                    if key in hashes and hashes[key] != digest:
                        raise RuntimeError(f"unstable native object: {name}/{key}")
                    hashes[key] = digest
                if iteration >= 0:
                    samples[mode].append(elapsed)
        row = dict(
            name=name,
            split=case["split"],
            samples=samples,
            median={m: statistics.median(v) for m, v in samples.items()},
            object_sha256=hashes,
            host_differences=host_differences,
        )
        report["cases"].append(row)
        (root / "compile-final.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            name, row["median"], "host differences:", list(host_differences), flush=True
        )


if __name__ == "__main__":
    main()
