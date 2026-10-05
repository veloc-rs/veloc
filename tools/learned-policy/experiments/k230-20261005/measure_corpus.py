#!/usr/bin/env python3
"""Target-side timings. Fixed seeds, equal repetitions and independent checksums."""
from pathlib import Path
import argparse
import json
import os
import random
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    root = args.root.resolve()
    os.sched_setaffinity(0, {0})
    manifest = json.loads((root / "corpus-build.json").read_text())
    results = []
    rng = random.Random(78239)

    def run(name, mode, repeats, seed):
        r = subprocess.run(
            [str(root / "bin" / name / mode), str(repeats), str(seed)],
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed, checksum = r.stdout.strip().split()
        return float(elapsed), checksum

    for case in manifest["cases"]:
        name = case["name"]
        sample, _ = run(name, "baseline", 20, 17)
        repeats = max(20, min(200000, int(20 * 0.04 / max(sample, 1e-6))))
        reference = {s: run(name, "llvm", repeats, s)[1] for s in (17, 9127)}
        samples = {mode: [] for mode in manifest["modes"]}
        for iteration in range(args.rounds + 1):
            modes = list(samples)
            rng.shuffle(modes)
            seed = (17, 9127)[iteration % 2]
            for mode in modes:
                elapsed, checksum = run(name, mode, repeats, seed)
                if checksum != reference[seed]:
                    raise RuntimeError(
                        f"{name}/{mode}: checksum {checksum} != LLVM {reference[seed]}"
                    )
                if iteration:
                    samples[mode].append(elapsed)
        row = dict(
            case,
            repeats=repeats,
            samples=samples,
            median={m: statistics.median(v) for m, v in samples.items()},
        )
        results.append(row)
        (root / "corpus-times.json").write_text(json.dumps(results, indent=2) + "\n")
        best = min(row["median"], key=row["median"].get)
        print(
            name,
            best,
            round(row["median"]["baseline"] / row["median"][best], 4),
            flush=True,
        )


if __name__ == "__main__":
    main()
