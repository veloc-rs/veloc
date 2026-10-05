#!/usr/bin/env python3
"""Serial K230 validation and paired timings of the pinned Embench subset."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--modes", nargs="+", default=["baseline", "learned", "llvm"])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seconds", type=float, default=0.15)
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--report", default="timings.json")
    args = parser.parse_args()
    root = args.root.resolve()
    os.sched_setaffinity(0, {0})
    rng = random.Random(79261)
    manifest = json.loads((root / "manifest.json").read_text())

    def run(exe, repeats):
        result = subprocess.run(
            [str(exe), str(repeats)], capture_output=True, text=True, timeout=60
        )
        if result.returncode:
            raise RuntimeError(
                f"{exe}: exit {result.returncode}; {result.stdout} {result.stderr}"
            )
        fields = result.stdout.strip().split()
        if len(fields) != 3:
            raise RuntimeError(f"{exe}: unexpected benchmark output {result.stdout!r}")
        elapsed, checksum, valid = fields
        if valid != "1":
            raise RuntimeError(f"{exe}: upstream verifier returned {valid}")
        return float(elapsed), checksum

    def measure(variants, repeats, rounds):
        reference = None
        samples = {key: [] for key in variants}
        groups = {}
        for mode, path in variants.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            groups.setdefault(digest, []).append(mode)
        for iteration in range(rounds + 1):
            order = list(groups.values())
            rng.shuffle(order)
            for aliases in order:
                mode = aliases[0]
                elapsed, checksum = run(variants[mode], repeats)
                if reference is None:
                    reference = checksum
                if checksum != reference:
                    raise RuntimeError(
                        f"{mode}: return checksum {checksum} != {reference}"
                    )
                if iteration:
                    for alias in aliases:
                        samples[alias].append(elapsed)
        return dict(
            repeats=repeats,
            checksum=reference,
            samples=samples,
            median={key: statistics.median(values) for key, values in samples.items()},
            execution_groups=list(groups.values()),
        )

    report = dict(
        revision=manifest["revision"],
        models=manifest["models"],
        uname=list(os.uname()),
        cpuinfo=Path("/proc/cpuinfo").read_text(),
        cpu=0,
        rounds=args.rounds,
        seconds=args.seconds,
        load_before=os.getloadavg(),
        cases=[],
    )
    if args.probes:
        calibration = {
            row["name"]: row
            for row in json.loads((root / "timings.json").read_text())["cases"]
            if "error" not in row
        }
        probes = json.loads((root / "probes/manifest.json").read_text())
        path = root / "probes/times.json"
        results = json.loads(path.read_text()) if path.exists() else {}
        for row in probes:
            if (
                not row["changed"]
                or row["measurement"] in results
                or row["name"] not in calibration
            ):
                continue
            tag, name = row["measurement"], row["name"]
            variants = dict(
                baseline=root / "bin" / name / "baseline",
                changed=root / "probes" / tag / "run",
            )
            try:
                result = measure(variants, calibration[name]["repeats"], args.rounds)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                result = dict(error=str(e))
            results[tag] = result
            path.write_text(json.dumps(results, indent=2) + "\n")
            print(
                tag,
                name,
                result.get(
                    "error",
                    round(
                        result.get("median", {}).get("baseline", 1)
                        / result.get("median", {}).get("changed", 1),
                        4,
                    ),
                ),
                flush=True,
            )
        return
    for case in manifest["cases"]:
        if case["split"] not in args.splits:
            continue
        row = dict(name=case["name"], split=case["split"])
        if case.get("error"):
            row["error"] = "build: " + case["error"]
        else:
            name = case["name"]
            variants = {mode: root / "bin" / name / mode for mode in args.modes}
            try:
                initial, _ = run(root / "bin" / name / "baseline", 1)
                repeats = max(
                    1, min(10000000, math.ceil(args.seconds / max(initial, 1e-6)))
                )
                if args.check_only:
                    for path in variants.values():
                        run(path, 1)
                    row["verified"] = True
                else:
                    # One cold invocation can overestimate steady-state cost.
                    # Calibrate again at scale; keep the same count for all modes.
                    for _ in range(3):
                        elapsed, _ = run(root / "bin" / name / "baseline", repeats)
                        if elapsed >= args.seconds * 0.8 or repeats == 10000000:
                            break
                        repeats = min(
                            10000000,
                            math.ceil(repeats * args.seconds / max(elapsed, 1e-6)),
                        )
                    row.update(measure(variants, repeats, args.rounds))
                    row["sha256"] = {
                        mode: hashlib.sha256(path.read_bytes()).hexdigest()
                        for mode, path in variants.items()
                    }
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                row["error"] = str(e)
        report["cases"].append(row)
        report["load_after"] = os.getloadavg()
        (root / args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(row["name"], row.get("error", row.get("median", "verified")), flush=True)


if __name__ == "__main__":
    main()
