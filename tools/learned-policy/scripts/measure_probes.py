#!/usr/bin/env python3
"""Evaluate interventions using a manifest-selected real or estimated metric."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import time

from data import SPLITS, digest, expand, read, validate_manifest, write


def evaluate(artifact, row, evaluator, root, timeout):
    variables = dict(
        row.get("variables", {}), artifact=str((root / artifact).resolve())
    )
    command = expand(evaluator["command"], variables)
    start = time.perf_counter()
    process = subprocess.run(
        command, cwd=root, capture_output=True, text=True, timeout=timeout, check=True
    )
    elapsed = time.perf_counter() - start
    output = process.stdout
    for pattern in evaluator.get("checks", []):
        if not re.search(pattern, output, re.MULTILINE):
            raise ValueError("output validation failed: " + pattern)
    metric = evaluator["value"]
    if metric["kind"] == "wall_time":
        value = elapsed
    elif metric["kind"] == "regex":
        match = re.search(metric["pattern"], output, re.MULTILINE)
        if match is None:
            raise ValueError("missing metric in output: " + output[-1000:])
        value = float(match.group(1)) * metric.get("scale", 1.0)
    else:
        raise ValueError("unknown metric extraction method")
    checksum = None
    if "checksum" in evaluator:
        match = re.search(evaluator["checksum"], output, re.MULTILINE)
        if match is None:
            raise ValueError("missing checksum")
        checksum = match.group(1)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("metric must be positive and finite")
    return value, checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--affinity",
        type=int,
        nargs="+",
        help="optional CPU indices on the measuring host",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=4105)
    parser.add_argument(
        "--selection",
        type=Path,
        help="JSON list of experiment ids chosen for this batch",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=["train", "validation", "calibration"],
    )
    parser.add_argument(
        "--environment",
        default=platform.node(),
        help="physical execution environment ID, recorded as provenance, not a feature",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.warmups < 0 or args.timeout <= 0:
        parser.error("invalid measurement budget")
    if args.affinity:
        if not hasattr(os, "sched_setaffinity"):
            parser.error("this host does not support CPU affinity")
        os.sched_setaffinity(0, set(args.affinity))
    document = validate_manifest(read(args.manifest))
    if document.get("complete") is False:
        parser.error("dataset build is incomplete")
    evaluator = document["evaluators"][args.evaluator]
    if evaluator["fidelity"] not in ("measured", "estimated") or evaluator[
        "direction"
    ] not in ("min", "max"):
        parser.error("invalid evaluator fidelity or direction")
    if evaluator["fidelity"] == "measured" and not (
        evaluator.get("checks") or evaluator.get("checksum")
    ):
        parser.error("measured programs require an output validator or checksum")
    root = args.manifest.resolve().parent
    selection = set(read(args.selection)) if args.selection else None
    if selection is not None and not selection <= {
        r["id"] for r in document["experiments"]
    }:
        parser.error("selection contains unknown experiment ids")
    rows = [
        r
        for r in document["experiments"]
        if r["split"] in args.splits and (selection is None or r["id"] in selection)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    report = dict(
        version=2,
        kind=document.get("kind", "interventions"),
        manifest_sha256=digest(args.manifest),
        evaluator_name=args.evaluator,
        evaluator=evaluator,
        environment=args.environment,
        decisions=document["decisions"],
        context_features=document["context_features"],
        host=dict(
            system=platform.platform(),
            machine=platform.machine(),
            affinity=(
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
        ),
        rounds=args.rounds,
        warmups=args.warmups,
        seed=args.seed,
        results=[],
    )
    if "model" in document:
        report["model"] = document["model"]
    if document.get("kind") == "whole_policy":
        report["test_programs"] = [
            {k: row[k] for k in ("id", "program", "group", "split", "source_sha256")}
            for row in document["experiments"]
            if row["split"] == "test"
        ]
    # Identical executables with identical inputs must not acquire fake gains
    # from independent noisy timings. Cache complete paired measurements.
    cache = {}
    for row in rows:
        result = {
            k: row[k]
            for k in (
                "id",
                "program",
                "group",
                "split",
                "context",
            )
        }
        if document.get("kind") != "whole_policy":
            result.update(decision=row["decision"], features=row["features"])
        result["actions"] = {}
        result["variables"] = row.get("variables", {})
        if "source_sha256" in row:
            result["source_sha256"] = row["source_sha256"]
        try:
            baseline = row["baseline"]
            base_hash = digest(root / baseline)
            for action, candidate in row["candidates"].items():
                candidate_hash = digest(root / candidate)
                key = (
                    base_hash,
                    candidate_hash,
                    json.dumps(row.get("variables", {}), sort_keys=True),
                )
                if key not in cache:
                    samples = {"baseline": [], "candidate": []}
                    reference = None
                    for iteration in range(args.warmups + args.rounds):
                        order = ["baseline", "candidate"]
                        if iteration % 2:
                            order.reverse()
                        readings = {}
                        for mode in order:
                            if readings and base_hash == candidate_hash:
                                value, checksum = next(iter(readings.values()))
                            else:
                                artifact = baseline if mode == "baseline" else candidate
                                value, checksum = evaluate(
                                    artifact, row, evaluator, root, args.timeout
                                )
                            if reference is not None and checksum != reference:
                                raise ValueError(
                                    "candidate checksum differs from baseline"
                                )
                            reference = checksum
                            readings[mode] = (value, checksum)
                            if iteration >= args.warmups:
                                samples[mode].append(value)
                    cache[key] = dict(
                        samples=samples,
                        median={k: statistics.median(v) for k, v in samples.items()},
                        sha256=dict(baseline=base_hash, candidate=candidate_hash),
                    )
                result["actions"][action] = cache[key]
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            result["error"] = str(error)
        report["results"].append(result)
        write(args.output, report)
        print(row["id"], "error" if "error" in result else "ok", flush=True)


if __name__ == "__main__":
    main()
