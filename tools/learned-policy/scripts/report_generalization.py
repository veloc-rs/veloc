#!/usr/bin/env python3
"""Evaluate a frozen whole-program policy on held-out program families."""
import argparse
import math
from pathlib import Path

import numpy as np

from data import Partitions, digest, read, write
from evaluation import policy_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, nargs="+")
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-families", type=int, default=5)
    parser.add_argument("--min-benefiting-families", type=int, default=2)
    parser.add_argument("--max-program-regression", type=float, default=0.01)
    parser.add_argument("--bootstrap-rounds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=4105)
    args = parser.parse_args()
    if (
        args.min_families < 2
        or args.min_benefiting_families < 1
        or args.bootstrap_rounds < 1
        or not 0 <= args.max_program_regression < 1
    ):
        parser.error("invalid evaluation criteria")
    training = read(args.training)
    partitions = Partitions()
    # All previously used development data counts, including failed measurements.
    # Validation/calibration have already influenced the model and are not tests.
    for dataset in training["datasets"]:
        for row in dataset["development"]:
            partitions.add(dict(row, split="train"))
    rows, failures, seen, datasets, expected = [], [], set(), [], {}
    for path in args.measurements:
        report = read(path)
        if (
            report.get("kind") != "whole_policy"
            or report["model"]["sha256"] != training["model_sha256"]
        ):
            parser.error(
                "evaluation must use the exact frozen training output: " + str(path)
            )
        evaluator = report["evaluator"]
        if (
            evaluator["fidelity"] != "measured"
            or evaluator["metric"] != training["metric"]
            or evaluator["direction"] not in ("min", "max")
        ):
            parser.error("evaluation metric or fidelity differs from training")
        datasets.append(dict(path=str(path), sha256=digest(path)))
        for row in report["test_programs"]:
            partitions.add(row)
            identity = (report["manifest_sha256"], row["id"], report["evaluator_name"])
            expected[identity] = row["program"]
        for row in report["results"]:
            if row["split"] != "test":
                continue
            if not row.get("source_sha256"):
                parser.error("test programs must include source provenance")
            partitions.add(row)
            identity = (report["manifest_sha256"], row["id"], report["evaluator_name"])
            if identity in seen:
                parser.error("duplicate test observation")
            if identity not in expected:
                parser.error("measurement absent from declared test corpus")
            seen.add(identity)
            if "error" in row:
                failures.append(dict(program=row["program"], error=row["error"]))
                continue
            pair = row["actions"]["1"]
            baseline, candidate = (
                pair["median"]["baseline"],
                pair["median"]["candidate"],
            )
            if any(not math.isfinite(x) or x <= 0 for x in (baseline, candidate)):
                parser.error("invalid runtime measurement")
            sign = 1 if evaluator["direction"] == "min" else -1
            reward = sign * math.log(baseline / candidate)
            changed = pair["sha256"]["baseline"] != pair["sha256"]["candidate"]
            rows.append(
                dict(
                    case=row["program"],
                    group=row["group"],
                    context=row["context"],
                    inputs=row.get("variables", {}),
                    environment=report["environment"],
                    y=[0.0, reward],
                    changed=[False, changed],
                )
            )
    if not rows:
        parser.error("no successful held-out whole-program measurements")
    summary = policy_summary(rows, np.ones(len(rows), dtype=int), 1.0)
    # Cluster bootstrap over family means: contexts/inputs from one family are
    # not independent programs. This interval is conditional on this corpus; it
    # neither models every source of timing noise nor proves universal transfer.
    values = list(summary["family_log_rewards"].values())
    rng = np.random.default_rng(args.seed)
    means = [
        float(np.mean(rng.choice(values, size=len(values))))
        for _ in range(args.bootstrap_rounds)
    ]
    interval = np.exp(np.quantile(means, [0.025, 0.975])).tolist()
    worst = min(math.exp(row["y"][1]) for row in rows)
    missing = [expected[key] for key in expected.keys() - seen]
    qualifies = (
        not failures
        and not missing
        and summary["families"] >= args.min_families
        and summary["benefiting_families"] >= args.min_benefiting_families
        and interval[0] > 1.0
        and worst >= 1 - args.max_program_regression
        and summary["changed_binary_fraction"] > 0
    )
    write(
        args.output,
        dict(
            evidence="frozen_whole_program_policy_on_held_out_families",
            qualifies_for_measured_distribution=qualifies,
            model_sha256=training["model_sha256"],
            training_sha256=digest(args.training),
            datasets=datasets,
            criteria=dict(
                min_families=args.min_families,
                min_benefiting_families=args.min_benefiting_families,
                max_program_regression=args.max_program_regression,
            ),
            family_bootstrap_speedup_interval=interval,
            bootstrap_rounds=args.bootstrap_rounds,
            seed=args.seed,
            worst_program_speedup=worst,
            failures=failures,
            missing_measurements=sorted(missing),
            summary=summary,
            programs=[
                dict(
                    program=r["case"],
                    group=r["group"],
                    inputs=r["inputs"],
                    environment=r["environment"],
                    speedup=math.exp(r["y"][1]),
                    changed=r["changed"][1],
                )
                for r in rows
            ],
        ),
    )
    print(
        "qualifies" if qualifies else "insufficient evidence",
        summary["family_balanced_speedup"],
    )


if __name__ == "__main__":
    main()
