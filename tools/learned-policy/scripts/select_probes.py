#!/usr/bin/env python3
"""Choose a training batch using ensemble disagreement, diversity and exploration.

This is a practical acquisition heuristic, not a calibrated confidence interval
or a reproduction of a paper's sampling algorithm. Test programs are excluded.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from data import digest, read, validate_manifest, write


def predictions(head, x):
    x = np.sign(x) * np.log1p(np.abs(x))
    x = np.clip((x - head["offset"]) * head["scale"], -16, 16).astype(np.float32)
    outputs = []
    for member in head["members"]:
        y = x.copy()
        for layer in member["layers"]:
            y = y @ np.asarray(layer["weights"], dtype=np.float32).T + layer["bias"]
            if layer["activation"] == "relu":
                y = np.maximum(y, 0)
            elif layer["activation"] == "tanh":
                y = np.tanh(y)
        if head["objective"] == "classification":
            y = np.exp(y - y.max(axis=1, keepdims=True))
            y /= y.sum(axis=1, keepdims=True)
        outputs.append(y)
    return x, np.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ensemble", type=Path)
    parser.add_argument("--observed", type=Path, nargs="*", default=[])
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4105)
    parser.add_argument("--exploration", type=float, default=0.2)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.count < 1 or not 0 <= args.exploration <= 1:
        parser.error("invalid acquisition budget")
    document = validate_manifest(read(args.manifest))
    if document.get("kind") == "whole_policy":
        parser.error("acquisition requires an intervention dataset")
    ensemble = read(args.ensemble) if args.ensemble else None
    if ensemble and ensemble["context_features"] != document["context_features"]:
        parser.error("ensemble context schema mismatch")
    observed = set()
    for path in args.observed:
        report = read(path)
        if report["manifest_sha256"] != digest(args.manifest):
            parser.error("observed report belongs to another manifest")
        if report["evaluator"]["fidelity"] != "measured":
            parser.error(
                "estimated observations must remain eligible for real measurement"
            )
        observed.update(r["id"] for r in report["results"] if "error" not in r)
    rows = [r for r in document["experiments"] if r["split"] == "train"]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)
    buckets = {}
    for row in rows:
        buckets.setdefault(row["decision"], []).append(row)
    ranked = {}
    for name, candidates in buckets.items():
        x = np.asarray(
            [
                r["features"]
                + [r["context"]["features"][n] for n in document["context_features"]]
                for r in candidates
            ],
            dtype=np.float32,
        )
        head = ensemble["decisions"].get(name) if ensemble else None
        if head:
            if head["contract"] != document["decisions"][name]["contract"]:
                parser.error("ensemble decision schema mismatch: " + name)
            z, predicted = predictions(head, x)
            mean, uncertainty = predicted.mean(axis=0), predicted.std(axis=0)
            score = (mean - mean[:, :1] + args.uncertainty_weight * uncertainty).max(
                axis=1
            )
            # Ranking avoids mixing score scales from different objectives.
            score = np.argsort(np.argsort(score)).astype(float) / max(1, len(score) - 1)
        else:
            z = np.sign(x) * np.log1p(np.abs(x))
            z = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 0.1)
            score = np.zeros(len(x))
        available = np.asarray([r["id"] not in observed for r in candidates])
        distance = np.full(len(x), np.inf)
        for index in np.flatnonzero(~available):
            distance = np.minimum(distance, ((z - z[index]) ** 2).mean(axis=1))
        if available.all():
            distance.fill(1.0)
        family_counts = Counter(r["group"] for r in candidates if r["id"] in observed)
        order = []
        for _ in range(min(args.count, int(available.sum()))):
            # Acquire from underrepresented families before choosing informative
            # sites. A single large program cannot consume the measurement pool.
            fewest = min(
                family_counts[r["group"]] for r, ok in zip(candidates, available) if ok
            )
            eligible = available & np.asarray(
                [family_counts[r["group"]] == fewest for r in candidates]
            )
            if rng.random() < args.exploration:
                index = int(rng.choice(np.flatnonzero(eligible)))
            else:
                priority = score + distance / max(
                    float(distance[available].max()), 1e-8
                )
                index = int(np.argmax(np.where(eligible, priority, -np.inf)))
            available[index] = False
            family_counts[candidates[index]["group"]] += 1
            order.append(candidates[index]["id"])
            distance = np.minimum(distance, ((z - z[index]) ** 2).mean(axis=1))
        ranked[name] = order
    # Round-robin across decisions keeps a prolific pass from consuming a batch.
    chosen = []
    while len(chosen) < args.count and any(ranked.values()):
        for name in sorted(ranked):
            if ranked[name] and len(chosen) < args.count:
                chosen.append(ranked[name].pop(0))
    write(args.output, chosen)
    print(
        json.dumps(
            dict(
                selected=len(chosen),
                available=sum(r["id"] not in observed for r in rows),
            )
        )
    )


if __name__ == "__main__":
    main()
