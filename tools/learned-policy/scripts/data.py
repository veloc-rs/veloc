"""Shared, versioned experiment contracts; no target or benchmark knowledge."""

import hashlib
import json
import math
from pathlib import Path

VERSION = 2
SPLITS = ("train", "validation", "calibration", "test")


class Partitions:
    """Keep related programs together, even across datasets and target machines."""

    def __init__(self):
        self.assignments = {}

    def add(self, row):
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("invalid split: " + split)
        identities = [("family", row["group"]), ("program", row["program"])]
        if row.get("source_sha256"):
            identities.append(("source bundle", row["source_sha256"]))
        for identity in identities:
            if self.assignments.setdefault(identity, split) != split:
                raise ValueError(f"{identity[0]} crosses partitions: {identity[1]}")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expand(command, variables):
    """An exact list-valued placeholder expands argv, never shell syntax."""
    result = []
    for arg in command:
        if arg.startswith("{") and arg.endswith("}") and arg[1:-1] in variables:
            value = variables[arg[1:-1]]
            result.extend(map(str, value if isinstance(value, list) else [value]))
        else:
            result.append(arg.format_map(variables))
    return result


def validate_manifest(document):
    if document["version"] != VERSION:
        raise ValueError("expected experiment format 2")
    if len(set(document["context_features"])) != len(document["context_features"]):
        raise ValueError("duplicate context feature")
    partitions, ids = Partitions(), set()
    for row in document["experiments"]:
        if row["id"] in ids:
            raise ValueError("duplicate experiment id: " + row["id"])
        ids.add(row["id"])
        partitions.add(row)
        if document.get("kind") == "whole_policy":
            if set(row["candidates"]) != {"1"}:
                raise ValueError(
                    "whole-policy comparison needs exactly one frozen candidate"
                )
            continue
        schema = document["decisions"][row["decision"]]["contract"]
        if len(row["features"]) != len(schema["features"]):
            raise ValueError("feature dimension mismatch")
        inputs = row["features"] + [
            row["context"]["features"][name] for name in document["context_features"]
        ]
        if any(not math.isfinite(x) for x in inputs):
            raise ValueError("non-finite feature")
        expected = {str(i) for i in range(1, len(schema["actions"]))}
        if set(row["candidates"]) != expected:
            raise ValueError("each context must declare every alternative action")
    return document


def load_training(paths, noise_floor):
    """Reports retain evaluator identity. Estimated labels are never truth labels."""
    schema, context_features = {}, None
    samples, partitions, provenance = [], Partitions(), []
    seen = set()
    for path in paths:
        report = read(path)
        if report["version"] != VERSION:
            raise ValueError("expected measurement format 2: " + str(path))
        if report.get("kind") == "whole_policy":
            raise ValueError(
                "whole-policy evaluation is not intervention training data"
            )
        if context_features is None:
            context_features = report["context_features"]
        if context_features != report["context_features"]:
            raise ValueError("incompatible context feature order")
        for name, contract in report["decisions"].items():
            if schema.setdefault(name, contract) != contract:
                raise ValueError("incompatible decision contract: " + name)
        evaluator = report["evaluator"]
        if evaluator["fidelity"] not in ("measured", "estimated"):
            raise ValueError("unknown evaluation fidelity")
        if evaluator["direction"] not in ("min", "max"):
            raise ValueError("unknown objective direction")
        provenance.append(
            dict(
                path=str(path),
                sha256=digest(path),
                evaluator=evaluator,
                development=[
                    {
                        k: row[k]
                        for k in ("program", "group", "source_sha256")
                        if k in row
                    }
                    for row in report["results"]
                    if row["split"] != "test"
                ],
            )
        )
        for row in report["results"]:
            partitions.add(row)
            if row["split"] == "test" or "error" in row:
                continue
            # Dataset identity and experiment id prevent accidentally repeating a
            # measurement file under another filename and weighting it twice.
            key = (report["manifest_sha256"], row["id"], report["evaluator_name"])
            if key in seen:
                raise ValueError("duplicate training observation: " + str(key))
            seen.add(key)
            sign = 1 if evaluator["direction"] == "min" else -1
            rewards, changed = [0.0], [False]
            for i in range(1, len(schema[row["decision"]]["contract"]["actions"])):
                pair = row["actions"][str(i)]
                if any(not math.isfinite(x) or x <= 0 for x in pair["median"].values()):
                    raise ValueError("invalid evaluation metric")
                reward = sign * math.log(
                    pair["median"]["baseline"] / pair["median"]["candidate"]
                )
                rewards.append(0.0 if abs(reward) < noise_floor else reward)
                changed.append(
                    pair["sha256"]["baseline"] != pair["sha256"]["candidate"]
                )
            samples.append(
                dict(
                    case=row["program"],
                    group=row["group"],
                    split=row["split"],
                    decision=row["decision"],
                    fidelity=evaluator["fidelity"],
                    metric=evaluator["metric"],
                    context=row["context"],
                    # Provenance only: never concatenate program or hardware IDs
                    # into neural inputs. They define balanced sampling strata.
                    environment=report.get("environment", report.get("host", {})),
                    inputs=row.get("variables", {}),
                    x=row["features"]
                    + [row["context"]["features"][n] for n in context_features],
                    y=rewards,
                    changed=changed,
                )
            )
    return schema, context_features, samples, provenance
