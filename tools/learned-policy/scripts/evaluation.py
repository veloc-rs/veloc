"""Family-balanced objectives and intervention diagnostics, independent of passes.

These rewards describe isolated probes. They do not estimate the performance of
combining every predicted decision into a complete compiled program.
"""

from collections import Counter, defaultdict
import json

import numpy as np


def sample_weights(rows):
    """Equal families, then equal program/input/target cases within each family."""
    units = [
        (
            r["group"],
            r["case"],
            json.dumps([r["context"], r["environment"], r["inputs"]], sort_keys=True),
        )
        for r in rows
    ]
    contexts = Counter(units)
    family_units = Counter(unit[0] for unit in contexts)
    weights = np.asarray(
        [1 / (contexts[u] * family_units[u[0]]) for u in units], dtype=np.float32
    )
    return weights / weights.sum()


def group_means(rows, values):
    totals, weights = defaultdict(float), defaultdict(float)
    for row, value, weight in zip(rows, values, sample_weights(rows)):
        totals[row["group"]] += float(value * weight)
        weights[row["group"]] += float(weight)
    return {name: totals[name] / weights[name] for name in sorted(totals)}


def policy_summary(rows, choices, regression_penalty):
    quality = np.asarray([r["y"] for r in rows])
    rewards = quality[np.arange(len(rows)), choices]
    weights = sample_weights(rows)
    groups = group_means(rows, rewards)
    changed = np.asarray([r["changed"] for r in rows])[np.arange(len(rows)), choices]
    return dict(
        contexts=len(rows),
        families=len(groups),
        family_log_rewards=groups,
        family_balanced_speedup=float(np.exp(np.mean(list(groups.values())))),
        worst_family_speedup=float(np.exp(min(groups.values()))),
        benefiting_families=sum(value > 0 for value in groups.values()),
        regressing_families=sum(value < 0 for value in groups.values()),
        advice_fraction=float(np.average(choices != 0, weights=weights)),
        changed_binary_fraction=float(np.average(changed, weights=weights)),
        changed_programs=sorted({r["case"] for r, c in zip(rows, changed) if c}),
        regret=float(np.average(quality.max(axis=1) - rewards, weights=weights)),
        penalized_reward=float(
            np.average(
                rewards + np.minimum(rewards, 0) * (regression_penalty - 1),
                weights=weights,
            )
        ),
    )


def calibrate(rows, prediction, supported, nearest, config):
    """Select abstention only on independent measured calibration families."""
    report = dict(
        evidence="isolated_interventions",
        accepted=False,
        candidates=[],
        reason="insufficient independent calibration families",
    )
    if rows:
        report["in_domain_fraction"] = float(
            np.average(supported, weights=sample_weights(rows))
        )
        report["ungated"] = policy_summary(
            rows, prediction.argmax(axis=1), config["regression_penalty"]
        )
    if len({r["group"] for r in rows}) < config["min_calibration_groups"]:
        return None, report

    def qualifies(summary):
        return (
            summary["benefiting_families"] >= config["min_benefiting_groups"]
            and summary["changed_binary_fraction"] >= config["min_changed_fraction"]
            and summary["worst_family_speedup"] >= 1 - config["max_family_regression"]
        )

    # A neural policy should justify its cost against constant strategies too.
    # Apply identical coverage/regression criteria to those cheaper baselines.
    report["fixed_actions"] = [
        policy_summary(rows, np.full(len(rows), action), config["regression_penalty"])
        for action in range(prediction.shape[1])
    ]
    best_radius, best_reward = None, 0.0
    for summary in report["fixed_actions"]:
        if qualifies(summary):
            best_reward = max(best_reward, summary["penalized_reward"])
    report["baseline_penalized_reward"] = best_reward
    for radius in sorted(config["radii"]):
        choices = np.where(
            supported & (nearest <= radius), prediction.argmax(axis=1), 0
        )
        summary = policy_summary(rows, choices, config["regression_penalty"])
        eligible = qualifies(summary)
        report["candidates"].append(dict(radius=radius, qualifies=eligible, **summary))
        if eligible and summary["penalized_reward"] > best_reward + 1e-8:
            best_radius, best_reward = radius, summary["penalized_reward"]
    report.update(
        accepted=best_radius is not None,
        radius=best_radius,
        reason=(
            "candidate requires frozen whole-program evaluation"
            if best_radius is not None
            else "no radius beats qualifying heuristic/fixed baselines under the family criteria"
        ),
    )
    return best_radius, report
