#!/usr/bin/env python3
"""Train pass policies; optionally pretrain on estimates, calibrate on measurements."""
import argparse
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch

from data import digest, load_training, read, write
from evaluation import calibrate, policy_summary, sample_weights


def transform(x):
    return np.sign(x) * np.log1p(np.abs(x))


def network(widths, activation, dropout=0.0):
    layers = []
    for i, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
        layers.append(torch.nn.Linear(inputs, outputs))
        if i + 2 < len(widths):
            layers.append({"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[activation]())
            if dropout:
                layers.append(torch.nn.Dropout(dropout))
    return torch.nn.Sequential(*layers)


def export_layers(net, activation):
    layers = [
        dict(
            weights=layer.weight.detach().tolist(),
            bias=layer.bias.detach().tolist(),
            activation=activation,
        )
        for layer in net
        if isinstance(layer, torch.nn.Linear)
    ]
    layers[-1]["activation"] = "linear"
    return layers


def train_member(widths, training, validation, pretraining, loss_for, config, seed):
    """One seed/capacity trial; calibration data is deliberately unavailable."""
    torch.manual_seed(seed)
    net = network(widths, config["activation"], config["dropout"])
    started = time.perf_counter()

    def optimizer():
        return torch.optim.AdamW(
            net.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

    def step(batch, update):
        x, y, groups = batch
        net.train()
        update.zero_grad()
        loss = loss_for(net(x), y, groups)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        loss.backward()
        update.step()

    if pretraining is not None:
        update = optimizer()
        for _ in range(config["pretrain_epochs"]):
            step(pretraining, update)
    # Estimated-label optimizer momentum is not transferred into measured fitting.
    update = optimizer()
    stale, best, curve = 0, None, []
    vx, vy, vg = validation
    tx, ty, tg = training
    for epoch in range(config["epochs"]):
        step(training, update)
        if epoch % config["validation_interval"]:
            continue
        net.eval()
        with torch.no_grad():
            loss = loss_for(net(vx), vy, vg).item()
            training_loss = loss_for(net(tx), ty, tg).item()
        if not math.isfinite(loss):
            raise ValueError("non-finite validation loss")
        curve.append(
            dict(epoch=epoch, training_loss=training_loss, validation_loss=loss)
        )
        if best is None or loss < best["loss"] - 1e-8:
            best = dict(
                loss=loss, state=copy.deepcopy(net.state_dict()), seed=seed, epoch=epoch
            )
            stale = 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    net.load_state_dict(best.pop("state"))
    net.eval()
    return net, dict(best, seconds=time.perf_counter() - started, curve=curve)


def fit(rows, contract, config):
    measured = [r for r in rows if r["fidelity"] == "measured"]
    training = [r for r in measured if r["split"] == "train"]
    validation = [r for r in measured if r["split"] == "validation"]
    calibration = [r for r in measured if r["split"] == "calibration"]
    estimated = [
        r for r in rows if r["fidelity"] == "estimated" and r["split"] == "train"
    ]
    if not training or not validation:
        raise ValueError(
            "each trained decision needs measured probes from separate training and validation families"
        )
    raw = np.asarray([row["x"] for row in training], dtype=np.float32)
    lower, upper = raw.min(axis=0), raw.max(axis=0)
    transformed = transform(raw)
    weights = sample_weights(training)
    mean = np.average(transformed, axis=0, weights=weights)
    variance = np.average((transformed - mean) ** 2, axis=0, weights=weights)
    std = np.maximum(np.sqrt(variance), 0.1)
    actions = len(contract["actions"])

    def tensors(data):
        x = np.asarray([r["x"] for r in data], dtype=np.float32)
        y = np.asarray([r["y"] for r in data], dtype=np.float32)
        groups = {g: i for i, g in enumerate(sorted({r["group"] for r in data}))}
        # Weights sum to one within each family. Scatter avoids a dense
        # family-by-context matrix for large training corpora.
        grouping = (
            torch.tensor(sample_weights(data) * len(groups)),
            torch.tensor([groups[r["group"]] for r in data]),
            len(groups),
        )
        return (
            torch.tensor(np.clip((transform(x) - mean) / std, -16, 16)),
            torch.tensor(y),
            grouping,
        )

    train_batch = tensors(training)
    validation_batch = tensors(validation)
    tx, ty, _ = train_batch
    vx, _, _ = validation_batch
    pretraining = tensors(estimated) if estimated else None

    def loss_for(prediction, quality, grouping):
        if config["objective"] == "classification":
            loss = torch.nn.functional.cross_entropy(
                prediction, quality.argmax(dim=1), reduction="none"
            )
        else:
            error = prediction - quality
            loss = (
                error.square() * torch.where(error > 0, config["optimism_penalty"], 1.0)
            ).mean(dim=1)
            # Learn relative action preferences as well as log-speedup values.
            # Ties contribute no ranking loss; the measured gap weights mistakes.
            gap = quality[:, :, None] - quality[:, None, :]
            difference = prediction[:, :, None] - prediction[:, None, :]
            ranking = gap.clamp(min=0) * torch.nn.functional.softplus(
                -difference / config["ranking_temperature"]
            )
            loss += config["ranking_weight"] * ranking.mean(dim=(1, 2))
        weights, indices, count = grouping
        families = torch.zeros(count).scatter_add(0, indices, weights * loss)
        tail_size = max(1, math.ceil(len(families) * config["tail_fraction"]))
        # A transparent tail-risk objective, not a claim of distribution-free
        # robustness: poor families cannot disappear in a large corpus average.
        mean = families.mean()
        tail = families.topk(tail_size).values.mean()
        return torch.lerp(mean, tail, config["group_risk_weight"])

    def behavior(net, x, rows):
        with torch.no_grad():
            prediction = net(x).numpy()
        prediction[:, 0] += config["baseline_bias"]
        return policy_summary(
            rows, prediction.argmax(axis=1), config["regression_penalty"]
        )

    best, selected_net, ensemble, capacity = None, None, [], []
    for hidden in config["architectures"] or [config["hidden"]]:
        widths = [raw.shape[1], *hidden, actions]
        parameters = sum(
            (inputs + 1) * outputs for inputs, outputs in zip(widths, widths[1:])
        )
        macs = sum(inputs * outputs for inputs, outputs in zip(widths, widths[1:]))
        if config["max_macs"] is not None and macs > config["max_macs"]:
            capacity.append(
                dict(
                    architecture=widths,
                    parameters=parameters,
                    macs=macs,
                    skipped="inference budget",
                )
            )
            continue
        trials, members = [], []
        architecture_best, architecture_net = None, None
        for seed in config["seeds"]:
            net, result = train_member(
                widths,
                train_batch,
                validation_batch,
                pretraining,
                loss_for,
                config,
                seed,
            )
            trials.append(
                dict(
                    result,
                    training=behavior(net, tx, training),
                    validation=behavior(net, vx, validation),
                )
            )
            members.append(
                dict(seed=seed, layers=export_layers(net, config["activation"]))
            )
            if architecture_best is None or result["loss"] < architecture_best["loss"]:
                architecture_best, architecture_net = result, net
        capacity.append(
            dict(architecture=widths, parameters=parameters, macs=macs, trials=trials)
        )
        rank = (architecture_best["loss"], macs)
        if best is None or rank < (best["loss"], best["macs"]):
            best = dict(
                architecture_best, architecture=widths, parameters=parameters, macs=macs
            )
            selected_net, ensemble = architecture_net, members
    if selected_net is None:
        raise ValueError("no architecture fits the configured inference budget")
    # Only the validation-selected architecture reaches calibration. Expanding a
    # grid never consults calibration/test labels to select network capacity.
    net = selected_net
    layers = export_layers(net, config["activation"])
    layers[-1]["bias"][0] += config["baseline_bias"]

    normalized = tx.numpy()
    selected = [int(np.argmin((normalized**2).mean(axis=1)))]
    distances = ((normalized - normalized[selected[0]]) ** 2).mean(axis=1)
    while len(selected) < min(config["support_points"], len(normalized)):
        index = int(np.argmax(distances))
        if distances[index] < 1e-8:
            break
        selected.append(index)
        distances = np.minimum(
            distances, ((normalized - normalized[index]) ** 2).mean(axis=1)
        )
    points = normalized[selected]
    radius, calibration_report = None, dict(
        accepted=False, reason="missing measured calibration families"
    )
    if calibration:
        cx, _, _ = tensors(calibration)
        nearest = (
            ((cx.numpy()[:, None] - points[None, :]) ** 2).mean(axis=2).min(axis=1)
        )
        with torch.no_grad():
            prediction = net(cx).numpy()
        prediction[:, 0] += config["baseline_bias"]
        bounds = np.asarray([row["x"] for row in calibration])
        in_bounds = ((bounds >= lower) & (bounds <= upper)).all(axis=1)
        radius, calibration_report = calibrate(
            calibration, prediction, in_bounds, nearest, config
        )
    advisor = dict(
        kind="network",
        domain=np.stack([lower, upper], axis=1).tolist(),
        transform="signed_log1p",
        offset=mean.tolist(),
        scale=(1 / std).tolist(),
        support=dict(points=points.tolist(), max_squared_distance=radius or 0.0),
        layers=layers,
    )
    report = dict(
        training_contexts=len(training),
        validation_contexts=len(validation),
        calibration_contexts=len(calibration),
        estimated_pretraining_contexts=len(estimated),
        architecture=best["architecture"],
        parameters=best["parameters"],
        macs=best["macs"],
        capacity_trials=capacity,
        training_signal=dict(
            positive_contexts=int((ty.numpy().max(axis=1) > 0).sum()),
            negative_contexts=int((ty.numpy().min(axis=1) < 0).sum()),
            unique_inputs=len(np.unique(raw, axis=0)),
            varying_features=int((lower != upper).sum()),
        ),
        config=config,
        training_groups=sorted({r["group"] for r in training}),
        validation_groups=sorted({r["group"] for r in validation}),
        calibration_groups=sorted({r["group"] for r in calibration}),
        target_contexts=[
            dict(t)
            for t in sorted(
                {tuple(sorted(r["context"]["labels"].items())) for r in measured}
            )
        ],
        validation_loss=best["loss"],
        seed=best["seed"],
        epoch=best["epoch"],
        support_radius=radius,
        calibration=calibration_report,
        exported=radius is not None,
        generalization_validated=False,
    )
    # Offline ensemble is used for acquisition, not loaded into the compiler.
    acquisition = dict(
        transform="signed_log1p",
        offset=mean.tolist(),
        scale=(1 / std).tolist(),
        members=ensemble,
        objective=config["objective"],
    )
    return advisor if radius is not None else None, report, acquisition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        required=True,
        help="measured objective to optimize, e.g. elapsed_seconds",
    )
    parser.add_argument(
        "--estimated-metric",
        help="optional surrogate objective used only for pretraining",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON with defaults, per-decision overrides and optional requires",
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden", type=int, nargs="*", default=[32, 16])
    parser.add_argument("--noise-floor", type=float, default=0.0075)
    args = parser.parse_args()
    torch.set_num_threads(1)
    configuration = read(args.config) if args.config else {}
    if set(configuration) - {"defaults", "decisions", "requires"}:
        parser.error("unknown configuration field")
    if args.noise_floor < 0:
        parser.error("noise floor must be nonnegative")
    defaults = dict(
        hidden=args.hidden,
        architectures=None,
        max_macs=None,
        dropout=0.0,
        epochs=args.epochs,
        activation="relu",
        objective="regression",
        seeds=[95123, 95124, 95125],
        learning_rate=0.002,
        weight_decay=0.02,
        optimism_penalty=2.0,
        ranking_weight=0.1,
        ranking_temperature=0.02,
        group_risk_weight=0.25,
        tail_fraction=0.25,
        regression_penalty=2.0,
        baseline_bias=0.01,
        validation_interval=20,
        patience=75,
        pretrain_epochs=200,
        support_points=128,
        radii=[1e-6, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        min_training_groups=2,
        min_validation_groups=2,
        min_calibration_groups=5,
        min_benefiting_groups=2,
        min_changed_fraction=0.05,
        max_family_regression=0.01,
    )
    unknown = set(configuration.get("defaults", {})) - defaults.keys()
    if unknown:
        parser.error("unknown training options: " + str(unknown))
    defaults.update(configuration.get("defaults", {}))
    decisions, context_features, samples, provenance = load_training(
        args.measurements, args.noise_floor
    )
    if set(configuration.get("decisions", {})) - decisions.keys():
        parser.error("configuration names an unregistered decision")
    samples = [
        r
        for r in samples
        if r["metric"]
        == (args.metric if r["fidelity"] == "measured" else args.estimated_metric)
    ]
    document = dict(
        version=2,
        context_features=context_features,
        requires=configuration.get("requires", {}),
        decisions={},
    )
    report = dict(
        torch=torch.__version__,
        metric=args.metric,
        estimated_metric=args.estimated_metric,
        datasets=provenance,
        evidence="isolated_interventions",
        generalization_validated=False,
        models={},
    )
    acquisition = dict(version=2, context_features=context_features, decisions={})
    for name in sorted({r["decision"] for r in samples}):
        rows = [r for r in samples if r["decision"] == name]
        config = defaults | configuration.get("decisions", {}).get(name, {})
        if set(config) != set(defaults) or config["objective"] not in (
            "regression",
            "classification",
        ):
            parser.error("invalid training options for " + name)
        architectures = (
            config["architectures"]
            if config["architectures"] is not None
            else [config["hidden"]]
        )
        if (
            not architectures
            or any(
                not isinstance(hidden, list)
                or any(type(width) is not int or width <= 0 for width in hidden)
                for hidden in architectures
            )
            or config["activation"] not in ("relu", "tanh")
            or not 0 <= config["dropout"] < 1
            or (config["max_macs"] is not None and config["max_macs"] < 1)
        ):
            parser.error("invalid network configuration for " + name)
        if (
            config["epochs"] < 1
            or not config["seeds"]
            or config["validation_interval"] < 1
            or config["support_points"] < 1
            or config["min_training_groups"] < 2
            or config["min_validation_groups"] < 2
            or config["min_calibration_groups"] < 2
            or not 1
            <= config["min_benefiting_groups"]
            <= config["min_calibration_groups"]
            or not 0 <= config["group_risk_weight"] <= 1
            or not 0 < config["tail_fraction"] <= 1
            or not 0 <= config["min_changed_fraction"] <= 1
            or not 0 <= config["max_family_regression"] < 1
            or config["ranking_temperature"] <= 0
            or config["ranking_weight"] < 0
            or config["optimism_penalty"] < 1
            or config["regression_penalty"] < 1
            or not config["radii"]
            or any(not math.isfinite(r) or r < 0 for r in config["radii"])
        ):
            parser.error("invalid training budget for " + name)
        counts = {
            split: len(
                {
                    r["group"]
                    for r in rows
                    if r["fidelity"] == "measured" and r["split"] == split
                }
            )
            for split in ("train", "validation", "calibration")
        }
        if (
            counts["train"] < config["min_training_groups"]
            or counts["validation"] < config["min_validation_groups"]
        ):
            report["models"][name] = dict(
                exported=False,
                groups=counts,
                reason="insufficient independent measured training or validation families",
            )
            print(name, "abstain:", report["models"][name]["reason"], flush=True)
            continue
        advisor, summary, ensemble = fit(
            rows,
            decisions[name]["contract"],
            config,
        )
        if advisor:
            document["decisions"][name] = dict(decisions[name], advisor=advisor)
        report["models"][name] = summary
        acquisition["decisions"][name] = dict(decisions[name], **ensemble)
        print(
            name,
            "exported for evaluation" if advisor else "abstain",
            summary["calibration"]["reason"],
            flush=True,
        )
    if not report["models"]:
        parser.error("no observations match the requested metric")
    write(args.output, document)
    report["model_sha256"] = digest(args.output)
    write(args.output.with_suffix(".training.json"), report)
    write(args.output.with_suffix(".ensemble.json"), acquisition)


if __name__ == "__main__":
    main()
