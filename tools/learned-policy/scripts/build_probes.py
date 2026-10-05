#!/usr/bin/env python3
"""Build context interventions from compiler traces and explicit command templates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from data import Partitions, digest, expand, read, validate_manifest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        help="build whole programs with this frozen model instead of isolated probes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="optional contexts per program, sampled before measuring",
    )
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    config = read(args.experiment)
    if config["version"] != 2 or (args.limit is not None and args.limit < 1):
        parser.error("expected experiment format 2 and a positive optional limit")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_vars = dict(
        config.get("variables", {}), source_root=str(args.experiment.resolve().parent)
    )
    names, partitions = set(), Partitions()
    for program in config["programs"]:
        if program["name"] in names:
            parser.error("duplicate program name: " + program["name"])
        names.add(program["name"])
        partitions.add(dict(program, program=program["name"]))
    empty = root / "baseline-policy.json"
    write(empty, dict(version=2, context_features=[], requires={}, decisions={}))
    candidate_policy = None
    if args.policy:
        candidate_policy = root / ("candidate-policy" + args.policy.suffix)
        candidate_policy.write_bytes(args.policy.read_bytes())
    result = dict(
        version=2,
        kind="whole_policy" if candidate_policy else "interventions",
        complete=False,
        context_features=None,
        decisions={},
        evaluators=config["evaluators"],
        experiments=[],
        provenance=dict(
            experiment_sha256=digest(args.experiment),
            configuration=config,
            tools={},
            programs=[],
        ),
    )
    if candidate_policy:
        result["model"] = dict(
            sha256=digest(candidate_policy), file=candidate_policy.name
        )

    def run(command, variables, log):
        argv = expand(command, variables)
        executable = shutil.which(argv[0]) or argv[0]
        executable = Path(executable)
        if not executable.is_absolute():
            executable = args.experiment.resolve().parent / executable
        if (
            executable.is_file()
            and str(executable) not in result["provenance"]["tools"]
        ):
            result["provenance"]["tools"][str(executable)] = digest(executable)
        process = subprocess.run(
            argv,
            cwd=args.experiment.resolve().parent,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        log.write_text(process.stdout + process.stderr)
        if process.returncode:
            raise RuntimeError(f"command failed; see {log}: {argv}")
        return argv

    for program in config["programs"]:
        # Paths use a digest; user-facing names never become path components.
        tag = hashlib.sha256(program["name"].encode()).hexdigest()[:16]
        directory = root / tag
        directory.mkdir(exist_ok=True)
        variables = base_vars | program.get("variables", {})
        compile_command = program.get("compile", config.get("compile"))
        link_command = program.get("link", config.get("link"))
        if not compile_command or not link_command:
            parser.error("every program needs compile and link commands")
        objects, contexts, source_hashes, sources, context = [], [], {}, [], None
        for index, source in enumerate(program["sources"]):
            source = Path(source.format_map(variables))
            if not source.is_absolute():
                source = args.experiment.resolve().parent / source
            obj, trace = (
                directory / f"baseline-{index}.o",
                directory / f"baseline-{index}.jsonl",
            )
            run(
                compile_command,
                variables
                | dict(
                    source=str(source),
                    output=str(obj),
                    trace=str(trace),
                    policy=str(empty),
                ),
                directory / f"baseline-{index}.log",
            )
            objects.append(obj)
            sources.append(source)
            source_hashes[str(source)] = digest(source)
            lines = [json.loads(line) for line in trace.read_text().splitlines()]
            header = lines[0]
            if header["version"] != 2:
                raise ValueError("compiler must emit trace format 2")
            if set(config.get("decisions", {})) - header["decisions"].keys():
                raise ValueError("configuration names an unregistered decision")
            if context is not None and context != header["context"]:
                raise ValueError("translation units use different target contexts")
            context = header["context"]
            names = sorted(context["features"])
            if result["context_features"] is None:
                result["context_features"] = names
            if result["context_features"] != names:
                raise ValueError("target context schemas differ")
            for name, contract in header["decisions"].items():
                head = dict(
                    contract=contract,
                    max_deviations=config.get("decisions", {})
                    .get(name, {})
                    .get("max_deviations"),
                )
                if result["decisions"].setdefault(name, head) != head:
                    raise ValueError("incompatible decision contract: " + name)
            seen = set()
            for observation in lines[1:]:
                key = (observation["decision"], tuple(observation["features"]))
                if key in seen:
                    continue
                seen.add(key)
                contexts.append(
                    dict(
                        decision=key[0],
                        features=list(key[1]),
                        unit=index,
                        source=str(source),
                    )
                )
        if context is None:
            raise ValueError("program has no sources")
        source_sha256 = hashlib.sha256(
            json.dumps(sorted(source_hashes.values())).encode()
        ).hexdigest()
        partitions.add(
            dict(program, program=program["name"], source_sha256=source_sha256)
        )
        baseline = directory / "baseline"
        run(
            link_command,
            variables | dict(objects=list(map(str, objects)), output=str(baseline)),
            directory / "baseline-link.log",
        )
        result["provenance"]["programs"].append(
            dict(name=program["name"], sources=source_hashes)
        )
        if candidate_policy:
            # Apply the frozen policy to every translation unit together. This
            # measures interactions between decisions and passes, not probe sums.
            candidate_objects = []
            for index, source in enumerate(sources):
                obj = directory / f"policy-{index}.o"
                run(
                    compile_command,
                    variables
                    | dict(
                        source=str(source),
                        output=str(obj),
                        policy=str(candidate_policy),
                        trace=str(directory / f"policy-{index}.jsonl"),
                    ),
                    directory / f"policy-{index}.log",
                )
                candidate_objects.append(obj)
            candidate = directory / "policy"
            run(
                link_command,
                variables
                | dict(
                    objects=list(map(str, candidate_objects)),
                    output=str(candidate),
                ),
                directory / "policy-link.log",
            )
            result["experiments"].append(
                dict(
                    id=tag,
                    program=program["name"],
                    group=program["group"],
                    split=program["split"],
                    source_sha256=source_sha256,
                    context=context,
                    baseline=str(baseline.relative_to(root)),
                    candidates={"1": str(candidate.relative_to(root))},
                    variables=program.get("run_variables", {}),
                )
            )
            write(root / "manifest.json", validate_manifest(result))
            print(program["name"], "whole policy", flush=True)
            continue
        contexts.sort(
            key=lambda row: hashlib.sha256(
                json.dumps(
                    {k: row[k] for k in ("decision", "features", "unit")},
                    sort_keys=True,
                ).encode()
            ).digest()
        )
        for ordinal, site in enumerate(contexts[: args.limit]):
            entry_id = f"{tag}-{ordinal}"
            entry = dict(
                id=entry_id,
                program=program["name"],
                group=program["group"],
                split=program["split"],
                source_sha256=source_sha256,
                context=context,
                decision=site["decision"],
                features=site["features"],
                baseline=str(baseline.relative_to(root)),
                candidates={},
                variables=program.get("run_variables", {}),
            )
            head = result["decisions"][site["decision"]]
            for action in range(1, len(head["contract"]["actions"])):
                candidate_dir = directory / f"{ordinal}-{action}"
                candidate_dir.mkdir(exist_ok=True)
                policy = candidate_dir / "policy.json"
                write(
                    policy,
                    dict(
                        version=2,
                        context_features=[],
                        requires={},
                        decisions={
                            site["decision"]: dict(
                                head,
                                advisor=dict(
                                    kind="probe",
                                    cases=[
                                        dict(features=site["features"], action=action)
                                    ],
                                ),
                            )
                        },
                    ),
                )
                obj = candidate_dir / "unit.o"
                run(
                    compile_command,
                    variables
                    | dict(
                        source=site["source"],
                        output=str(obj),
                        policy=str(policy),
                        trace=str(candidate_dir / "trace.jsonl"),
                    ),
                    candidate_dir / "compile.log",
                )
                executable = baseline
                if obj.read_bytes() != objects[site["unit"]].read_bytes():
                    changed = list(objects)
                    changed[site["unit"]] = obj
                    executable = candidate_dir / "run"
                    run(
                        link_command,
                        variables
                        | dict(objects=list(map(str, changed)), output=str(executable)),
                        candidate_dir / "link.log",
                    )
                entry["candidates"][str(action)] = str(executable.relative_to(root))
            result["experiments"].append(entry)
        write(root / "manifest.json", validate_manifest(result))
        print(
            program["name"],
            "contexts",
            min(len(contexts), args.limit or len(contexts)),
            flush=True,
        )
    result["complete"] = True
    write(root / "manifest.json", validate_manifest(result))


if __name__ == "__main__":
    main()
