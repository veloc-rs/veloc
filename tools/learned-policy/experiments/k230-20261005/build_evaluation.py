#!/usr/bin/env python3
"""Build learned-policy ablations and the untouched CoreMark evaluation input."""
from pathlib import Path
import argparse
import collections
import json
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--coremark-input", type=Path, required=True)
    p.add_argument("--port", type=Path, required=True)
    p.add_argument("--compiler", type=Path, default=ROOT / "target/release/veloc-c")
    p.add_argument("--clang", default="/opt/homebrew/opt/llvm/bin/clang")
    p.add_argument("--sysroot", type=Path, default=ROOT / "target/riscv64-sysroot")
    args = p.parse_args()
    out = args.out.resolve()
    model = args.model or out / "learned.json"
    compiler = args.compiler
    clang = args.clang
    common = [
        "--target=riscv64-linux-gnu",
        "--sysroot=" + str(args.sysroot),
        "-march=rv64gc_zba_zbb",
        "-mabi=lp64d",
    ]
    document = json.loads(model.read_text())
    (out / "policies" / "learned.json").write_text(json.dumps(document) + "\n")
    for decision in ("inline", "schedule"):
        (out / "policies" / ("learned-" + decision + ".json")).write_text(
            json.dumps(
                {
                    k: v
                    for k, v in document.items()
                    if k in ("version", "target", decision)
                }
            )
        )

    def compile_one(source, obj, policy, trace):
        command = [
            str(compiler),
            str(source),
            "-O1",
            "--cpu",
            "c908",
            "--verify-ir",
            "--policy-trace",
            str(trace),
            "-o",
            str(obj),
        ]
        if policy:
            command += ["--policy", str(policy)]
        subprocess.run(command, check=True)

    for case in json.loads((out / "corpus/manifest.json").read_text()):
        name = case["name"]
        obj = out / "bin" / name / "learned.o"
        compile_one(
            out / "corpus" / (name + ".i"),
            obj,
            model,
            out / "traces" / name / "learned.jsonl",
        )
        subprocess.run(
            [
                clang,
                *common,
                "-fuse-ld=lld",
                "-no-pie",
                str(obj),
                str(out / "harness.o"),
                "-o",
                str(out / "bin" / name / "learned"),
            ],
            check=True,
        )
    cm = out / "coremark"
    (cm / "input").mkdir(parents=True, exist_ok=True)
    for source in args.coremark_input.glob("*.i"):
        destination = cm / "input" / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    if args.port.resolve() != (cm / "port.o").resolve():
        shutil.copy2(args.port, cm / "port.o")
    policies = {
        "baseline": None,
        "learned": model,
        **{
            m: out / "policies" / (m + ".json")
            for m in (
                "learned-inline",
                "learned-schedule",
                "inline-expand",
                "schedule-pressure",
                "schedule-latency",
                "schedule-source",
                "schedule-fanout",
            )
        },
    }
    actions = {}
    for mode, policy in policies.items():
        (cm / mode).mkdir(exist_ok=True)
        objs = []
        counts = collections.Counter()
        for unit in ("main", "list_join", "matrix", "state", "util"):
            obj = cm / mode / (unit + ".o")
            trace = cm / mode / (unit + ".jsonl")
            objs.append(str(obj))
            compile_one(cm / "input" / (unit + ".i"), obj, policy, trace)
            for line in trace.read_text().splitlines()[1:]:
                row = json.loads(line)
                counts[(row["decision"], row["action"])] += 1
        subprocess.run(
            [
                clang,
                *common,
                "-fuse-ld=lld",
                "-no-pie",
                *objs,
                str(cm / "port.o"),
                "-o",
                str(cm / ("coremark-" + mode)),
            ],
            check=True,
        )
        actions[mode] = {str(k): v for k, v in counts.items()}
        print(mode, actions[mode], flush=True)
    (cm / "actions.json").write_text(json.dumps(actions, indent=2) + "\n")
    with tarfile.open(out / "evaluation-target.tar.gz", "w:gz") as tar:
        for file in (out / "bin").glob("*/learned"):
            tar.add(file, arcname=str(file.relative_to(out)))
        for file in cm.glob("coremark-*"):
            tar.add(file, arcname=str(file.relative_to(out)))
        for file in (out / "policies").glob("*.json"):
            tar.add(file, arcname=str(file.relative_to(out)))
        for file in (cm / "input").glob("*.i"):
            tar.add(file, arcname=str(file.relative_to(out)))
        for mode in policies:
            for file in (cm / mode).glob("*.o"):
                tar.add(file, arcname=str(file.relative_to(out)))
        tar.add(out / "corpus-times.json", arcname="corpus-times.json")
        tar.add(
            Path(__file__).with_name("measure_evaluation.py"),
            arcname="measure_evaluation.py",
        )
        tar.add(
            Path(__file__).with_name("measure_final.py"), arcname="measure_final.py"
        )


if __name__ == "__main__":
    main()
