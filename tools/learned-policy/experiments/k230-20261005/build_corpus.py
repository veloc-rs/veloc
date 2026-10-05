#!/usr/bin/env python3
"""Compile counterfactual policy actions locally; execute/timestamp them on target."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MODES = {
    "baseline": None,
    "inline-keep": ("inline", 1),
    "inline-expand": ("inline", 2),
    "schedule-pressure": ("schedule", 1),
    "schedule-latency": ("schedule", 2),
    "schedule-source": ("schedule", 3),
    "schedule-fanout": ("schedule", 4),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--compiler", type=Path, default=ROOT / "target/release/veloc-c")
    p.add_argument("--clang", default="/opt/homebrew/opt/llvm/bin/clang")
    p.add_argument("--sysroot", type=Path, default=ROOT / "target/riscv64-sysroot")
    args = p.parse_args()
    out = args.out.resolve()
    common = [
        "--target=riscv64-linux-gnu",
        "--sysroot=" + str(args.sysroot),
        "-march=rv64gc_zba_zbb",
        "-mabi=lp64d",
    ]
    subprocess.run(
        [
            args.clang,
            *common,
            "-O2",
            "-c",
            str(Path(__file__).with_name("harness.c")),
            "-o",
            str(out / "harness.o"),
        ],
        check=True,
    )
    cases = json.loads((out / "corpus/manifest.json").read_text())
    (out / "policies").mkdir(exist_ok=True)
    for mode, decision in MODES.items():
        if decision:
            (out / "policies" / (mode + ".json")).write_text(
                json.dumps(
                    {
                        "version": 1,
                        "target": "riscv64/c908",
                        decision[0]: dict(kind="fixed", action=decision[1]),
                    }
                )
            )
    for case in cases:
        name = case["name"]
        source = out / "corpus" / (name + ".i")
        (out / "bin" / name).mkdir(parents=True, exist_ok=True)
        (out / "traces" / name).mkdir(parents=True, exist_ok=True)
        for mode in (*MODES, "llvm"):
            obj = out / "bin" / name / (mode + ".o")
            if mode == "llvm":
                command = [
                    args.clang,
                    *common,
                    "-O3",
                    "-c",
                    str(source),
                    "-o",
                    str(obj),
                ]
            else:
                command = [
                    str(args.compiler),
                    str(source),
                    "-O1",
                    "--cpu",
                    "c908",
                    "--verify-ir",
                    "--policy-trace",
                    str(out / "traces" / name / (mode + ".jsonl")),
                    "-o",
                    str(obj),
                ]
                if MODES[mode]:
                    command += ["--policy", str(out / "policies" / (mode + ".json"))]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(
                [
                    args.clang,
                    *common,
                    "-fuse-ld=lld",
                    "-no-pie",
                    str(obj),
                    str(out / "harness.o"),
                    "-o",
                    str(out / "bin" / name / mode),
                ],
                check=True,
            )
        print("compiled", name, flush=True)
    provenance = dict(
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_sha256={
            c["name"]: hashlib.sha256(
                (out / "corpus" / (c["name"] + ".i")).read_bytes()
            ).hexdigest()
            for c in cases
        },
        cases=cases,
        modes=list(MODES),
    )
    (out / "corpus-build.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
