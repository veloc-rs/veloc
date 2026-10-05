#!/usr/bin/env python3
"""Measure the causal effect of changing one structural decision context.

CoreMark probes are tuning data, explicitly separated from independent kernels.
The exported neural model never receives a program name or decision index.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
UNITS = ("main", "list_join", "matrix", "state", "util")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--coremark", action="store_true")
    parser.add_argument(
        "--compiler", type=Path, default=ROOT / "target/release/veloc-c"
    )
    parser.add_argument("--clang", default="/opt/homebrew/opt/llvm/bin/clang")
    parser.add_argument("--sysroot", type=Path, default=ROOT / "target/riscv64-sysroot")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / ("coremark-probes" if args.coremark else "probes")
    out.mkdir(exist_ok=True)
    compiler = args.compiler
    clang = args.clang
    link = [
        clang,
        "--target=riscv64-linux-gnu",
        "--sysroot=" + str(args.sysroot),
        "-march=rv64gc_zba_zbb",
        "-mabi=lp64d",
        "-fuse-ld=lld",
        "-no-pie",
    ]
    cases = (
        [dict(name=u, split="tuning") for u in UNITS]
        if args.coremark
        else json.loads((root / "corpus/manifest.json").read_text())
    )
    tasks = []
    for case in cases:
        name = case["name"]
        trace = (
            root / "coremark/baseline" / (name + ".jsonl")
            if args.coremark
            else root / "traces" / name / "baseline.jsonl"
        )
        seen = set()
        rows = [json.loads(line) for line in trace.read_text().splitlines()[1:]]
        for row in rows:
            decision, features = row["decision"], row["features"]
            key = (decision, tuple(features))
            if key in seen:
                continue
            seen.add(key)
            if decision == "schedule" and (features[0] < 5 or features[15] == 0):
                continue
            for action in range(1, 3 if decision == "inline" else 5):
                tasks.append(
                    dict(
                        name=name,
                        split=case["split"],
                        decision=decision,
                        features=features,
                        action=action,
                    )
                )

    def build(item):
        index, row = item
        tag = "%04d" % index
        directory = out / tag
        directory.mkdir(exist_ok=True)
        policy = directory / "policy.json"
        policy.write_text(
            json.dumps(
                dict(
                    version=1,
                    target="riscv64/c908",
                    **{
                        row["decision"]: dict(
                            kind="probe",
                            cases=[
                                dict(features=row["features"], action=row["action"])
                            ],
                        )
                    }
                )
            )
        )
        name = row["name"]
        source = (
            root / "coremark/input" / (name + ".i")
            if args.coremark
            else root / "corpus" / (name + ".i")
        )
        baseline = (
            root / "coremark/baseline" / (name + ".o")
            if args.coremark
            else root / "bin" / name / "baseline.o"
        )
        obj = directory / "changed.o"
        subprocess.run(
            [
                str(compiler),
                str(source),
                "-O1",
                "--cpu",
                "c908",
                "--verify-ir",
                "--policy",
                str(policy),
                "-o",
                str(obj),
            ],
            check=True,
        )
        changed = obj.read_bytes() != baseline.read_bytes()
        digest = None
        if changed:
            if args.coremark:
                objects = [
                    str(obj if u == name else root / "coremark/baseline" / (u + ".o"))
                    for u in UNITS
                ] + [str(root / "coremark/port.o")]
            else:
                objects = [str(obj), str(root / "harness.o")]
            executable = directory / "run"
            subprocess.run([*link, *objects, "-o", str(executable)], check=True)
            digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        return dict(row, id=tag, changed=changed, sha256=digest)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(build, enumerate(tasks)))
    # Identical machine code needs only one measurement, including across contexts.
    canonical = {}
    for row in results:
        if row["changed"]:
            key = ("coremark" if args.coremark else row["name"], row["sha256"])
            row["measurement"] = canonical.setdefault(key, row["id"])
    (out / "manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    with tarfile.open(root / (out.name + ".tar.gz"), "w:gz") as archive:
        archive.add(out / "manifest.json", arcname=out.name + "/manifest.json")
        for tag in canonical.values():
            archive.add(out / tag / "run", arcname=out.name + "/" + tag + "/run")
        archive.add(
            Path(__file__).with_name("measure_probes.py"), arcname="measure_probes.py"
        )
    print(
        out.name,
        len(results),
        "interventions;",
        len(canonical),
        "distinct binaries",
        flush=True,
    )


if __name__ == "__main__":
    main()
