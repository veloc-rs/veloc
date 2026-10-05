#!/usr/bin/env python3
"""Generate structural kernel families for policy training, without CoreMark."""
from pathlib import Path
import argparse
import json
import random

HEADER = "typedef unsigned int u32;\nextern u32 external_mix(u32);\n"


def program(family, variant):
    rng = random.Random(7301 + variant * 19 + family * 107)
    width = (2, 4, 6, 8, 12, 16)[variant % 6]
    constants = [rng.randrange(3, 65535) | 1 for _ in range(width)]
    if family == 0:  # Independent integer chains and memory parallelism.
        decl = ",".join("x%d=seed+%du" % (i, c) for i, c in enumerate(constants))
        steps = "\n".join(
            "x%d = (x%d * %du + a[(i+%d)&255u]) ^ (x%d >> %d);"
            % (j, j, c, j, j, 3 + j % 11)
            for j, c in enumerate(constants)
        )
        result = "^".join("x%d" % j for j in range(width))
        return (
            HEADER
            + "u32 kernel(u32*a,u32 n,u32 seed){u32 "
            + decl
            + ";for(u32 i=0;i<n;++i){"
            + steps
            + "}return "
            + result
            + ";}"
        )
    if family == 1:  # Recurrences mixed with independent address chains.
        steps = "\n".join(
            "x = (x + a[(i+%d)&255u])*%du; y ^= (x >> %d) + a[(i*%du)&255u];"
            % (j, c, 3 + j % 13, 2 * j + 1)
            for j, c in enumerate(constants)
        )
        return (
            HEADER
            + "u32 kernel(u32*a,u32 n,u32 seed){u32 x=seed,y=1;for(u32 i=0;i<n;++i){"
            + steps
            + "}return x^y;}"
        )
    if family == 2:  # Small reductions with varying numbers of live accumulators.
        decl = ",".join("x%d=seed" % i for i in range(width))
        steps = "\n".join(
            "x%d += a[(i+%d)&255u] * a[(i*%du+%d)&255u];" % (j, j, 2 * j + 1, j + 3)
            for j in range(width)
        )
        return (
            HEADER
            + "u32 kernel(u32*a,u32 n,u32 seed){u32 "
            + decl
            + ";for(u32 i=0;i<n;++i){"
            + steps
            + "}return "
            + "+".join("x%d" % j for j in range(width))
            + ";}"
        )
    if family == 3:  # Load/store interleaving with distinct and dependent addresses.
        steps = "\n".join(
            "x ^= a[(i+%d)&255u]; a[(i+%d)&255u] = x * %du + seed;" % (j, 64 + j, c)
            for j, c in enumerate(constants)
        )
        return (
            HEADER
            + "u32 kernel(u32*a,u32 n,u32 seed){u32 x=seed;for(u32 i=0;i<n;++i){"
            + steps
            + "}return x+a[127];}"
        )
    if family == 4:  # Inlining a branchy scalar helper exposes constant inputs.
        steps = "\n".join(
            "if ((x & %du) != 0) x = x*%du + mode; else x = (x>>%d)^%du;"
            % (1 << (j % 16), c, 1 + j % 7, c)
            for j, c in enumerate(constants)
        )
        return (
            HEADER
            + "static u32 helper(u32 x,u32 mode){"
            + steps
            + "return x;} u32 kernel(u32*a,u32 n,u32 seed){u32 x=seed;for(u32 i=0;i<n;++i){x=helper(x+a[i&255u],%du);}return x;}"
            % (variant + 1)
        )
    if family == 5:  # Larger leaf helper, called with caller values still live.
        steps = "\n".join(
            "x = (x * %du) ^ (x >> %d);" % (c, 1 + j % 15)
            for j, c in enumerate(constants)
        )
        return (
            HEADER
            + "static u32 helper(u32 x){"
            + steps
            + "return x;} u32 kernel(u32*a,u32 n,u32 seed){u32 x=seed,y=3,z=7;for(u32 i=0;i<n;++i){y=y*33u+a[i&255u];z=(z<<3)^a[(i+1)&255u];x+=helper(y^z)^z;}return x^y^z;}"
        )
    if family == 6:  # Retained external calls stress ABI clobbers after inlining.
        steps = "\n".join(
            "x = (x ^ a[%d]) * %du;" % (j, c) for j, c in enumerate(constants)
        )
        return (
            HEADER
            + "static u32 helper(u32*a,u32 x){"
            + steps
            + "return external_mix(x);} u32 kernel(u32*a,u32 n,u32 seed){u32 x=seed,y=3;for(u32 i=0;i<n;++i){y=y*33u+a[i&255u];x=helper(a,x)^y;}return x+y;}"
        )
    if family == 7:  # Nested loop in a helper; trip counts become constants.
        return (
            HEADER
            + "static u32 helper(u32*a,u32 n,u32 x){for(u32 j=0;j<n;++j){x=(x+a[j&255u])*33u; x^=x>>11;}return x;} u32 kernel(u32*a,u32 n,u32 seed){u32 x=seed;for(u32 i=0;i<n;++i){x=helper(a,%du,x)^a[i&255u];}return x;}"
            % width
        )
    if family == 8:  # Held-out family: unsigned data-dependent selection.
        steps = "\n".join(
            "x = x < a[(i+%d)&255u] ? x*%du + y : (x>>3)^y; y+=x^%du;" % (j, c, c)
            for j, c in enumerate(constants)
        )
        return (
            HEADER
            + "u32 kernel(u32*a,u32 n,u32 seed){u32 x=seed,y=1;for(u32 i=0;i<n;++i){"
            + steps
            + "}return x^y;}"
        )
    # Held-out family: matrix-like blocked reduction.
    return (
        HEADER
        + "u32 kernel(u32*a,u32 n,u32 seed){u32 x=seed;for(u32 i=0;i<n;++i){for(u32 j=0;j<%du;++j){x+=a[(i+j)&255u]*a[(i*3u+j*7u)&255u];}}return x;}"
        % width
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for family in range(10):
        for variant in range(6):
            name = "f%d-v%d" % (family, variant)
            path = args.out / (name + ".i")
            path.write_text(program(family, variant) + "\n")
            manifest.append(
                dict(
                    name=name,
                    family=family,
                    variant=variant,
                    split=(
                        "test"
                        if family >= 8
                        else ("validation" if variant == 5 else "train")
                    ),
                )
            )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
