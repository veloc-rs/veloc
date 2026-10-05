#!/usr/bin/env python3
"""Larger CFGs, call chains, nested loops and integer state machines.

These independent kernels cover contexts absent from the simple loop corpus.
All arithmetic is unsigned; array indices are explicitly bounded.
"""
import argparse
import json
from pathlib import Path
import random

HEADER = "typedef unsigned int u32;\nextern u32 external_mix(u32);\n"


def program(family, variant):
    rng = random.Random(52634 + family * 157 + variant * 13)
    width = (2, 4, 6, 8, 12, 16)[variant]
    constants = [rng.randrange(3, 65535) | 1 for _ in range(width)]
    if family == 10:
        helper = """static u32 helper(u32*a,u32 x,u32 mode){
          for(u32 j=0;j<8;++j){
            if(x&1u) x=(x>>1)^0x82f63b78u; else x>>=1;
          }return x+a[mode&255u];} """
    elif family == 11:
        helper = """static u32 helper(u32*a,u32 x,u32 mode){
          for(u32 j=0;j<4;++j){
            switch(x&7u){
              case 0:x+=a[(j+mode)&255u];break;
              case 1:x^=a[(j*3u+mode)&255u];break;
              case 2:x=x*33u+17u;break;
              case 3:x=(x<<7)|(x>>25);break;
              case 4:x=x*7u+(x>>16);break;
              default:x+=0x9e3779b9u;
            }
          }return x;} """
    elif family == 12:
        steps = "".join(
            "if(x&%du)x=(x>>3)^%du;else x=x*%du+mode;" % (1 << (j % 16), c, c)
            for j, c in enumerate(constants)
        )
        helper = (
            "static u32 helper(u32*a,u32 x,u32 mode){"
            + steps
            + "return external_mix(x);}\n"
        )
    elif family == 13:
        helper = """static u32 helper(u32*a,u32 x,u32 mode){
          u32 y=mode;
          for(u32 j=0;j<8;++j){
            y=a[(x+j)&255u];
            if(y<32u) x+=y;
            else if(y<256u) x^=y*33u;
            else if(y&1u) x=(x>>1)^y;
            else x+=y>>3;
          }return x;} """
    elif family == 14:
        steps = "".join(
            "x=(x+a[%d])*%du; y=(y^(x>>%d))+mode;" % (j, c, j % 13 + 1)
            for j, c in enumerate(constants)
        )
        helper = (
            "static u32 helper(u32*a,u32 x,u32 mode){u32 y=x;"
            + steps
            + "return x^y;}\n"
        )
    elif family == 15:
        helper = """static u32 helper(u32*a,u32 x,u32 mode){
          for(u32 j=0;j<4;++j){
            u32 y=a[(x+j)&255u];
            for(u32 k=0;k<4;++k){x=(x+y)*33u; x^=x>>11;}
            x^=mode;
          }return x;} """
    elif family == 16:  # Held-out: a data-dependent pointer-index chase.
        helper = """static u32 helper(u32*a,u32 x,u32 mode){
          for(u32 j=0;j<16;++j){
            u32 y=a[(x+mode)&255u];x=(y>>16)^(y*0x85ebca6bu);
          }return x;} """
    else:  # Held-out: bounded insertion into a rolling ordered window.
        helper = """static u32 helper(u32*a,u32 x,u32 mode){
          u32 pos=mode&127u;
          for(u32 j=0;j<8;++j){
            u32 y=a[(pos+j)&255u];
            if(y>x){a[(pos+j)&255u]=x;x=y;}
          }return x;} """
    body = "u32 x=seed,y=3,z=7;for(u32 i=0;i<n;++i){"
    for j, c in enumerate(constants):
        body += "y=(y+a[(i+%d)&255u])*%du;" % (j, c)
        body += "if((y&7u)==%du){z^=helper(a,x+y,i+%du);x+=z;}else{x=(x>>3)^y;}" % (
            j % 8,
            j,
        )
        if j % 3 == 0:
            body += "for(u32 k=0;k<2;++k){x=helper(a,x^z,k)^y;}"
    body += "x=helper(a,x,i);y^=x;}return x^y^z;"
    return HEADER + helper + "u32 kernel(u32*a,u32 n,u32 seed){" + body + "}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for family in range(10, 18):
        for variant in range(6):
            name = "f%d-v%d" % (family, variant)
            (args.out / (name + ".i")).write_text(program(family, variant))
            split = (
                "test" if family >= 16 else "validation" if variant == 5 else "train"
            )
            manifest.append(
                dict(name=name, family=family, variant=variant, split=split)
            )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
