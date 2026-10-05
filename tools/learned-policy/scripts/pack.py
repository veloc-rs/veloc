#!/usr/bin/env python3
"""Export the exact f32 model as MessagePack; this does not retrain it."""
import argparse
import json
from pathlib import Path
import msgpack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    document = json.loads(args.source.read_text())
    if document["version"] != 2:
        parser.error("expected policy format 2")
    for head in document["decisions"].values():
        if head["advisor"].pop("kind") != "network":
            parser.error(
                "deployment files require trained networks; use JSON for controls"
            )
    payload = msgpack.packb(document, use_bin_type=True, use_single_float=True)
    args.output.write_bytes(b"VLPM\x02" + payload)


if __name__ == "__main__":
    main()
