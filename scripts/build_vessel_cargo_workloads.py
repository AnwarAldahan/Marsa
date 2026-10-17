"""Build a separate, deterministic vessel workload proxy; never modify AIS.

Usage: python scripts/build_vessel_cargo_workloads.py [--source PATH] [--output PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marsa.twin.calls import CALLS_PATH  # noqa: E402
from marsa.twin.workloads import WORKLOADS_PATH, write_workloads  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=CALLS_PATH)
    parser.add_argument("--output", type=Path, default=WORKLOADS_PATH)
    args = parser.parse_args()
    result = write_workloads(args.source, args.output)
    print(f"Wrote {result['records']} estimated per-call workloads to {args.output}")


if __name__ == "__main__":
    main()
