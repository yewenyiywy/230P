"""Command line: `newsalpha run --config configs/synthetic.yaml`."""

from __future__ import annotations

import argparse
import json
import logging

from newsalpha.config import load_config
from newsalpha.pipeline import distill_data, distill_eval, run, run_extraction


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="newsalpha")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline for one config")
    r.add_argument("--config", required=True)
    r.add_argument("--out", default=None, help="runs directory (default <data.root>/runs)")
    e = sub.add_parser(
        "extract", help="stages 1-4 only: load, clean, align, extract (warms the cache)"
    )
    e.add_argument("--config", required=True)
    dd = sub.add_parser("distill-data", help="T6: teacher-label a training-period sample -> JSONL")
    dd.add_argument("--config", required=True)
    de = sub.add_parser("distill-eval", help="T6: student (extraction) vs teacher on the val split")
    de.add_argument("--config", required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "run":
        report = run(load_config(args.config), args.out)
        print(
            json.dumps(
                {k: report[k] for k in ("selected", "headline_test", "run_dir")},
                indent=2,
                default=str,
            )
        )
    elif args.cmd == "extract":
        _, _, events, extractions, signals = run_extraction(load_config(args.config))
        print(
            f"{len(events)} events, {len(extractions)} articles extracted, signals {list(signals.columns)}"
        )
    elif args.cmd == "distill-data":
        print(json.dumps(distill_data(load_config(args.config)), indent=2))
    elif args.cmd == "distill-eval":
        print(distill_eval(load_config(args.config)).to_string(index=False))


if __name__ == "__main__":
    main()
