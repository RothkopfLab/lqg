#!/usr/bin/env python3
"""Convert annotated tracking data to the public ``data.mat`` layout.

This script reads ``*_tracking_MasterData.mat`` files from
``../AnnotatedDataRothkopf/tracking`` and writes files containing ``target``,
``response``, and ``sigma`` arrays in the format consumed by
``lqg.io.load_tracking_data``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.io as spio

# sigma labels in annotated tracking files -> labels expected by data/data.mat
TRACKING_TO_DATA_MAT_SIGMA = {
    8: 11,
    10: 13,
    13: 17,
    16: 21,
    19: 25,
    22: 29,
}


def _map_sigma_labels(tracking_sigma: np.ndarray) -> np.ndarray:
    keys = np.array(list(TRACKING_TO_DATA_MAT_SIGMA.keys()))
    if not np.all(np.isin(tracking_sigma, keys)):
        bad = np.setdiff1d(np.unique(tracking_sigma), keys)
        raise ValueError(f"Unexpected sigma values in tracking file: {bad}")
    out = np.empty(tracking_sigma.shape, dtype=np.uint8)
    for src, dst in TRACKING_TO_DATA_MAT_SIGMA.items():
        out[tracking_sigma == src] = dst
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.normpath(
        os.path.join(here, "../AnnotatedDataRothkopf/tracking")
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "initials",
        help="Participant initials, e.g. JDB or KLB (reads {initials}_tracking_MasterData.mat)",
    )
    parser.add_argument(
        "--input-dir",
        default=default_input_dir,
        help="Directory containing *_tracking_MasterData.mat",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output .mat path (default: data/data_{initials}.mat)",
    )
    args = parser.parse_args()

    initials = args.initials.strip().upper()
    in_path = os.path.join(args.input_dir, f"{initials}_tracking_MasterData.mat")
    if not os.path.isfile(in_path):
        print(f"ERROR: missing input: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or os.path.join(here, "data", f"data_{initials}.mat")

    raw = spio.loadmat(in_path, struct_as_record=False, squeeze_me=True)
    tracking_data = raw["trkData"]
    target = np.asarray(tracking_data.hTargCoords, dtype=np.float64)
    response = np.asarray(tracking_data.hRespCoords, dtype=np.uint16)
    sigma_tracking = np.asarray(tracking_data.sigma)
    sigma = _map_sigma_labels(sigma_tracking)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    spio.savemat(
        out_path,
        {"target": target, "response": response, "sigma": sigma},
        do_compression=True,
    )
    print(f"Wrote {out_path}  trials={target.shape[0]}  time={target.shape[1]}")


if __name__ == "__main__":
    main()
