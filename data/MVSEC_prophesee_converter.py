"""
Convert an MVSEC event stream (HDF5) to the Prophesee .dat format used by GeoIMO.

The MVSEC dataset distributes raw data as `*_data.hdf5` files containing many
sensor streams. The events live under `davis/<side>/events` as an (N, 4) array
with columns:

    [ x, y, timestamp_seconds, polarity ]

where `timestamp_seconds` is an absolute time in seconds (float) and `polarity`
is +1 / -1. This script extracts those events, converts them to the Prophesee
Event2D binary layout (8 bytes/event: uint32 timestamp in µs + a packed
x/y/polarity word), and writes a `.dat` file readable by the vendored
`third_party/prophesee` loader used throughout the pipeline.

Requires `h5py` and `numpy` (h5py is NOT part of the GeoIMO runtime container —
this is a one-time, offline preprocessing step):

    pip install h5py numpy

Usage:

    python data/MVSEC_prophesee_converter.py \
        --input_hdf5 /path/to/outdoor_day2_data.hdf5 \
        --output_dat data/outdoor_day2_events.dat
"""

import struct
from datetime import datetime, timezone

import h5py
import numpy as np


def read_mvsec_events(hdf5_path, camera="left"):
    """
    Load the DAVIS events from an MVSEC `*_data.hdf5` file into memory.

    Parameters:
        hdf5_path (str): path to the MVSEC HDF5 file
        camera (str): which DAVIS camera to read ('left' or 'right')

    Returns:
        np.ndarray of shape (N, 4) with columns [x, y, timestamp_s, polarity]
    """
    with h5py.File(hdf5_path, "r") as f:
        # Copy into memory so we can edit it; the file is opened read-only.
        events = f["davis"][camera]["events"][:]

    print("events.shape:", events.shape)
    duration_s = float(events[-1, 2]) - float(events[0, 2])
    print(f"Dataset duration: {duration_s:.3f} seconds")
    return events


def prepare_events(events):
    """
    Normalise events for the Prophesee format, operating on a copy:
        - timestamps shifted to start at 0 and converted to integer microseconds
        - polarity mapped to {0, 1}

    Returns a float64 array (still [x, y, t_us, p]); the final integer casts
    happen in `save_events_prophesee_format`.
    """
    events = events.astype(np.float64, copy=True)

    print("Normalising timestamps to start at 0 and converting to microseconds")
    events[:, 2] = np.round((events[:, 2] - events[0, 2]) * 1e6)

    print("Mapping polarity to {0, 1}")
    events[:, 3] = (events[:, 3] > 0).astype(np.float64)

    return events


def save_events_prophesee_format(events_array, output_filename, height=260, width=346):
    """
    Save events in Prophesee's Event2D .dat format.

    Parameters:
        events_array: (N, 4) array with columns [x, y, timestamp_us, polarity]
        output_filename: path to the output .dat file
        height, width: sensor resolution (MVSEC DAVIS346 -> 260 x 346)
    """
    if max(height, width) > 2 ** 14 - 1:
        raise ValueError(
            f"Resolution {width}x{height} exceeds the 14-bit coordinate range "
            "of the Prophesee .dat format."
        )

    with open(output_filename, "wb") as f:
        # ---- text header ----
        f.write(b"% Data file containing Event2D events.\n")
        f.write(b"% Version 2\n")
        now = datetime.now(timezone.utc)
        f.write(
            f"% Date {now.year}-{now.month}-{now.day} "
            f"{now.hour}:{now.minute}:{now.second}\n".encode()
        )
        f.write(f"% Height {height}\n".encode())
        f.write(f"% Width {width}\n".encode())

        # ---- event type (0 = Event2D) and size (8 bytes/event) ----
        f.write(struct.pack("BB", 0, 8))

        # ---- binary events: uint32 timestamp + packed x/y/polarity word ----
        # Layout (matches third_party/prophesee/io/dat_events_tools.py):
        #   bits  0-13 : x   (14 bits)
        #   bits 14-27 : y   (14 bits)
        #   bit     28 : polarity
        N = len(events_array)
        binary_events = np.zeros(N, dtype=[("t", "u4"), ("_", "i4")])
        binary_events["t"] = events_array[:, 2].astype(np.uint32)

        x = events_array[:, 0].astype(np.int32) & 0x3FFF
        y = (events_array[:, 1].astype(np.int32) & 0x3FFF) << 14
        p = (events_array[:, 3].astype(np.int32) & 0x1) << 28
        binary_events["_"] = x | y | p

        print("First 3 binary events:")
        print(binary_events[:3])

        binary_events.tofile(f)


def parse_args():
    """Parse input arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert an MVSEC HDF5 event stream to Prophesee .dat format."
    )
    parser.add_argument("--input_hdf5", type=str, required=True,
                        help="Input MVSEC *_data.hdf5 file path.")
    parser.add_argument("--output_dat", type=str, required=True,
                        help="Output .dat file path.")
    parser.add_argument("--camera", type=str, default="left",
                        choices=["left", "right"],
                        help="Which DAVIS camera to convert (default: left).")
    parser.add_argument("--height", type=int, default=260,
                        help="Sensor height (default: 260).")
    parser.add_argument("--width", type=int, default=346,
                        help="Sensor width (default: 346).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Reading MVSEC events from {args.input_hdf5} (camera: {args.camera})...")
    events = read_mvsec_events(args.input_hdf5, camera=args.camera)

    print("Preparing events...")
    prepared = prepare_events(events)
    print("Events prepared. Shape:", prepared.shape)

    print("Saving events in Prophesee format...")
    save_events_prophesee_format(prepared, args.output_dat,
                                 height=args.height, width=args.width)

    print(f"Events saved to {args.output_dat} in Prophesee format.")
