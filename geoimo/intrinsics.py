import numpy as np
from dataclasses import dataclass


@dataclass
class Intrinsics:
    fx: float
    fy: float
    f_avg: float
    dataset_name: str


def resolve_dataset_mode(dataset_arg: str, width: int, height: int) -> str:
    if dataset_arg == "mvsec":
        return "mvsec"
    if dataset_arg == "prophesee":
        return "prophesee"
    if int(width) == 346 and int(height) == 260:
        return "mvsec"
    return "prophesee"


def compute_intrinsics(dataset: str, width: int, height: int) -> Intrinsics:
    if dataset == "mvsec":
        fx = 346.0 / 2.0 / np.tan(np.radians(65.0 / 2.0))
        fy = 260.0 / 2.0 / np.tan(np.radians(50.0 / 2.0))
        return Intrinsics(fx=fx, fy=fy, f_avg=(fx + fy) / 2.0, dataset_name="MVSEC")

    # Prophesee: assume HFOV=110°
    hfov_deg = 110.0
    fx = (float(width) / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    vfov_rad = 2.0 * np.arctan(
        (float(height) / float(width)) * np.tan(np.radians(hfov_deg / 2.0))
    )
    fy = (float(height) / 2.0) / np.tan(vfov_rad / 2.0)
    return Intrinsics(fx=fx, fy=fy, f_avg=(fx + fy) / 2.0, dataset_name="Prophesee")
