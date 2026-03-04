#!/usr/bin/env python3
"""Compare CPU vs GPU WRF output to verify GPU correctness."""
from netCDF4 import Dataset
import numpy as np

pairs = [
    ("/home/drew/wrf_cpu_verify/wrfout_d01_2021-12-30_17:01:00", "CPU F1"),
    ("/home/drew/wrf_gpu_test/wrfout_d01_2021-12-30_17:01:00", "GPU F1"),
]

for fname, label in pairs:
    nc = Dataset(fname, "r")
    src = fname.split("/")[-2]
    print(f"\n=== {label} ({src}) ===")
    for v in ["T2", "U10", "V10", "PSFC", "HFX", "LH", "TSK", "PBLH",
              "T", "U", "V", "W", "PH", "P", "MU_2", "RAINNC"]:
        if v not in nc.variables:
            continue
        d = np.array(nc.variables[v][0], dtype=float)
        nans = np.sum(np.isnan(d))
        print(f"  {v:8s}: nan={nans:8d}/{d.size:8d}  range=[{np.nanmin(d):12.4g}, {np.nanmax(d):12.4g}]")
    nc.close()
