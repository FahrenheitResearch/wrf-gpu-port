#!/usr/bin/env python3
"""Compare frame 0 vs frame 1 to diagnose GPU data sync issues."""
from netCDF4 import Dataset
import numpy as np

nc0 = Dataset("/home/drew/wrf_gpu_test/wrfout_d01_2021-12-30_17:00:00", "r")
nc1 = Dataset("/home/drew/wrf_gpu_test/wrfout_d01_2021-12-30_17:01:00", "r")

print(f"{'Field':15s} | {'F0 valid':>8s} | {'F1 valid':>8s} | {'Changed':>7s} | Notes")
print("-" * 80)
for v in sorted(nc0.variables):
    d0 = nc0.variables[v][0]
    d1 = nc1.variables[v][0]
    if len(d0.shape) > 2 or d0.size < 10:
        continue
    nan0 = np.isnan(d0).sum()
    nan1 = np.isnan(d1).sum()
    valid0 = d0.size - nan0
    valid1 = d1.size - nan1
    mask = ~np.isnan(d0) & ~np.isnan(d1)
    if mask.sum() > 0:
        same = np.allclose(d0[mask], d1[mask], rtol=1e-5)
    else:
        same = True
    if nan1 == d1.size:
        status = "ALL NaN in F1!"
    elif nan1 > 0:
        status = f"{nan1} NaN in F1"
    elif same:
        status = "UNCHANGED - stale CPU copy?"
    else:
        status = "evolved OK"
    print(f"{v:15s} | {valid0:8d} | {valid1:8d} | {'same' if same else 'diff':>7s} | {status}")

# Also check a few 3D fields
print("\n3D Fields:")
for v in ["T", "U", "V", "W", "PH", "P", "MU_2"]:
    if v not in nc1.variables:
        continue
    d1 = nc1.variables[v][0]
    nan1 = np.isnan(d1).sum()
    print(f"  {v:10s}: nan={nan1}/{d1.size} ({100*nan1/d1.size:.1f}%)  range=[{np.nanmin(d1):.4g}, {np.nanmax(d1):.4g}]")

nc0.close()
nc1.close()
