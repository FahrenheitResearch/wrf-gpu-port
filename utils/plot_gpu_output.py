#!/usr/bin/env python3
"""Plot GPU WRF output fields."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset

gpu_dir = "/home/drew/wrf_gpu_clean_test"
time = "2021-12-30_17:02:00"
ds = Dataset(gpu_dir + "/wrfout_d01_" + time, "r")

lat = np.array(ds.variables["XLAT"][0])
lon = np.array(ds.variables["XLONG"][0])

fields = [
    ("T2", "2m Temperature (K)", "RdYlBu_r", None),
    ("U10", "10m U-Wind (m/s)", "RdBu_r", None),
    ("V10", "10m V-Wind (m/s)", "RdBu_r", None),
    ("PSFC", "Surface Pressure (Pa)", "viridis", None),
    ("HFX", "Sensible Heat Flux (W/m2)", "RdBu_r", None),
    ("W", "Vertical Velocity (m/s) lev40", "RdBu_r", 40),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("WRF GPU Output (RTX 5090) -- " + time + "\nMarshall Fire 3km, 200x200x80",
             fontsize=14, fontweight="bold")

for ax, (field, label, cmap, lev) in zip(axes.flat, fields):
    data = np.array(ds.variables[field][0])
    if lev is not None:
        data = data[lev]
    im = ax.pcolormesh(lon, lat, data, cmap=cmap, shading="auto")
    ax.set_title(field + ": " + label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
outpath = "/home/drew/wrf_gpu_output.png"
plt.savefig(outpath, dpi=150)
print("Saved: " + outpath)

# Also print stats
print("\nField stats:")
for field, label, _, lev in fields:
    data = np.array(ds.variables[field][0])
    if lev is not None:
        data = data[lev]
    print("  %s: min=%.4f max=%.4f mean=%.4f NaN=%s" % (
        field, np.nanmin(data), np.nanmax(data), np.nanmean(data), np.any(np.isnan(data))))

ds.close()
