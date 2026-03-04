#!/usr/bin/env python3
"""Plot all 7 timesteps from the 30-min GPU run (no-advect config)."""
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

outdir = "/home/drew/wrf_gpu_clean_test"
files = sorted(glob.glob(f"{outdir}/wrfout_d01_*"))
print(f"Found {len(files)} files")

# Variables to plot
vars_2d = ["T2", "U10", "V10", "PSFC", "HFX"]

fig, axes = plt.subplots(len(vars_2d), len(files), figsize=(3.5*len(files), 3*len(vars_2d)))

for col, fpath in enumerate(files):
    ds = nc.Dataset(fpath)
    time_str = os.path.basename(fpath).replace("wrfout_d01_", "")

    for row, var in enumerate(vars_2d):
        ax = axes[row, col]
        data = ds.variables[var][0]  # first (only) time

        # Check for NaN
        nnan = np.isnan(data).sum()
        vmin, vmax = np.nanmin(data), np.nanmax(data)

        im = ax.pcolormesh(data, cmap='RdYlBu_r' if var == 'T2' else 'viridis')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if row == 0:
            ax.set_title(time_str[-8:], fontsize=9)
        if col == 0:
            ax.set_ylabel(var, fontsize=10, fontweight='bold')

        # Stats in corner
        ax.text(0.02, 0.98, f"{vmin:.1f}-{vmax:.1f}", transform=ax.transAxes,
                fontsize=6, va='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        if nnan > 0:
            ax.text(0.5, 0.5, f"NaN: {nnan}", transform=ax.transAxes,
                    fontsize=12, ha='center', color='red', fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
    ds.close()

plt.suptitle("WRF GPU 30-min Run (small_step+big_step+diffusion+module_em on GPU, advect on CPU)\n200x200x80, dx=3km, Marshall Fire",
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/mnt/c/Users/drew/aifs-90d/wrf_gpu/gpu_30min_no_advect.png", dpi=150)
print("Saved gpu_30min_no_advect.png")

# Also plot W at level 40 for first and last timestep
fig2, axes2 = plt.subplots(1, len(files), figsize=(3.5*len(files), 3.5))
for col, fpath in enumerate(files):
    ds = nc.Dataset(fpath)
    time_str = os.path.basename(fpath).replace("wrfout_d01_", "")
    w = ds.variables["W"][0, 40, :, :]
    ax = axes2[col]
    vabs = max(abs(np.nanmin(w)), abs(np.nanmax(w)), 5)
    im = ax.pcolormesh(w, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{time_str[-8:]}\nW[40] {np.nanmin(w):.1f} to {np.nanmax(w):.1f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ds.close()

plt.suptitle("Vertical Velocity W at level 40 — GPU 30-min Run", fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/mnt/c/Users/drew/aifs-90d/wrf_gpu/gpu_30min_W_evolution.png", dpi=150)
print("Saved gpu_30min_W_evolution.png")
