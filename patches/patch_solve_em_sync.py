#!/usr/bin/env python3
"""
patch_solve_em_sync.py — Add !$acc update device() directives in solve_em
to sync CPU-computed grid arrays to GPU before GPU kernels reference them.

Key data flow in WRF solve_em:
  1. rk_step_prep computes grid%ru, grid%rv, grid%rw, grid%ww, grid%alt, grid%php
     on CPU from grid%u_2, grid%v_2, etc.
  2. rk_tendency calls coriolis, curvature, advection which have
     !$acc parallel loop present(ru, rv, ...) — requires data on GPU
  3. Without sync, GPU has stale data from init copyin

This script adds !$acc update device() after rk_step_prep so the
CPU-computed values are available on GPU.

Run AFTER patch_solve_em_gpu.py.
"""

import os
import sys
import re
from pathlib import Path

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

SOLVE_EM = Path(WRF_DIR) / "dyn_em" / "solve_em.f90"

if not SOLVE_EM.exists():
    print(f"ERROR: {SOLVE_EM} not found")
    sys.exit(1)

text = SOLVE_EM.read_text()

# Idempotency check
if "!$acc update device(grid%ru" in text:
    print("solve_em.f90 already has update device directives — skipping.")
    sys.exit(0)

# Find all rk_step_prep calls and insert update after the OMP END DO that follows
# Pattern: CALL rk_step_prep(...) followed by END DO / !$OMP END PARALLEL DO
lines = text.split("\n")
insertions = []

# Arrays computed by rk_step_prep that are used in present() clauses:
# grid%ru, grid%rv, grid%rw, grid%ww, grid%php, grid%alt, grid%mut
# Also cqu, cqv, cqw are solve_em locals (already in enter data create)
update_block = (
    "   ! --- Sync CPU-computed arrays to GPU ---\n"
    "   !$acc update device(grid%ru, grid%rv, grid%rw, grid%ww, &\n"
    "   !$acc&  grid%php, grid%alt, grid%mut, grid%muu, grid%muv, &\n"
    "   !$acc&  grid%muts, cqu, cqv, cqw)"
)

# Also need to sync back GPU-computed tendencies to CPU for physics
# After the RK tendency computation, before physics coupling
update_host_block = (
    "   ! --- Sync GPU-computed tendencies back to CPU ---\n"
    "   !$acc update self(grid%ru_tend, grid%rv_tend)"
)

# Find rk_step_prep calls
in_omp_region = False
rk_step_prep_found = False
insertion_points = []

for i, line in enumerate(lines):
    stripped = line.strip()

    # Find CALL rk_step_prep
    if "CALL rk_step_prep" in line:
        rk_step_prep_found = True
        continue

    # After rk_step_prep, find the END DO / !$OMP END PARALLEL DO block
    if rk_step_prep_found and "!$OMP END PARALLEL DO" in line:
        insertion_points.append(i + 1)
        rk_step_prep_found = False

if not insertion_points:
    print("ERROR: Could not find rk_step_prep + OMP END PARALLEL DO pattern")
    sys.exit(1)

# Insert update directives after each rk_step_prep block
# Process in reverse so line numbers don't shift
new_lines = list(lines)
for point in reversed(insertion_points):
    new_lines.insert(point, "")
    new_lines.insert(point + 1, update_block)

SOLVE_EM.write_text("\n".join(new_lines))
print(f"Added {len(insertion_points)} !$acc update device() blocks after rk_step_prep in solve_em")
print(f"  Syncing: grid%ru, grid%rv, grid%rw, grid%ww, grid%php, grid%alt, grid%mut, grid%muu, grid%muv, grid%muts, cqu, cqv, cqw")


if __name__ == "__main__":
    main() if "main" in dir() else None
