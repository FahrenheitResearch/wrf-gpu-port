#!/usr/bin/env python3
"""Fix GPU data initialization timing issue in solve_em.f90.

ROOT CAUSE: gpu_init_domain_data copies grid arrays to GPU at startup,
but derived quantities (mut, muts, muu, muv, alt, php, etc.) are computed
AFTER gpu_init runs. The GPU has zeros for these arrays.

Then the first !$acc update host() copies stale GPU zeros back to host,
destroying correct host values.

FIX: Add !$acc update device() for derived quantities at the start of
the RK loop, BEFORE any GPU->CPU sync. This pushes correct host values
to GPU, so subsequent GPU->CPU syncs are no-ops (same values).

Targets .f90 (post-preprocessor), consistent with all other patches.
"""
import os, sys

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

filepath = os.path.join(WRF_DIR, "dyn_em", "solve_em.f90")

with open(filepath, 'r') as f:
    content = f.read()

# Idempotency check
if "FIX: Push derived quantities to GPU" in content:
    print("SKIP: GPU init timing fix already applied to solve_em.f90")
    sys.exit(0)

# Find the first !$acc update host in solve_em (the GPU->CPU sync before physics)
# Insert a device update BEFORE it to fix derived quantities
# In the .f90, there's no #ifdef — the ACC directives are already present
# if compiled with -DGPU_OPENACC

# Look for the first !$acc update host that syncs state arrays
marker = "!$acc update host(grid%t_1, grid%t_2, grid%u_1, grid%u_2)"

if marker not in content:
    # Try alternate marker — the .f90 may have different formatting
    # or the directive may not exist yet (added by patch_solve_em_gpu.py later)
    print("NOTE: Could not find GPU->CPU sync marker in solve_em.f90")
    print("  This fix will be applied by patch_solve_em_gpu.py's sync points instead.")
    sys.exit(0)

fix = """! FIX: Push derived quantities to GPU (computed after gpu_init_domain_data)
! Without this, GPU has zeros for mut, muts, muu, muv, alt, php, etc.
!$acc update device(grid%mut, grid%muts, grid%mudf) &
!$acc   device(grid%muu, grid%muus, grid%muv, grid%muvs) &
!$acc   device(grid%alt, grid%php, grid%p, grid%pb, grid%al, grid%alb) &
!$acc   device(grid%ph_1, grid%ph_2, grid%phb) &
!$acc   device(grid%t_1, grid%t_2, grid%t_save) &
!$acc   device(grid%u_1, grid%u_2, grid%u_save) &
!$acc   device(grid%v_1, grid%v_2, grid%v_save) &
!$acc   device(grid%w_1, grid%w_2) &
!$acc   device(grid%mu_1, grid%mu_2, grid%mub) &
!$acc   device(grid%z, grid%z_at_w, grid%p_hyd, grid%p_hyd_w, grid%rho)

""" + marker

content = content.replace(marker, fix, 1)

with open(filepath, 'w') as f:
    f.write(content)

print("fix_gpu_init_timing.py: Added derived quantity GPU update before GPU->CPU physics sync")
