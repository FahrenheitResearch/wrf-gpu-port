#!/usr/bin/env python3
"""Fix GPU data initialization timing issue.

ROOT CAUSE: gpu_init_domain_data copies grid arrays to GPU at startup,
but derived quantities (mut, muts, muu, muv, alt, php, etc.) are computed
AFTER gpu_init runs. The GPU has zeros for these arrays.

Then line 878 in solve_em.F does !$acc update host() which copies stale
GPU zeros back to host, destroying correct host values.

FIX: Add !$acc update device() for derived quantities at the start of
solve_em, BEFORE the GPU->CPU sync at line 878. This pushes correct host
values to GPU, so the subsequent GPU->CPU sync is a no-op (same values).
"""

filepath = "/home/drew/WRF_BUILD_GPU/dyn_em/solve_em.F"

with open(filepath, 'r') as f:
    content = f.read()

# Remove diagnostic code (from add_synca_diagnostics.py and add_advance_w_diagnostics.py)
# These will be handled by removing the whole blocks

# Find the GPU->CPU sync at line 878 area
# Insert a CPU->GPU update BEFORE it to fix the derived quantities
marker = """#ifdef GPU_OPENACC
! GPU_OPENACC: Sync state arrays GPU->CPU before physics runs on CPU
!$acc update host(grid%t_1, grid%t_2, grid%u_1, grid%u_2)"""

fix = """#ifdef GPU_OPENACC
! FIX: Push derived quantities to GPU (computed after gpu_init_domain_data)
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
#endif

#ifdef GPU_OPENACC
! GPU_OPENACC: Sync state arrays GPU->CPU before physics runs on CPU
!$acc update host(grid%t_1, grid%t_2, grid%u_1, grid%u_2)"""

content = content.replace(marker, fix, 1)

with open(filepath, 'w') as f:
    f.write(content)

print("Added derived quantity GPU update before the GPU->CPU physics sync")
