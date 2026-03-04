#!/usr/bin/env python3
"""Fix GPU<->CPU sync in the acoustic loop of solve_em.F.

This script:
1. Removes all previously-inserted SYNC markers
2. Adds correct micro-sync points within the acoustic loop

The acoustic loop structure for specified BCs (non-polar) is:
  [Sync A: CPU->GPU for tendencies] (already exists)
  small_step_prep (GPU)
  calc_p_rho (GPU)
  calc_coef_w (GPU) [first RK step only]
  set_physical_bc (CPU) on ru_tend, rv_tend, ph_2, al, p, t_1, t_save, mu_1, mu_2, mudf
  [acoustic iteration loop]
    advance_uv (GPU) -> writes u_2, v_2
    spec_bdyupdate u_2, v_2 (CPU)
    advance_mu_t (GPU) -> writes mu_2, t_2, ww, muave
    [polar filter mu,t - skipped for non-polar]
    spec_bdyupdate t_2, mu_2, muts (CPU)
    advance_w (GPU) -> writes w_2, ph_2
    [polar filter w,ph - skipped for non-polar]
    sumflux (GPU) -> writes ru_m, rv_m, ww_m [reads u_2,v_2,ww]
    spec_bdynhyd: spec_bdyupdate_ph + zero_grad_bdy w_2 (CPU)
    calc_p_rho (GPU)
    set_physical_bc ph_2, al, p, muts, mu_2, mudf (CPU)
  [end acoustic loop]
  small_step_finish (GPU)
  [Sync B: GPU->CPU for state] (already exists)
"""
import re
import os, sys

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

filepath = os.path.join(WRF_DIR, "dyn_em", "solve_em.F")

with open(filepath, 'r') as f:
    lines = f.readlines()

# Step 1: Remove all existing SYNC markers (from previous fix attempts)
cleaned = []
skip_ifdef = False
for i, line in enumerate(lines):
    # Remove blocks: #ifdef GPU_OPENACC / ! SYNC N: ... / !$acc update ... / #endif
    if '! SYNC ' in line and ('GPU->CPU' in line or 'CPU->GPU' in line):
        # This is a sync marker line - remove it and surrounding #ifdef/#endif
        # Go back and remove the #ifdef GPU_OPENACC line
        while cleaned and cleaned[-1].strip() in ['#ifdef GPU_OPENACC', '']:
            cleaned.pop()
        skip_ifdef = True
        continue
    if skip_ifdef:
        if '!$acc update' in line:
            continue
        if line.strip() == '#endif':
            skip_ifdef = False
            continue
        # If we hit something else, stop skipping
        skip_ifdef = False
    cleaned.append(line)

content = ''.join(cleaned)

# Step 2: Add micro-sync points

# ============================================================
# SYNC 0: After small_step_prep + calc_coef_w, sync GPU-written
# local arrays to host (ph_save, w_save needed by CPU spec_bdy later)
# Insert after the set_phys_bc2_tim block that follows calc_coef_w
# ============================================================
marker0 = "     small_steps : DO iteration = 1 , number_of_small_timesteps"
sync0 = """#ifdef GPU_OPENACC
! SYNC 0: GPU->CPU for locally-computed arrays needed by CPU boundary routines
!$acc update host(ph_save, w_save, mu_save, ww1, c2a, a, alpha, gamma)
#endif
"""
content = content.replace(marker0, sync0 + marker0, 1)

# ============================================================
# SYNC 1: After advance_uv (GPU), before spec_bdyupdate u,v (CPU)
# ============================================================
marker1 = "BENCH_START(spec_bdy_uv_tim)"
sync1 = """#ifdef GPU_OPENACC
! SYNC 1: GPU->CPU after advance_uv
!$acc update host(grid%u_2, grid%v_2)
#endif
"""
content = content.replace(marker1, sync1 + marker1, 1)

# ============================================================
# SYNC 2: After spec_bdyupdate u,v (CPU), before advance_mu_t (GPU)
# ============================================================
marker2 = "        !  advance the mass in the column, theta, and calculate ww"
sync2 = """#ifdef GPU_OPENACC
! SYNC 2: CPU->GPU after boundary update u,v
!$acc update device(grid%u_2, grid%v_2)
#endif
"""
content = content.replace(marker2, sync2 + marker2, 1)

# ============================================================
# SYNC 3: After advance_mu_t (GPU), before spec_bdyupdate t,mu (CPU)
# Need to sync: t_2, mu_2, ww, muts (muts = mut + mu_2 at polar filter)
# Also muave which advance_mu_t computes
# ============================================================
marker3 = "BENCH_START(spec_bdy_t_tim)"
sync3 = """#ifdef GPU_OPENACC
! SYNC 3: GPU->CPU after advance_mu_t
!$acc update host(grid%t_2, grid%mu_2, grid%ww, grid%muts, muave)
#endif
"""
content = content.replace(marker3, sync3 + marker3, 1)

# ============================================================
# SYNC 4: After spec_bdyupdate t,mu,muts (CPU), before advance_w (GPU)
# Also muts may be updated by polar filter or spec_bdy
# ============================================================
marker4 = "         ! small (acoustic) step for the vertical momentum,"
sync4 = """#ifdef GPU_OPENACC
! SYNC 4: CPU->GPU after boundary update t,mu,muts
!$acc update device(grid%t_2, grid%mu_2, grid%muts)
#endif
"""
content = content.replace(marker4, sync4 + marker4, 1)

# ============================================================
# SYNC 5: After advance_w (GPU), before sumflux+spec_bdynhyd
# sumflux reads u_2,v_2,ww (already synced) and writes ru_m,rv_m,ww_m
# spec_bdynhyd reads ph_2, w_2 (need sync from GPU)
# Insert between the polar filter section and the tile loop with sumflux
# Actually sumflux and spec_bdynhyd are in same tile loop.
# sumflux reads u_2,v_2,ww from GPU — fine
# spec_bdyupdate_ph reads ph_save, ph_2 from CPU — need sync!
# But we can do GPU->CPU for w_2, ph_2 right before spec_bdynhyd
# ============================================================
marker5 = "BENCH_START(spec_bdynhyd_tim)"
sync5 = """#ifdef GPU_OPENACC
! SYNC 5: GPU->CPU for w_2, ph_2 before boundary update
!$acc update host(grid%w_2, grid%ph_2)
#endif
"""
content = content.replace(marker5, sync5 + marker5, 1)

# ============================================================
# SYNC 6: After spec_bdynhyd (CPU), before calc_p_rho (GPU)
# Need ph_2, w_2 back on GPU after boundary modification
# Also need muts on GPU (set_physical_bc2d modifies it)
# ============================================================
marker6 = "BENCH_START(cald_p_rho_tim)"
# cald_p_rho_tim is the acoustic-loop calc_p_rho (only one occurrence)
sync6 = """#ifdef GPU_OPENACC
! SYNC 6: CPU->GPU after spec_bdynhyd, before calc_p_rho
!$acc update device(grid%w_2, grid%ph_2)
#endif
"""
content = content.replace(marker6, sync6 + marker6, 1)

# ============================================================
# SYNC 7: After calc_p_rho (GPU), before set_physical_bc (CPU)
# calc_p_rho computes al, p, ph_2 on GPU. set_physical_bc needs them on CPU.
# ============================================================
marker7 = "BENCH_START(phys_bc_tim)"
sync7 = """#ifdef GPU_OPENACC
! SYNC 7: GPU->CPU after calc_p_rho, before physical BC
!$acc update host(grid%al, grid%p, grid%ph_2, grid%mu_2, grid%muts, grid%mudf)
#endif
"""
content = content.replace(marker7, sync7 + marker7, 1)

# ============================================================
# SYNC 8: After set_physical_bc (CPU), before next acoustic iteration
# Push boundary-updated arrays back to GPU
# ============================================================
marker8 = "BENCH_END(phys_bc_tim)"
sync8 = """
#ifdef GPU_OPENACC
! SYNC 8: CPU->GPU after physical BC, before next acoustic step
!$acc update device(grid%al, grid%p, grid%ph_2, grid%mu_2, grid%muts, grid%mudf, grid%u_2, grid%v_2, grid%w_2)
#endif
"""
# Replace first occurrence only (inside acoustic loop)
idx8 = content.find(marker8)
if idx8 >= 0:
    content = content[:idx8 + len(marker8)] + sync8 + content[idx8 + len(marker8):]

with open(filepath, 'w') as f:
    f.write(content)

# Verify
sync_count = len(re.findall(r'! SYNC \d+:', content))
update_count = content.count('!$acc update')
print(f"solve_em.F acoustic loop sync v2 applied:")
print(f"  Sync points added: {sync_count}")
print(f"  Total !$acc update directives: {update_count}")

# Print sync point locations
for i, line in enumerate(content.split('\n'), 1):
    if '! SYNC' in line:
        print(f"  Line {i}: {line.strip()}")
