#!/usr/bin/env python3
"""Fix GPU<->CPU sync in the acoustic loop of solve_em.f90.

This script:
1. Removes all previously-inserted SYNC markers
2. Adds correct micro-sync points within the acoustic loop

Targets the .f90 file (post-preprocessor), consistent with all other patches.
The !$acc directives are inserted directly (no #ifdef guards needed).

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

filepath = os.path.join(WRF_DIR, "dyn_em", "solve_em.f90")

with open(filepath, 'r') as f:
    lines = f.readlines()

# Step 1: Remove all existing SYNC markers (from previous fix attempts)
cleaned = []
for i, line in enumerate(lines):
    # Remove SYNC comment + following !$acc update line
    if '! SYNC ' in line and ('GPU->CPU' in line or 'CPU->GPU' in line):
        # Check if next line is an acc update and skip it too
        continue
    # Skip orphaned !$acc update lines that follow a removed SYNC marker
    if i > 0 and '! SYNC ' in lines[i-1] and '!$acc update' in line:
        continue
    cleaned.append(line)

content = ''.join(cleaned)

# Step 2: Add micro-sync points (no #ifdef guards — this is .f90)

# ============================================================
# SYNC 0: Before acoustic loop, sync GPU-written local arrays
# to host (ph_save, w_save needed by CPU spec_bdy later)
# These locals are on GPU via !$acc data create() from patch_solve_em_gpu.py
# ============================================================
marker0 = "     small_steps : DO iteration = 1 , number_of_small_timesteps"
sync0 = """! SYNC 0: GPU->CPU for locally-computed arrays needed by CPU boundary routines
!$acc update host(ph_save, w_save, mu_save, ww1, c2a, a, alpha, gamma)
"""
content = content.replace(marker0, sync0 + marker0, 1)

# ============================================================
# SYNC 1: After advance_uv (GPU), before spec_bdyupdate u,v (CPU)
# ============================================================
marker1 = "BENCH_START(spec_bdy_uv_tim)"
sync1 = """! SYNC 1: GPU->CPU after advance_uv
!$acc update host(grid%u_2, grid%v_2)
"""
content = content.replace(marker1, sync1 + marker1, 1)

# ============================================================
# SYNC 2: After spec_bdyupdate u,v (CPU), before advance_mu_t (GPU)
# ============================================================
marker2 = "        !  advance the mass in the column, theta, and calculate ww"
sync2 = """! SYNC 2: CPU->GPU after boundary update u,v
!$acc update device(grid%u_2, grid%v_2)
"""
content = content.replace(marker2, sync2 + marker2, 1)

# ============================================================
# SYNC 3: After advance_mu_t (GPU), before spec_bdyupdate t,mu (CPU)
# ============================================================
marker3 = "BENCH_START(spec_bdy_t_tim)"
sync3 = """! SYNC 3: GPU->CPU after advance_mu_t
!$acc update host(grid%t_2, grid%mu_2, grid%ww, grid%muts, muave)
"""
content = content.replace(marker3, sync3 + marker3, 1)

# ============================================================
# SYNC 4: After spec_bdyupdate t,mu,muts (CPU), before advance_w (GPU)
# ============================================================
marker4 = "         ! small (acoustic) step for the vertical momentum,"
sync4 = """! SYNC 4: CPU->GPU after boundary update t,mu,muts
!$acc update device(grid%t_2, grid%mu_2, grid%muts)
"""
content = content.replace(marker4, sync4 + marker4, 1)

# ============================================================
# SYNC 5: After advance_w (GPU), before spec_bdynhyd (CPU)
# ============================================================
marker5 = "BENCH_START(spec_bdynhyd_tim)"
sync5 = """! SYNC 5: GPU->CPU for w_2, ph_2 before boundary update
!$acc update host(grid%w_2, grid%ph_2)
"""
content = content.replace(marker5, sync5 + marker5, 1)

# ============================================================
# SYNC 6: After spec_bdynhyd (CPU), before calc_p_rho (GPU)
# ============================================================
marker6 = "BENCH_START(cald_p_rho_tim)"
sync6 = """! SYNC 6: CPU->GPU after spec_bdynhyd, before calc_p_rho
!$acc update device(grid%w_2, grid%ph_2)
"""
content = content.replace(marker6, sync6 + marker6, 1)

# ============================================================
# SYNC 7: After calc_p_rho (GPU), before set_physical_bc (CPU)
# ============================================================
marker7 = "BENCH_START(phys_bc_tim)"
sync7 = """! SYNC 7: GPU->CPU after calc_p_rho, before physical BC
!$acc update host(grid%al, grid%p, grid%ph_2, grid%mu_2, grid%muts, grid%mudf)
"""
content = content.replace(marker7, sync7 + marker7, 1)

# ============================================================
# SYNC 8: After set_physical_bc (CPU), before next acoustic iteration
# ============================================================
marker8 = "BENCH_END(phys_bc_tim)"
sync8 = """
! SYNC 8: CPU->GPU after physical BC, before next acoustic step
!$acc update device(grid%al, grid%p, grid%ph_2, grid%mu_2, grid%muts, grid%mudf, grid%u_2, grid%v_2, grid%w_2)
"""
idx8 = content.find(marker8)
if idx8 >= 0:
    content = content[:idx8 + len(marker8)] + sync8 + content[idx8 + len(marker8):]

with open(filepath, 'w') as f:
    f.write(content)

# Verify
sync_count = len(re.findall(r'! SYNC \d+:', content))
update_count = content.count('!$acc update')
print(f"solve_em.f90 acoustic loop sync v2 applied:")
print(f"  Sync points added: {sync_count}")
print(f"  Total !$acc update directives: {update_count}")

for i, line in enumerate(content.split('\n'), 1):
    if '! SYNC' in line:
        print(f"  Line {i}: {line.strip()}")
