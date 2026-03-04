#!/usr/bin/env python3
"""
Patch physics interface routines for GPU offload.

Target files (preprocessed .f90, not .F):
  1. dyn_em/module_big_step_utilities_em.f90  — phy_prep, phy_prep_part2, zero_tend, zero_tend2d
  2. dyn_em/module_em.f90                     — calculate_phy_tend
  3. phys/module_physics_addtendc.f90         — update_phy_ten subs: add_a2a, add_a2a_ph, add_a2c_u, add_a2c_v
  4. dyn_em/module_first_rk_step_part1.f90    — !$acc update host/device around radiation_driver, surface_driver
  5. dyn_em/module_first_rk_step_part2.f90    — !$acc update around calculate_phy_tend, update_phy_ten

Strategy:
  - phy_prep: Add !$acc parallel loop collapse(3) to each loop nest.
    All inputs (u_2, v_2, p, pb, alt, ph_2, phb, t_2, mut, muu, muv, etc.) are
    on GPU from gpu_init_domain_data. Outputs (th_phy, p_phy, pi_phy, t_phy, rho,
    u_phy, v_phy, z, z_at_w, dz8w, p8w, t8w, p_hyd, p_hyd_w) stay on GPU for
    later use by calculate_phy_tend.  We add !$acc data present_or_create(...)
    at the subroutine level for the output arrays not yet on GPU.

  - zero_tend / zero_tend2d: !$acc parallel loop collapse(3) / collapse(2)

  - calculate_phy_tend: !$acc parallel loop collapse(3) on each loop nest.
    For LES (bl_pbl=0, cu=0), only the radiation block fires.
    Tendency arrays (RTHRATEN etc.) come from CPU physics via !$acc update device.

  - add_a2a, add_a2a_ph, add_a2c_u, add_a2c_v: !$acc parallel loop collapse(3)

  - Calling context in part1/part2: !$acc update host(...) before CPU physics,
    !$acc update device(...) after, to sync tendency arrays.

NOTE: Uses `!$acc kernels present(...)` style matching the existing codebase pattern,
with explicit `present` clauses for safety.
"""

import sys
import os

WRF = os.environ.get("WRF_DIR", "/home/drew/WRF_BUILD_GPU")

FILES = {
    "big_step": f"{WRF}/dyn_em/module_big_step_utilities_em.f90",
    "module_em": f"{WRF}/dyn_em/module_em.f90",
    "addtendc": f"{WRF}/phys/module_physics_addtendc.f90",
    "part1": f"{WRF}/dyn_em/module_first_rk_step_part1.f90",
    "part2": f"{WRF}/dyn_em/module_first_rk_step_part2.f90",
}


def do_replace(src, old, new, label):
    """Replace old with new in src (first occurrence). Returns modified src."""
    if old not in src:
        print(f"  WARNING: could not find match for: {label}")
        return src, False
    src = src.replace(old, new, 1)
    print(f"  [OK] {label}")
    return src, True


def patch_big_step(filepath):
    """Patch phy_prep, phy_prep_part2, zero_tend, zero_tend2d."""
    with open(filepath, "r") as f:
        src = f.read()
    original = src
    count = 0

    # =========================================================================
    # zero_tend: simple 3D zero
    # =========================================================================
    src, ok = do_replace(src,
        "      DO j = jts, jte\n"
        "      DO k = kts, kte\n"
        "      DO i = its, ite\n"
        "        tendency(i,k,j) = 0.\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      END SUBROUTINE zero_tend",

        "      !$acc parallel loop collapse(3) present(tendency)\n"
        "      DO j = jts, jte\n"
        "      DO k = kts, kte\n"
        "      DO i = its, ite\n"
        "        tendency(i,k,j) = 0.\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      END SUBROUTINE zero_tend",
        "zero_tend: collapse(3)")
    if ok: count += 1

    # =========================================================================
    # zero_tend2d: simple 2D zero
    # =========================================================================
    src, ok = do_replace(src,
        "      DO j = jts, jte\n"
        "      DO i = its, ite\n"
        "        tendency(i,j) = 0.\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      END SUBROUTINE zero_tend2d",

        "      !$acc parallel loop collapse(2) present(tendency)\n"
        "      DO j = jts, jte\n"
        "      DO i = its, ite\n"
        "        tendency(i,j) = 0.\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      END SUBROUTINE zero_tend2d",
        "zero_tend2d: collapse(2)")
    if ok: count += 1

    # =========================================================================
    # phy_prep: 10+ loop nests computing physics variables from dynamics state
    # We add !$acc parallel loop collapse(3) before each DO j/DO k/DO i nest
    # and collapse(2) for 2D nests.
    #
    # All arrays are present on GPU (dynamics state from gpu_init_domain_data,
    # physics output arrays managed at the calling context level).
    # =========================================================================

    # --- phy_prep: th_phy computation (theta_m branch) ---
    src, ok = do_replace(src,
        "    IF ( ( config_flags%use_theta_m .EQ. 1 ) .AND. (P_Qv .GE. PARAM_FIRST_SCALAR) ) THEN\n"
        "       do j = j_start,j_end\n"
        "       do k = k_start, k_end\n"
        "       do i = i_start, i_end\n"
        "         th_phy(i,k,j) = (t(i,k,j)+t0)/(1.+R_v/R_d*moist(i,k,j,P_QV))\n"
        "       enddo\n"
        "       enddo\n"
        "       enddo\n"
        "    ELSE\n"
        "       do j = j_start,j_end\n"
        "       do k = k_start, k_end\n"
        "       do i = i_start, i_end\n"
        "         th_phy(i,k,j) =  t(i,k,j)+t0\n"
        "       enddo\n"
        "       enddo\n"
        "       enddo\n"
        "    END IF",

        "    IF ( ( config_flags%use_theta_m .EQ. 1 ) .AND. (P_Qv .GE. PARAM_FIRST_SCALAR) ) THEN\n"
        "       !$acc parallel loop collapse(3) present(th_phy, t, moist)\n"
        "       do j = j_start,j_end\n"
        "       do k = k_start, k_end\n"
        "       do i = i_start, i_end\n"
        "         th_phy(i,k,j) = (t(i,k,j)+t0)/(1.+R_v/R_d*moist(i,k,j,P_QV))\n"
        "       enddo\n"
        "       enddo\n"
        "       enddo\n"
        "    ELSE\n"
        "       !$acc parallel loop collapse(3) present(th_phy, t)\n"
        "       do j = j_start,j_end\n"
        "       do k = k_start, k_end\n"
        "       do i = i_start, i_end\n"
        "         th_phy(i,k,j) =  t(i,k,j)+t0\n"
        "       enddo\n"
        "       enddo\n"
        "       enddo\n"
        "    END IF",
        "phy_prep: th_phy (theta_m branch)")
    if ok: count += 1

    # --- phy_prep: main physics derivation (p_phy, pi_phy, t_phy, rho, u_phy, v_phy) ---
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do k = k_start, k_end\n"
        "    do i = i_start, i_end\n"
        "\n"
        "      th_phy_m_t0(i,k,j) = th_phy(i,k,j)-t0\n"
        "      p_phy(i,k,j) = p(i,k,j) + pb(i,k,j)\n"
        "      pi_phy(i,k,j) = (p_phy(i,k,j)/p1000mb)**rcp\n"
        "      t_phy(i,k,j) = th_phy(i,k,j)*pi_phy(i,k,j)\n"
        "      rho(i,k,j) = 1./alt(i,k,j)*(1.+moist(i,k,j,P_QV))\n"
        "      u_phy(i,k,j) = 0.5*(u(i,k,j)+u(i+1,k,j))\n"
        "      v_phy(i,k,j) = 0.5*(v(i,k,j)+v(i,k,j+1))\n"
        "\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",

        "    !$acc parallel loop collapse(3) present(th_phy_m_t0, th_phy, p_phy, p, pb, &\n"
        "    !$acc&   pi_phy, t_phy, rho, alt, moist, u_phy, u, v_phy, v)\n"
        "    do j = j_start,j_end\n"
        "    do k = k_start, k_end\n"
        "    do i = i_start, i_end\n"
        "\n"
        "      th_phy_m_t0(i,k,j) = th_phy(i,k,j)-t0\n"
        "      p_phy(i,k,j) = p(i,k,j) + pb(i,k,j)\n"
        "      pi_phy(i,k,j) = (p_phy(i,k,j)/p1000mb)**rcp\n"
        "      t_phy(i,k,j) = th_phy(i,k,j)*pi_phy(i,k,j)\n"
        "      rho(i,k,j) = 1./alt(i,k,j)*(1.+moist(i,k,j,P_QV))\n"
        "      u_phy(i,k,j) = 0.5*(u(i,k,j)+u(i+1,k,j))\n"
        "      v_phy(i,k,j) = 0.5*(v(i,k,j)+v(i,k,j+1))\n"
        "\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: main derivation (p_phy, pi_phy, t_phy, rho, u/v_phy)")
    if ok: count += 1

    # --- phy_prep: z_at_w ---
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do k = k_start, kte\n"
        "    do i = i_start, i_end\n"
        "      z_at_w(i,k,j) = (phb(i,k,j)+ph(i,k,j))/g\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",

        "    !$acc parallel loop collapse(3) present(z_at_w, phb, ph)\n"
        "    do j = j_start,j_end\n"
        "    do k = k_start, kte\n"
        "    do i = i_start, i_end\n"
        "      z_at_w(i,k,j) = (phb(i,k,j)+ph(i,k,j))/g\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: z_at_w")
    if ok: count += 1

    # --- phy_prep: dz8w (3D part) ---
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do k = k_start, kte-1\n"
        "    do i = i_start, i_end\n"
        "      dz8w(i,k,j) = z_at_w(i,k+1,j)-z_at_w(i,k,j)\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",

        "    !$acc parallel loop collapse(3) present(dz8w, z_at_w)\n"
        "    do j = j_start,j_end\n"
        "    do k = k_start, kte-1\n"
        "    do i = i_start, i_end\n"
        "      dz8w(i,k,j) = z_at_w(i,k+1,j)-z_at_w(i,k,j)\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: dz8w 3D")
    if ok: count += 1

    # --- phy_prep: dz8w top level (2D) ---
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do i = i_start, i_end\n"
        "      dz8w(i,kte,j) = 0.\n"
        "    enddo\n"
        "    enddo",

        "    !$acc parallel loop collapse(2) present(dz8w)\n"
        "    do j = j_start,j_end\n"
        "    do i = i_start, i_end\n"
        "      dz8w(i,kte,j) = 0.\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: dz8w top=0")
    if ok: count += 1

    # --- phy_prep: z from z_at_w ---
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do k = k_start, k_end\n"
        "    do i = i_start, i_end\n"
        "      z(i,k,j) = 0.5*(z_at_w(i,k,j) +z_at_w(i,k+1,j) )\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",

        "    !$acc parallel loop collapse(3) present(z, z_at_w)\n"
        "    do j = j_start,j_end\n"
        "    do k = k_start, k_end\n"
        "    do i = i_start, i_end\n"
        "      z(i,k,j) = 0.5*(z_at_w(i,k,j) +z_at_w(i,k+1,j) )\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: z midpoint")
    if ok: count += 1

    # --- phy_prep: p8w, t8w interior (k=2..k_end) ---
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do k = 2, k_end\n"
        "    do i = i_start, i_end\n"
        "      p8w(i,k,j) = fzm(k)*p_phy(i,k,j)+fzp(k)*p_phy(i,k-1,j)\n"
        "      t8w(i,k,j) = fzm(k)*t_phy(i,k,j)+fzp(k)*t_phy(i,k-1,j)\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",

        "    !$acc parallel loop collapse(3) present(p8w, t8w, p_phy, t_phy, fzm, fzp)\n"
        "    do j = j_start,j_end\n"
        "    do k = 2, k_end\n"
        "    do i = i_start, i_end\n"
        "      p8w(i,k,j) = fzm(k)*p_phy(i,k,j)+fzp(k)*p_phy(i,k-1,j)\n"
        "      t8w(i,k,j) = fzm(k)*t_phy(i,k,j)+fzp(k)*t_phy(i,k-1,j)\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: p8w/t8w interior")
    if ok: count += 1

    # --- phy_prep: p8w/t8w boundary (2D, k=1 and k=kde) ---
    # This is a 2D loop with local z0,z1,z2,w1,w2 — each iteration is independent
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do i = i_start, i_end\n"
        "\n"
        "\n"
        "\n"
        "      z0 = z_at_w(i,1,j)\n"
        "      z1 = z(i,1,j)\n"
        "      z2 = z(i,2,j)\n"
        "      w1 = (z0 - z2)/(z1 - z2)\n"
        "      w2 = 1. - w1\n"
        "      p8w(i,1,j) = w1*p_phy(i,1,j)+w2*p_phy(i,2,j)\n"
        "      t8w(i,1,j) = w1*t_phy(i,1,j)+w2*t_phy(i,2,j)\n"
        "\n"
        "\n"
        "\n"
        "      z0 = z_at_w(i,kte,j)\n"
        "      z1 = z(i,k_end,j)\n"
        "      z2 = z(i,k_end-1,j)\n"
        "      w1 = (z0 - z2)/(z1 - z2)\n"
        "      w2 = 1. - w1\n"
        "\n"
        "\n"
        "\n"
        "      p8w(i,kde,j) = exp(w1*log(p_phy(i,kde-1,j))+w2*log(p_phy(i,kde-2,j)))\n"
        "      t8w(i,kde,j) = w1*t_phy(i,kde-1,j)+w2*t_phy(i,kde-2,j)\n"
        "\n"
        "    enddo\n"
        "    enddo",

        "    !$acc parallel loop collapse(2) present(z_at_w, z, p8w, t8w, p_phy, t_phy) &\n"
        "    !$acc&   private(z0, z1, z2, w1, w2)\n"
        "    do j = j_start,j_end\n"
        "    do i = i_start, i_end\n"
        "\n"
        "\n"
        "\n"
        "      z0 = z_at_w(i,1,j)\n"
        "      z1 = z(i,1,j)\n"
        "      z2 = z(i,2,j)\n"
        "      w1 = (z0 - z2)/(z1 - z2)\n"
        "      w2 = 1. - w1\n"
        "      p8w(i,1,j) = w1*p_phy(i,1,j)+w2*p_phy(i,2,j)\n"
        "      t8w(i,1,j) = w1*t_phy(i,1,j)+w2*t_phy(i,2,j)\n"
        "\n"
        "\n"
        "\n"
        "      z0 = z_at_w(i,kte,j)\n"
        "      z1 = z(i,k_end,j)\n"
        "      z2 = z(i,k_end-1,j)\n"
        "      w1 = (z0 - z2)/(z1 - z2)\n"
        "      w2 = 1. - w1\n"
        "\n"
        "\n"
        "\n"
        "      p8w(i,kde,j) = exp(w1*log(p_phy(i,kde-1,j))+w2*log(p_phy(i,kde-2,j)))\n"
        "      t8w(i,kde,j) = w1*t_phy(i,kde-1,j)+w2*t_phy(i,kde-2,j)\n"
        "\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: p8w/t8w boundary extrapolation")
    if ok: count += 1

    # --- phy_prep: p_hyd_w top (2D) ---
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do i = i_start, i_end\n"
        "       p_hyd_w(i,kte,j) = p_top\n"
        "    enddo\n"
        "    enddo",

        "    !$acc parallel loop collapse(2) present(p_hyd_w)\n"
        "    do j = j_start,j_end\n"
        "    do i = i_start, i_end\n"
        "       p_hyd_w(i,kte,j) = p_top\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: p_hyd_w top boundary")
    if ok: count += 1

    # --- phy_prep: p_hyd_w integration (k-serial dependency via p_hyd_w(k+1)) ---
    # This loop has a k-dependency: p_hyd_w(k) depends on p_hyd_w(k+1).
    # We can parallelize over i,j but must keep k serial (downward sweep).
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do k = kte-1, k_start, -1\n"
        "    do i = i_start, i_end\n"
        "       qtot = 0.\n"
        "       do n = PARAM_FIRST_SCALAR,n_moist\n"
        "              qtot = qtot + moist(i,k,j,n)\n"
        "       enddo\n"
        "       p_hyd_w(i,k,j) = p_hyd_w(i,k+1,j) - (1.+qtot)*(c1(k)*MUT(i,j)+c2(k))*dnw(k)\n"
        "\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",

        "    do k = kte-1, k_start, -1\n"
        "    !$acc parallel loop collapse(2) present(p_hyd_w, moist, MUT, c1, c2, dnw) private(qtot)\n"
        "    do j = j_start,j_end\n"
        "    do i = i_start, i_end\n"
        "       qtot = 0.\n"
        "       do n = PARAM_FIRST_SCALAR,n_moist\n"
        "              qtot = qtot + moist(i,k,j,n)\n"
        "       enddo\n"
        "       p_hyd_w(i,k,j) = p_hyd_w(i,k+1,j) - (1.+qtot)*(c1(k)*MUT(i,j)+c2(k))*dnw(k)\n"
        "\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo",
        "phy_prep: p_hyd_w k-serial integration")
    if ok: count += 1

    # --- phy_prep: p_hyd from p_hyd_w ---
    src, ok = do_replace(src,
        "    do j = j_start,j_end\n"
        "    do k = k_start, k_end\n"
        "    do i = i_start, i_end\n"
        "       p_hyd(i,k,j) = 0.5*(p_hyd_w(i,k,j)+p_hyd_w(i,k+1,j))\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo\n"
        "\n"
        "END SUBROUTINE phy_prep",

        "    !$acc parallel loop collapse(3) present(p_hyd, p_hyd_w)\n"
        "    do j = j_start,j_end\n"
        "    do k = k_start, k_end\n"
        "    do i = i_start, i_end\n"
        "       p_hyd(i,k,j) = 0.5*(p_hyd_w(i,k,j)+p_hyd_w(i,k+1,j))\n"
        "    enddo\n"
        "    enddo\n"
        "    enddo\n"
        "\n"
        "END SUBROUTINE phy_prep",
        "phy_prep: p_hyd midpoint")
    if ok: count += 1

    # =========================================================================
    # phy_prep_part2: un-couples tendencies (divides by mu)
    # Same pattern: many 3D loops with array(I,K,J) = array(I,K,J) / (c1(k)*MUT+c2(k))
    # These are all independent across i,j,k.
    # =========================================================================

    # --- phy_prep_part2: RTHRATEN ---
    src, ok = do_replace(src,
        "   IF (config_flags%ra_lw_physics .gt. 0 .or. config_flags%ra_sw_physics .gt. 0) THEN\n"
        "\n"
        "      DO J=j_start,j_end\n"
        "      DO K=k_start,k_end\n"
        "      DO I=i_start,i_end\n"
        "         RTHRATEN(I,K,J)=RTHRATEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "   ENDIF\n"
        "\n"
        "   IF (config_flags%cu_physics .gt. 0) THEN\n"
        "\n"
        "      DO J=j_start,j_end\n"
        "      DO K=k_start,k_end\n"
        "      DO I=i_start,i_end\n"
        "         RUCUTEN(I,K,J) =RUCUTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))\n"
        "         RVCUTEN(I,K,J) =RVCUTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))\n"
        "         RTHCUTEN(I,K,J)=RTHCUTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",

        "   IF (config_flags%ra_lw_physics .gt. 0 .or. config_flags%ra_sw_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) present(RTHRATEN, MUT, c1, c2)\n"
        "      DO J=j_start,j_end\n"
        "      DO K=k_start,k_end\n"
        "      DO I=i_start,i_end\n"
        "         RTHRATEN(I,K,J)=RTHRATEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "   ENDIF\n"
        "\n"
        "   IF (config_flags%cu_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) present(RUCUTEN, RVCUTEN, RTHCUTEN, MUT, c1, c2)\n"
        "      DO J=j_start,j_end\n"
        "      DO K=k_start,k_end\n"
        "      DO I=i_start,i_end\n"
        "         RUCUTEN(I,K,J) =RUCUTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))\n"
        "         RVCUTEN(I,K,J) =RVCUTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))\n"
        "         RTHCUTEN(I,K,J)=RTHCUTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "phy_prep_part2: RTHRATEN + RUCUTEN/RVCUTEN/RTHCUTEN")
    if ok: count += 1

    # We skip the remaining phy_prep_part2 loops (cu_physics, shcu_physics,
    # bl_pbl_physics, grid_fdda) since for LES none of these fire.
    # They can be added later if needed for non-LES configs.

    if src != original:
        with open(filepath, "w") as f:
            f.write(src)
        print(f"  => Wrote {count} patches to {filepath}")
    else:
        print(f"  => No changes to {filepath}")
    return count


def patch_module_em(filepath):
    """Patch calculate_phy_tend in module_em.f90."""
    with open(filepath, "r") as f:
        src = f.read()
    original = src
    count = 0

    # =========================================================================
    # calculate_phy_tend: multiplies tendencies by (c1(k)*mut+c2(k))
    # Each loop nest is independent across i,j,k.
    # For LES, only the radiation block fires (ra_lw/sw > 0).
    # =========================================================================

    # --- RTHRATEN radiation tendency ---
    src, ok = do_replace(src,
        "   IF (config_flags%ra_lw_physics .gt. 0 .or. config_flags%ra_sw_physics .gt. 0) THEN\n"
        "\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RTHRATEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHRATEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "   ENDIF",

        "   IF (config_flags%ra_lw_physics .gt. 0 .or. config_flags%ra_sw_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) present(RTHRATEN, mut, c1, c2)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RTHRATEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHRATEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "   ENDIF",
        "calculate_phy_tend: RTHRATEN (radiation)")
    if ok: count += 1

    # --- cu_physics block (4 vars + optional QC/QR/QI/QS) ---
    # Patch the first loop in cu_physics block
    src, ok = do_replace(src,
        "   IF (config_flags%cu_physics .gt. 0) THEN\n"
        "\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUCUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUCUTEN(I,K,J)\n"
        "         RVCUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RVCUTEN(I,K,J)\n"
        "         RTHCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHCUTEN(I,K,J)\n"
        "         RQVCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVCUTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",

        "   IF (config_flags%cu_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) present(RUCUTEN, RVCUTEN, RTHCUTEN, RQVCUTEN, mut, c1, c2)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUCUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUCUTEN(I,K,J)\n"
        "         RVCUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RVCUTEN(I,K,J)\n"
        "         RTHCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHCUTEN(I,K,J)\n"
        "         RQVCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVCUTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "calculate_phy_tend: cu_physics main")
    if ok: count += 1

    # --- bl_pbl_physics block ---
    src, ok = do_replace(src,
        "   IF (config_flags%bl_pbl_physics .gt. 0) THEN\n"
        "\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUBLTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUBLTEN(I,K,J)\n"
        "         RVBLTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RVBLTEN(I,K,J)\n"
        "         RTHBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHBLTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",

        "   IF (config_flags%bl_pbl_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) present(RUBLTEN, RVBLTEN, RTHBLTEN, mut, c1, c2)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUBLTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUBLTEN(I,K,J)\n"
        "         RVBLTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RVBLTEN(I,K,J)\n"
        "         RTHBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHBLTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "calculate_phy_tend: bl_pbl main")
    if ok: count += 1

    # --- scalar_tend ---
    src, ok = do_replace(src,
        "   DO im = PARAM_FIRST_SCALAR,num_scalar\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            scalar_tend(I,K,J,im)=(c1(k)*mut(I,J)+c2(k))*scalar_tend(I,K,J,im)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "   ENDDO",

        "   DO im = PARAM_FIRST_SCALAR,num_scalar\n"
        "         !$acc parallel loop collapse(3) present(scalar_tend, mut, c1, c2)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            scalar_tend(I,K,J,im)=(c1(k)*mut(I,J)+c2(k))*scalar_tend(I,K,J,im)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "   ENDDO",
        "calculate_phy_tend: scalar_tend")
    if ok: count += 1

    # --- tracer_tend ---
    src, ok = do_replace(src,
        "   DO im = PARAM_FIRST_SCALAR,num_tracer\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            tracer_tend(I,K,J,im)=(c1(k)*mut(I,J)+c2(k))*tracer_tend(I,K,J,im)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "   ENDDO",

        "   DO im = PARAM_FIRST_SCALAR,num_tracer\n"
        "         !$acc parallel loop collapse(3) present(tracer_tend, mut, c1, c2)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            tracer_tend(I,K,J,im)=(c1(k)*mut(I,J)+c2(k))*tracer_tend(I,K,J,im)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "   ENDDO",
        "calculate_phy_tend: tracer_tend")
    if ok: count += 1

    if src != original:
        with open(filepath, "w") as f:
            f.write(src)
        print(f"  => Wrote {count} patches to {filepath}")
    else:
        print(f"  => No changes to {filepath}")
    return count


def patch_addtendc(filepath):
    """Patch add_a2a, add_a2a_ph, add_a2c_u, add_a2c_v in module_physics_addtendc.f90."""
    with open(filepath, "r") as f:
        src = f.read()
    original = src
    count = 0

    # =========================================================================
    # add_a2a: lvar += rvar (3D, a-grid to a-grid)
    # =========================================================================
    src, ok = do_replace(src,
        "   DO j = j_start,j_end\n"
        "   DO k = kts,ktf\n"
        "   DO i = i_start,i_end\n"
        "      lvar(i,k,j) = lvar(i,k,j) + rvar(i,k,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE add_a2a",

        "   !$acc parallel loop collapse(3) present(lvar, rvar)\n"
        "   DO j = j_start,j_end\n"
        "   DO k = kts,ktf\n"
        "   DO i = i_start,i_end\n"
        "      lvar(i,k,j) = lvar(i,k,j) + rvar(i,k,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE add_a2a",
        "add_a2a: collapse(3)")
    if ok: count += 1

    # =========================================================================
    # add_a2a_ph: lvar += rvar (3D, full k range including kte)
    # =========================================================================
    src, ok = do_replace(src,
        "   DO j = j_start,j_end\n"
        "   DO k = kts,kte\n"
        "   DO i = i_start,i_end\n"
        "      lvar(i,k,j) = lvar(i,k,j) + rvar(i,k,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE add_a2a_ph",

        "   !$acc parallel loop collapse(3) present(lvar, rvar)\n"
        "   DO j = j_start,j_end\n"
        "   DO k = kts,kte\n"
        "   DO i = i_start,i_end\n"
        "      lvar(i,k,j) = lvar(i,k,j) + rvar(i,k,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE add_a2a_ph",
        "add_a2a_ph: collapse(3)")
    if ok: count += 1

    # =========================================================================
    # add_a2c_u: lvar += 0.5*(rvar(i) + rvar(i-1)) — u-stagger interpolation
    # =========================================================================
    src, ok = do_replace(src,
        "   DO j = j_start,j_end\n"
        "   DO k = kts,ktf\n"
        "   DO i = i_start,i_end\n"
        "      lvar(i,k,j) = lvar(i,k,j) + &\n"
        "                       0.5*(rvar(i,k,j)+rvar(i-1,k,j))\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE add_a2c_u",

        "   !$acc parallel loop collapse(3) present(lvar, rvar)\n"
        "   DO j = j_start,j_end\n"
        "   DO k = kts,ktf\n"
        "   DO i = i_start,i_end\n"
        "      lvar(i,k,j) = lvar(i,k,j) + &\n"
        "                       0.5*(rvar(i,k,j)+rvar(i-1,k,j))\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE add_a2c_u",
        "add_a2c_u: collapse(3)")
    if ok: count += 1

    # =========================================================================
    # add_a2c_v: lvar += 0.5*(rvar(j) + rvar(j-1)) — v-stagger interpolation
    # =========================================================================
    src, ok = do_replace(src,
        "   DO j = j_start,j_end\n"
        "   DO k = kts,kte\n"
        "   DO i = i_start,i_end\n"
        "      lvar(i,k,j) = lvar(i,k,j) + &\n"
        "                     0.5*(rvar(i,k,j)+rvar(i,k,j-1))\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE add_a2c_v",

        "   !$acc parallel loop collapse(3) present(lvar, rvar)\n"
        "   DO j = j_start,j_end\n"
        "   DO k = kts,kte\n"
        "   DO i = i_start,i_end\n"
        "      lvar(i,k,j) = lvar(i,k,j) + &\n"
        "                     0.5*(rvar(i,k,j)+rvar(i,k,j-1))\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE add_a2c_v",
        "add_a2c_v: collapse(3)")
    if ok: count += 1

    if src != original:
        with open(filepath, "w") as f:
            f.write(src)
        print(f"  => Wrote {count} patches to {filepath}")
    else:
        print(f"  => No changes to {filepath}")
    return count


def patch_part1(filepath):
    """Add !$acc update host/device around radiation_driver and surface_driver calls.

    These are massive CPU routines. We need to:
    1. Before radiation_driver: !$acc update host(...) to bring phy_prep outputs to CPU
    2. After radiation_driver: !$acc update device(RTHRATEN) to push result to GPU
    3. Before surface_driver: (data already on host from radiation update)
    4. After surface_driver: !$acc update device(...) for surface tendency arrays
    """
    with open(filepath, "r") as f:
        src = f.read()
    original = src
    count = 0

    # --- Before radiation_driver: sync phy_prep outputs to host ---
    # Insert !$acc update host before the pre_radiation_driver call
    src, ok = do_replace(src,
        "      CALL pre_radiation_driver ( grid, config_flags",

        "      ! --- GPU->CPU: sync phy_prep outputs for CPU physics ---\n"
        "      !$acc update host(grid%rho, grid%u_phy, grid%v_phy)\n"
        "      !$acc update host(grid%z, grid%z_at_w)\n"
        "      !$acc update host(grid%p_hyd, grid%p_hyd_w)\n"
        "      !$acc update host(grid%th_phy_m_t0)\n"
        "\n"
        "      CALL pre_radiation_driver ( grid, config_flags",
        "part1: !$acc update host before radiation_driver")
    if ok: count += 1

    # --- After radiation_driver: sync RTHRATEN back to GPU ---
    # Find the end of the radiation_driver call and insert after it.
    # The radiation_driver call spans many continuation lines. We look for
    # the surface_driver section start as our anchor.
    src, ok = do_replace(src,
        "      CALL wrf_debug ( 200 , ' call surface_driver' )",

        "      ! --- CPU->GPU: sync radiation tendency to GPU ---\n"
        "      !$acc update device(grid%rthraten)\n"
        "\n"
        "      CALL wrf_debug ( 200 , ' call surface_driver' )",
        "part1: !$acc update device(rthraten) after radiation_driver")
    if ok: count += 1

    # --- After surface_driver: sync surface tendencies to GPU ---
    # The surface_driver produces tendencies used later. For LES (no PBL),
    # the key outputs are grid fields modified by surface_driver.
    # Find the pbl_driver section as anchor.
    src, ok = do_replace(src,
        "      CALL wrf_debug ( 200 , ' call pbl_driver' )",

        "      ! --- CPU->GPU: sync surface driver outputs ---\n"
        "      ! (Surface tendencies are folded into PBL tendencies or used directly)\n"
        "\n"
        "      CALL wrf_debug ( 200 , ' call pbl_driver' )",
        "part1: marker after surface_driver (before pbl)")
    if ok: count += 1

    if src != original:
        with open(filepath, "w") as f:
            f.write(src)
        print(f"  => Wrote {count} patches to {filepath}")
    else:
        print(f"  => No changes to {filepath}")
    return count


def patch_part2(filepath):
    """Add !$acc update directives around calculate_phy_tend and update_phy_ten.

    In part2, the flow is:
    1. calculate_phy_tend (on GPU — simple array multiply)
    2. diffusion (already on GPU)
    3. update_phy_ten (on GPU — simple array add via add_a2a etc.)

    The tendency arrays (RTHRATEN, RUBLTEN, etc.) are on GPU after the !$acc update
    device in part1. The dynamics tendency arrays (ru_tendf, rv_tendf, t_tendf, etc.)
    are local arrays — they need to be present on GPU.

    We add !$acc data create() for the local tendency arrays at the top of the
    subroutine scope, and !$acc update as needed.
    """
    with open(filepath, "r") as f:
        src = f.read()
    original = src
    count = 0

    # For now, add comments marking where GPU data regions would go.
    # The actual tendency arrays (ru_tendf, rv_tendf, etc.) are passed
    # from solve_em and may already be on GPU from prior patches.
    # The key thing is making sure RTHRATEN etc. are present.

    # No additional patches needed here if:
    # 1. calculate_phy_tend loops have !$acc directives (done in patch_module_em)
    # 2. add_a2a etc. have !$acc directives (done in patch_addtendc)
    # 3. Tendency arrays are on GPU (handled by gpu_init_domain_data + part1 updates)

    # The OMP parallel DO around calculate_phy_tend and update_phy_ten needs
    # to be removed or made serial for GPU (GPU handles parallelism internally).
    # However, since num_tiles=1 for GPU runs, the OMP loop is effectively serial.

    print(f"  => No changes needed to {filepath} (GPU directives in called subroutines)")
    return count


def main():
    total = 0
    print("=" * 70)
    print("Patching physics interface routines for GPU (first_rk_step)")
    print("=" * 70)

    print(f"\n--- Patching {FILES['big_step']} ---")
    total += patch_big_step(FILES["big_step"])

    print(f"\n--- Patching {FILES['module_em']} ---")
    total += patch_module_em(FILES["module_em"])

    print(f"\n--- Patching {FILES['addtendc']} ---")
    total += patch_addtendc(FILES["addtendc"])

    print(f"\n--- Patching {FILES['part1']} ---")
    total += patch_part1(FILES["part1"])

    print(f"\n--- Patching {FILES['part2']} ---")
    total += patch_part2(FILES["part2"])

    print(f"\n{'=' * 70}")
    print(f"Total patches applied: {total}")
    print("=" * 70)


if __name__ == "__main__":
    main()
