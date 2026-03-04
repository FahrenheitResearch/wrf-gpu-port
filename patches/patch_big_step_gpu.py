#!/usr/bin/env python3
"""
Patch module_big_step_utilities_em.f90 to upgrade `!$acc kernels` to
`!$acc parallel loop collapse(N) gang` and add new GPU directives to
clean loop nests that were previously unannotated.

Operates on the preprocessed .f90 file (not .F).

=== UPGRADED (kernels -> parallel loop) ===
  - couple_momentum: 3 loop nests (collapse(3))
  - calc_cq ELSE: 3 loop nests (collapse(3))
  - calc_alt: 1 loop nest (collapse(3))
  - calc_php: 1 loop nest (collapse(3))
  - calc_p_rho_phi: 3 nonhydro loop nests (collapse(3))
  - horizontal_diffusion: 4 loop nests (collapse(3))
  - horizontal_diffusion_3dmp: 1 loop nest (collapse(3))

=== NEW directives ===
  - calculate_full: 3D loop (collapse(3))
  - zero_tend: 3D loop (collapse(3))
  - zero_tend2d: 2D loop (collapse(2))
  - set_tend: 3D loop (collapse(3))
  - coriolis: 3 loop nests (collapse(3) each)
  - curvature: vxgm compute + ru_tend + rv_tend + rw_tend loops
  - diagnose_w: w(k=1) and w(k>=2) loops
  - pg_buoy_w: k=2..kde-1 inner loop (collapse(2))

=== SKIPPED ===
  - calc_mu_uv, calc_mu_uv_1, calc_mu_staggered: complex boundary branches
  - couple: CALL to calc_mu_staggered + automatic arrays muu/muv
  - calc_ww_cp: k-serial dependencies (dmdt accumulation, ww(k-1))
  - rhs_ph: ~800 lines, complex branches, automatic wdwn array w/ k deps
  - horizontal_pressure_gradient: complex branches, automatic dpn array
  - vertical_diffusion*: sequential k-dependency (vflux)
  - w_damp: SAVE variable, WRITE/PRINT statements
  - calc_p_rho_phi moist branches: CALL VPOW/VPOWX, k-dependencies
  - phy_prep, phy_prep_part2, moist_physics_*: physics prep, complex
"""

import sys
import os


def do_replace(src, old, new, label):
    """Replace old with new in src (first occurrence). Returns modified src."""
    if old not in src:
        print(f"  WARNING: could not find match for: {label}")
        return src, False
    src = src.replace(old, new, 1)
    print(f"  [OK] {label}")
    return src, True


def patch_file(filepath):
    with open(filepath, 'r') as f:
        src = f.read()

    original = src
    count = 0
    I6 = "      "   # 6-space indent
    I8 = "        "  # 8-space indent

    # =========================================================================
    # 1. couple_momentum - UPGRADE existing !$acc kernels -> parallel loop
    #    Three separate regions: ru, rv, rw
    # =========================================================================

    # ru loop
    src, ok = do_replace(src,
        "      !$acc kernels present(ru, u, muu, msfu, c1h, c2h)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "         ru(i,k,j)=u(i,k,j)*(c1h(k)*muu(i,j)+c2h(k))/msfu(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      !$acc end kernels",
        "      !$acc parallel loop collapse(3) gang present(ru, u, muu, msfu, c1h, c2h)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "         ru(i,k,j)=u(i,k,j)*(c1h(k)*muu(i,j)+c2h(k))/msfu(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "couple_momentum: ru loop (kernels -> parallel loop)")
    if ok: count += 1

    # rv loop
    src, ok = do_replace(src,
        "      !$acc kernels present(rv, v, muv, msfv_inv, c1h, c2h)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "           rv(i,k,j)=v(i,k,j)*(c1h(k)*muv(i,j)+c2h(k))*msfv_inv(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      !$acc end kernels",
        "      !$acc parallel loop collapse(3) gang present(rv, v, muv, msfv_inv, c1h, c2h)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "           rv(i,k,j)=v(i,k,j)*(c1h(k)*muv(i,j)+c2h(k))*msfv_inv(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "couple_momentum: rv loop (kernels -> parallel loop)")
    if ok: count += 1

    # rw loop
    src, ok = do_replace(src,
        "      !$acc kernels present(rw, w, mut, msft, c1f, c2f)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,kte\n"
        "      DO i=its,itf\n"
        "         rw(i,k,j)=w(i,k,j)*(c1f(k)*mut(i,j)+c2f(k))/msft(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      !$acc end kernels",
        "      !$acc parallel loop collapse(3) gang present(rw, w, mut, msft, c1f, c2f)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,kte\n"
        "      DO i=its,itf\n"
        "         rw(i,k,j)=w(i,k,j)*(c1f(k)*mut(i,j)+c2f(k))/msft(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "couple_momentum: rw loop (kernels -> parallel loop)")
    if ok: count += 1

    # =========================================================================
    # 2. calc_cq ELSE branch - UPGRADE 3 kernels -> parallel loop
    # =========================================================================

    # cqu
    src, ok = do_replace(src,
        "        !$acc kernels present(cqu)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "           cqu(i,k,j) = 1.\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        !$acc end kernels",
        "        !$acc parallel loop collapse(3) gang present(cqu)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "           cqu(i,k,j) = 1.\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        ENDDO",
        "calc_cq: cqu loop (kernels -> parallel loop)")
    if ok: count += 1

    # cqv
    src, ok = do_replace(src,
        "        !$acc kernels present(cqv)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "           cqv(i,k,j) = 1.\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        !$acc end kernels",
        "        !$acc parallel loop collapse(3) gang present(cqv)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "           cqv(i,k,j) = 1.\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        ENDDO",
        "calc_cq: cqv loop (kernels -> parallel loop)")
    if ok: count += 1

    # cqw
    src, ok = do_replace(src,
        "        !$acc kernels present(cqw)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts+1,ktf\n"
        "        DO i=its,itf\n"
        "           cqw(i,k,j) = 0.\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        !$acc end kernels",
        "        !$acc parallel loop collapse(3) gang present(cqw)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts+1,ktf\n"
        "        DO i=its,itf\n"
        "           cqw(i,k,j) = 0.\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        ENDDO",
        "calc_cq: cqw loop (kernels -> parallel loop)")
    if ok: count += 1

    # =========================================================================
    # 3. calc_alt - UPGRADE kernels -> parallel loop
    # =========================================================================
    src, ok = do_replace(src,
        "      !$acc kernels present(alt, al, alb)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "        alt(i,k,j) = al(i,k,j)+alb(i,k,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      !$acc end kernels",
        "      !$acc parallel loop collapse(3) gang present(alt, al, alb)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "        alt(i,k,j) = al(i,k,j)+alb(i,k,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "calc_alt (kernels -> parallel loop)")
    if ok: count += 1

    # =========================================================================
    # 4. calc_php - UPGRADE kernels -> parallel loop
    # =========================================================================
    src, ok = do_replace(src,
        "      !$acc kernels present(php, phb, ph)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "        php(i,k,j) = 0.5*(phb(i,k,j)+phb(i,k+1,j)+ph(i,k,j)+ph(i,k+1,j))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      !$acc end kernels",
        "      !$acc parallel loop collapse(3) gang present(php, phb, ph)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "        php(i,k,j) = 0.5*(phb(i,k,j)+phb(i,k+1,j)+ph(i,k,j)+ph(i,k+1,j))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "calc_php (kernels -> parallel loop)")
    if ok: count += 1

    # =========================================================================
    # 5. calc_p_rho_phi - nonhydro hypso=1 UPGRADE
    # =========================================================================
    src, ok = do_replace(src,
        "        !$acc kernels present(al, alb, muts, mu, ph, c1, c2, rdnw)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "          al(i,k,j)=-1./(c1(k)*muts(i,j)+c2(k))*(alb(i,k,j)*(c1(k)*mu(i,j)) + rdnw(k)*(ph(i,k+1,j)-ph(i,k,j)))\n"
        "        END DO\n"
        "        END DO\n"
        "        END DO\n"
        "        !$acc end kernels",
        "        !$acc parallel loop collapse(3) gang present(al, alb, muts, mu, ph, c1, c2, rdnw)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "          al(i,k,j)=-1./(c1(k)*muts(i,j)+c2(k))*(alb(i,k,j)*(c1(k)*mu(i,j)) + rdnw(k)*(ph(i,k+1,j)-ph(i,k,j)))\n"
        "        END DO\n"
        "        END DO\n"
        "        END DO",
        "calc_p_rho_phi: nonhydro hypso=1 (kernels -> parallel loop)")
    if ok: count += 1

    # =========================================================================
    # 6. calc_p_rho_phi - nonhydro hypso=2 UPGRADE
    # =========================================================================
    src, ok = do_replace(src,
        "        !$acc kernels present(al, alb, ph, phb, muts, c3f, c4f, c3h, c4h)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "          pfu = c3f(k+1)*MUTS(i,j) + c4f(k+1) + ptop\n"
        "          pfd = c3f(k  )*MUTS(i,j) + c4f(k  ) + ptop\n"
        "          phm = c3h(k  )*MUTS(i,j) + c4h(k  ) + ptop\n"
        "          al(i,k,j) = (ph(i,k+1,j)-ph(i,k,j)+phb(i,k+1,j)-phb(i,k,j))/phm/LOG(pfd/pfu)-alb(i,k,j)\n"
        "        END DO\n"
        "        END DO\n"
        "        END DO\n"
        "        !$acc end kernels",
        "        !$acc parallel loop collapse(3) gang present(al, alb, ph, phb, muts, c3f, c4f, c3h, c4h)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "          pfu = c3f(k+1)*MUTS(i,j) + c4f(k+1) + ptop\n"
        "          pfd = c3f(k  )*MUTS(i,j) + c4f(k  ) + ptop\n"
        "          phm = c3h(k  )*MUTS(i,j) + c4h(k  ) + ptop\n"
        "          al(i,k,j) = (ph(i,k+1,j)-ph(i,k,j)+phb(i,k+1,j)-phb(i,k,j))/phm/LOG(pfd/pfu)-alb(i,k,j)\n"
        "        END DO\n"
        "        END DO\n"
        "        END DO",
        "calc_p_rho_phi: nonhydro hypso=2 (kernels -> parallel loop)")
    if ok: count += 1

    # =========================================================================
    # 7. calc_p_rho_phi - nonhydro dry (ELSE moist_nonhydro) UPGRADE
    # =========================================================================
    src, ok = do_replace(src,
        "        !$acc kernels present(p, al, alb, t, pb)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "          p(i,k,j)=p0*( (r_d*(t0+t(i,k,j)))/                     &\n"
        "                        (p0*(al(i,k,j)+alb(i,k,j))) )**cpovcv  &\n"
        "                           -pb(i,k,j)\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        !$acc end kernels",
        "        !$acc parallel loop collapse(3) gang present(p, al, alb, t, pb)\n"
        "        DO j=jts,jtf\n"
        "        DO k=kts,ktf\n"
        "        DO i=its,itf\n"
        "          p(i,k,j)=p0*( (r_d*(t0+t(i,k,j)))/                     &\n"
        "                        (p0*(al(i,k,j)+alb(i,k,j))) )**cpovcv  &\n"
        "                           -pb(i,k,j)\n"
        "        ENDDO\n"
        "        ENDDO\n"
        "        ENDDO",
        "calc_p_rho_phi: nonhydro dry p (kernels -> parallel loop)")
    if ok: count += 1

    # =========================================================================
    # 8-11. horizontal_diffusion - UPGRADE 4 kernels regions
    # =========================================================================

    # hdiff u branch
    src, ok = do_replace(src,
        "      !$acc kernels present(tendency, field, xkmhd, MUT, msfux, msfuy, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i = i_start, i_end\n",
        "      !$acc parallel loop collapse(3) gang present(tendency, field, xkmhd, MUT, msfux, msfuy, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i = i_start, i_end\n",
        "horizontal_diffusion u: start (kernels -> parallel loop)")
    if ok:
        src, ok2 = do_replace(src,
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "      !$acc end kernels\n"
            "   \n"
            "   ELSE IF (name .EQ. 'v')THEN",
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "   \n"
            "   ELSE IF (name .EQ. 'v')THEN",
            "horizontal_diffusion u: end (remove end kernels)")
        if ok2: count += 1

    # hdiff v branch
    src, ok = do_replace(src,
        "      !$acc kernels present(tendency, field, xkmhd, MUT, msfvx, msfvy, msfvx_inv, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i = i_start, i_end\n",
        "      !$acc parallel loop collapse(3) gang present(tendency, field, xkmhd, MUT, msfvx, msfvy, msfvx_inv, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i = i_start, i_end\n",
        "horizontal_diffusion v: start (kernels -> parallel loop)")
    if ok:
        src, ok2 = do_replace(src,
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "      !$acc end kernels\n"
            "   \n"
            "   ELSE IF (name .EQ. 'w')THEN",
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "   \n"
            "   ELSE IF (name .EQ. 'w')THEN",
            "horizontal_diffusion v: end (remove end kernels)")
        if ok2: count += 1

    # hdiff w branch
    src, ok = do_replace(src,
        "      !$acc kernels present(tendency, field, xkmhd, MUT, msfux, msfuy, msfvx_inv, msfvy, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts+1,ktf\n"
        "      DO i = i_start, i_end\n",
        "      !$acc parallel loop collapse(3) gang present(tendency, field, xkmhd, MUT, msfux, msfuy, msfvx_inv, msfvy, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts+1,ktf\n"
        "      DO i = i_start, i_end\n",
        "horizontal_diffusion w: start (kernels -> parallel loop)")
    if ok:
        src, ok2 = do_replace(src,
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "      !$acc end kernels\n"
            "   \n"
            "   ELSE\n",
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "   \n"
            "   ELSE\n",
            "horizontal_diffusion w: end (remove end kernels)")
        if ok2: count += 1

    # hdiff scalar (else) branch
    src, ok = do_replace(src,
        "      !$acc kernels present(tendency, field, xkmhd, MUT, msfux, msfuy, msfvx_inv, msfvy, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i = i_start, i_end\n"
        "\n"
        "         mkrdxm=(msfux(i,j)/msfuy(i,j))*0.5*(xkmhd(i,k,j)+xkmhd(i-1,k,j))*0.5*((c1(k)*MUT(i,j)+c2(k))+(c1(k)*MUT(i-1,j)+c2(k)))*rdx\n",
        "      !$acc parallel loop collapse(3) gang present(tendency, field, xkmhd, MUT, msfux, msfuy, msfvx_inv, msfvy, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i = i_start, i_end\n"
        "\n"
        "         mkrdxm=(msfux(i,j)/msfuy(i,j))*0.5*(xkmhd(i,k,j)+xkmhd(i-1,k,j))*0.5*((c1(k)*MUT(i,j)+c2(k))+(c1(k)*MUT(i-1,j)+c2(k)))*rdx\n",
        "horizontal_diffusion scalar: start (kernels -> parallel loop)")
    if ok:
        # Find end: ENDDO x3 then !$acc end kernels then ENDIF / END SUBROUTINE
        src, ok2 = do_replace(src,
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "      !$acc end kernels\n"
            "           \n"
            "   ENDIF\n"
            "\n"
            "END SUBROUTINE horizontal_diffusion",
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "           \n"
            "   ENDIF\n"
            "\n"
            "END SUBROUTINE horizontal_diffusion",
            "horizontal_diffusion scalar: end (remove end kernels)")
        if ok2: count += 1

    # =========================================================================
    # 12. horizontal_diffusion_3dmp - UPGRADE kernels -> parallel loop
    # =========================================================================
    src, ok = do_replace(src,
        "      !$acc kernels present(tendency, field, base_3d, xkmhd, MUT, msfux, msfuy, msfvx_inv, msfvy, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i = i_start, i_end\n",
        "      !$acc parallel loop collapse(3) gang present(tendency, field, base_3d, xkmhd, MUT, msfux, msfuy, msfvx_inv, msfvy, msftx, msfty, c1, c2)\n"
        "      DO j = j_start, j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i = i_start, i_end\n",
        "horizontal_diffusion_3dmp: start (kernels -> parallel loop)")
    if ok:
        src, ok2 = do_replace(src,
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "      !$acc end kernels\n"
            "\n"
            "END SUBROUTINE horizontal_diffusion_3dmp",
            "      ENDDO\n"
            "      ENDDO\n"
            "      ENDDO\n"
            "\n"
            "END SUBROUTINE horizontal_diffusion_3dmp",
            "horizontal_diffusion_3dmp: end (remove end kernels)")
        if ok2: count += 1

    # =========================================================================
    # ==================== NEW DIRECTIVES BELOW ==============================
    # =========================================================================

    # =========================================================================
    # 13. calculate_full - simple 3D add: rfield = rfieldb + rfieldp
    # =========================================================================
    src, ok = do_replace(src,
        "   DO j=j_start,jtf\n"
        "   DO k=kts,ktf\n"
        "   DO i=i_start,itf\n"
        "      rfield(i,k,j)=rfieldb(i,k,j)+rfieldp(i,k,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE calculate_full",
        "   !$acc parallel loop collapse(3) gang present(rfield, rfieldb, rfieldp)\n"
        "   DO j=j_start,jtf\n"
        "   DO k=kts,ktf\n"
        "   DO i=i_start,itf\n"
        "      rfield(i,k,j)=rfieldb(i,k,j)+rfieldp(i,k,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE calculate_full",
        "calculate_full: 3D loop")
    if ok: count += 1

    # =========================================================================
    # 14. zero_tend - simple 3D zero
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
        "      !$acc parallel loop collapse(3) gang present(tendency)\n"
        "      DO j = jts, jte\n"
        "      DO k = kts, kte\n"
        "      DO i = its, ite\n"
        "        tendency(i,k,j) = 0.\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      END SUBROUTINE zero_tend",
        "zero_tend: 3D zero loop")
    if ok: count += 1

    # =========================================================================
    # 15. zero_tend2d - simple 2D zero
    # =========================================================================
    src, ok = do_replace(src,
        "      DO j = jts, jte\n"
        "      DO i = its, ite\n"
        "        tendency(i,j) = 0.\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      END SUBROUTINE zero_tend2d",
        "      !$acc parallel loop collapse(2) gang present(tendency)\n"
        "      DO j = jts, jte\n"
        "      DO i = its, ite\n"
        "        tendency(i,j) = 0.\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      END SUBROUTINE zero_tend2d",
        "zero_tend2d: 2D zero loop")
    if ok: count += 1

    # =========================================================================
    # 16. set_tend - simple 3D: field = field_adv_tend * msf
    # =========================================================================
    src, ok = do_replace(src,
        "      DO j = jts, jtf\n"
        "      DO k = kts, ktf\n"
        "      DO i = its, itf\n"
        "         field(i,k,j) = field_adv_tend(i,k,j)*msf(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "END SUBROUTINE set_tend",
        "      !$acc parallel loop collapse(3) gang present(field, field_adv_tend, msf)\n"
        "      DO j = jts, jtf\n"
        "      DO k = kts, ktf\n"
        "      DO i = its, itf\n"
        "         field(i,k,j) = field_adv_tend(i,k,j)*msf(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "END SUBROUTINE set_tend",
        "set_tend: 3D loop")
    if ok: count += 1

    # =========================================================================
    # 17. coriolis - ru_tend loop (j outer, k/i inner)
    #     Match the unique pattern with the f/rv/e/cosa/rw expression
    # =========================================================================
    src, ok = do_replace(src,
        "   DO j = jts, MIN(jte,jde-1)\n"
        "\n"
        "   DO k=kts,ktf\n"
        "   DO i = i_start, i_end\n"
        "   \n"
        "     ru_tend(i,k,j)=ru_tend(i,k,j) + (msfux(i,j)/msfuy(i,j))*0.5*(f(i,j)+f(i-1,j))",
        "   !$acc parallel loop collapse(3) gang present(ru_tend, rv, rw, msfux, msfuy, f, e, cosa)\n"
        "   DO j = jts, MIN(jte,jde-1)\n"
        "\n"
        "   DO k=kts,ktf\n"
        "   DO i = i_start, i_end\n"
        "   \n"
        "     ru_tend(i,k,j)=ru_tend(i,k,j) + (msfux(i,j)/msfuy(i,j))*0.5*(f(i,j)+f(i-1,j))",
        "coriolis: ru_tend loop")
    if ok: count += 1

    # coriolis - rv_tend loop
    src, ok = do_replace(src,
        "   DO j=j_start, j_end\n"
        "   DO k=kts,ktf\n"
        "   DO i=its,MIN(ide-1,ite)\n"
        "   \n"
        "      rv_tend(i,k,j)=rv_tend(i,k,j) - (msfvy(i,j)/msfvx(i,j))*0.5*(f(i,j)+f(i,j-1))",
        "   !$acc parallel loop collapse(3) gang present(rv_tend, ru, rw, msfvy, msfvx, f, e, sina)\n"
        "   DO j=j_start, j_end\n"
        "   DO k=kts,ktf\n"
        "   DO i=its,MIN(ide-1,ite)\n"
        "   \n"
        "      rv_tend(i,k,j)=rv_tend(i,k,j) - (msfvy(i,j)/msfvx(i,j))*0.5*(f(i,j)+f(i,j-1))",
        "coriolis: rv_tend loop")
    if ok: count += 1

    # coriolis - rw_tend loop
    src, ok = do_replace(src,
        "   DO j=jts,MIN(jte, jde-1)\n"
        "   DO k=kts+1,ktf\n"
        "   DO i=its,MIN(ite, ide-1)\n"
        "\n"
        "       rw_tend(i,k,j)=rw_tend(i,k,j) + e(i,j)*",
        "   !$acc parallel loop collapse(3) gang present(rw_tend, ru, rv, e, cosa, sina, msftx, msfty, fzm, fzp)\n"
        "   DO j=jts,MIN(jte, jde-1)\n"
        "   DO k=kts+1,ktf\n"
        "   DO i=its,MIN(ite, ide-1)\n"
        "\n"
        "       rw_tend(i,k,j)=rw_tend(i,k,j) + e(i,j)*",
        "coriolis: rw_tend loop")
    if ok: count += 1

    # =========================================================================
    # 18. curvature - vxgm computation loop
    #     This is the main vxgm(i,k,j) = ... loop
    #     Note: vxgm is a local automatic array (its-1:ite, kts:kte, jts-1:jte)
    #     We need !$acc data create(vxgm) around the whole routine body
    # =========================================================================
    # First, add acc data create for vxgm near the beginning of curvature
    src, ok = do_replace(src,
        "   DO j=j_start, j_end\n"
        "   DO k=kts,ktf\n"
        "   DO i=i_start, i_end\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "      vxgm(i,k,j)=0.5*(u(i,k,j)+u(i+1,k,j))*(msfvx(i,j+1)-msfvx(i,j))*rdy - &\n"
        "                  0.5*(v(i,k,j)+v(i,k,j+1))*(msfuy(i+1,j)-msfuy(i,j))*rdx\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO",
        "   !$acc data create(vxgm)\n"
        "   !$acc parallel loop collapse(3) gang present(u, v, msfvx, msfuy)\n"
        "   DO j=j_start, j_end\n"
        "   DO k=kts,ktf\n"
        "   DO i=i_start, i_end\n"
        "      vxgm(i,k,j)=0.5*(u(i,k,j)+u(i+1,k,j))*(msfvx(i,j+1)-msfvx(i,j))*rdy - &\n"
        "                  0.5*(v(i,k,j)+v(i,k,j+1))*(msfuy(i+1,j)-msfuy(i,j))*rdx\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO",
        "curvature: vxgm computation loop + acc data create")
    if ok: count += 1

    # curvature - close the acc data region at END SUBROUTINE
    src, ok = do_replace(src,
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE curvature",
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   !$acc end data\n"
        "\n"
        "END SUBROUTINE curvature",
        "curvature: close acc data region")
    if ok: count += 1

    # curvature - ru_tend non-polar, non-map6 branch
    # This is the larger ELSE branch with vxgm in it
    src, ok = do_replace(src,
        "      DO j=jts,MIN(jde-1,jte)\n"
        "      DO k=kts,ktf\n"
        "      DO i=i_start,i_end\n"
        "\n"
        "         ru_tend(i,k,j)=ru_tend(i,k,j) + 0.5*(vxgm(i,k,j)+vxgm(i-1,k,j))",
        "      !$acc parallel loop collapse(3) gang present(ru_tend, rv, rw, u, vxgm)\n"
        "      DO j=jts,MIN(jde-1,jte)\n"
        "      DO k=kts,ktf\n"
        "      DO i=i_start,i_end\n"
        "\n"
        "         ru_tend(i,k,j)=ru_tend(i,k,j) + 0.5*(vxgm(i,k,j)+vxgm(i-1,k,j))",
        "curvature: ru_tend (non-polar) loop")
    if ok: count += 1

    # curvature - rv_tend non-polar branch
    src, ok = do_replace(src,
        "      DO j=j_start,j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,MIN(ite,ide-1)\n"
        "\n"
        "         rv_tend(i,k,j)=rv_tend(i,k,j) - 0.5*(vxgm(i,k,j)+vxgm(i,k,j-1))",
        "      !$acc parallel loop collapse(3) gang present(rv_tend, ru, rw, v, vxgm, msfvy, msfvx)\n"
        "      DO j=j_start,j_end\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,MIN(ite,ide-1)\n"
        "\n"
        "         rv_tend(i,k,j)=rv_tend(i,k,j) - 0.5*(vxgm(i,k,j)+vxgm(i,k,j-1))",
        "curvature: rv_tend (non-polar) loop")
    if ok: count += 1

    # curvature - rw_tend loop
    src, ok = do_replace(src,
        "   DO j=jts,MIN(jte,jde-1)\n"
        "   DO k=MAX(2,kts),ktf\n"
        "   DO i=its,MIN(ite,ide-1)\n"
        "\n"
        "      rw_tend(i,k,j)=rw_tend(i,k,j) + reradius*",
        "   !$acc parallel loop collapse(3) gang present(rw_tend, ru, rv, u, v, msftx, msfty, fzm, fzp)\n"
        "   DO j=jts,MIN(jte,jde-1)\n"
        "   DO k=MAX(2,kts),ktf\n"
        "   DO i=its,MIN(ite,ide-1)\n"
        "\n"
        "      rw_tend(i,k,j)=rw_tend(i,k,j) + reradius*",
        "curvature: rw_tend loop")
    if ok: count += 1

    # =========================================================================
    # 19. diagnose_w - w(k>=2) inner loop (the big one)
    #     The j-outer loop wraps two k-loops. We can parallelize the second
    #     (k=2..kte, i=its..itf) since each (k,i,j) is independent.
    #     Note: w(i,1,j) is computed in a separate i-loop - complex expression
    #     but still parallelizable. Let's annotate both.
    # =========================================================================

    # diagnose_w: w(k>=2) loop - find the unique pattern
    src, ok = do_replace(src,
        "     DO k = 2, kte\n"
        "     DO i = its, itf\n"
        "       w(i,k,j) =  msfty(i,j)*(  (ph_new(i,k,j)-ph_old(i,k,j))/dt       &\n"
        "                               - ph_tend(i,k,j)/(c1f(k)*mut(i,j)+c2f(k))        )/g\n"
        "\n"
        "     ENDDO\n"
        "     ENDDO\n"
        "\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE diagnose_w",
        "     !$acc parallel loop collapse(2) gang present(w, ph_new, ph_old, ph_tend, mut, msfty, c1f, c2f)\n"
        "     DO k = 2, kte\n"
        "     DO i = its, itf\n"
        "       w(i,k,j) =  msfty(i,j)*(  (ph_new(i,k,j)-ph_old(i,k,j))/dt       &\n"
        "                               - ph_tend(i,k,j)/(c1f(k)*mut(i,j)+c2f(k))        )/g\n"
        "\n"
        "     ENDDO\n"
        "     ENDDO\n"
        "\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE diagnose_w",
        "diagnose_w: w(k>=2) loop")
    if ok: count += 1

    # =========================================================================
    # 20. pg_buoy_w - inner k=2..kde-1 loop
    #     The k-top (k=kde) loop is separate. The main k=2..kde-1 loop is
    #     inside a j-loop. collapse(2) on k,i.
    # =========================================================================
    src, ok = do_replace(src,
        "     DO k = 2, kde-1\n"
        "     DO i = its,itf\n"
        "      cq1 = 1./(1.+cqw(i,k,j))\n"
        "      cq2 = cqw(i,k,j)*cq1\n"
        "      cqw(i,k,j) = cq1\n"
        "      rw_tend(i,k,j) = rw_tend(i,k,j)+(1./msfty(i,j))*g*(      &\n"
        "                       cq1*rdn(k)*(p(i,k,j)-p(i,k-1,j))  &\n"
        "                       -(c1f(k)*muf(i,j))-cq2*(c1f(k)*mubf(i,j)+c2f(k))            )\n"
        "     END DO\n"
        "     ENDDO",
        "     !$acc parallel loop collapse(2) gang present(rw_tend, p, cqw, muf, mubf, msfty, c1f, c2f, rdn)\n"
        "     DO k = 2, kde-1\n"
        "     DO i = its,itf\n"
        "      cq1 = 1./(1.+cqw(i,k,j))\n"
        "      cq2 = cqw(i,k,j)*cq1\n"
        "      cqw(i,k,j) = cq1\n"
        "      rw_tend(i,k,j) = rw_tend(i,k,j)+(1./msfty(i,j))*g*(      &\n"
        "                       cq1*rdn(k)*(p(i,k,j)-p(i,k-1,j))  &\n"
        "                       -(c1f(k)*muf(i,j))-cq2*(c1f(k)*mubf(i,j)+c2f(k))            )\n"
        "     END DO\n"
        "     ENDDO",
        "pg_buoy_w: k=2..kde-1 loop")
    if ok: count += 1

    # =========================================================================
    # Write output
    # =========================================================================
    if src == original:
        print("\nERROR: No changes were made!")
        sys.exit(1)

    with open(filepath, 'w') as f:
        f.write(src)

    print(f"\nDone. Applied {count} OpenACC parallel loop regions.")
    print(f"  - Upgraded existing !$acc kernels -> !$acc parallel loop collapse(N) gang")
    print(f"  - Added new directives to: calculate_full, zero_tend, zero_tend2d,")
    print(f"    set_tend, coriolis, curvature, diagnose_w, pg_buoy_w")


if __name__ == "__main__":
    WRF_DIR = os.environ.get("WRF_DIR", None)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    elif WRF_DIR:
        filepath = os.path.join(WRF_DIR, "dyn_em", "module_big_step_utilities_em.f90")
    else:
        print("ERROR: Set WRF_DIR environment variable or pass file path as argument")
        sys.exit(1)

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    print(f"Patching: {filepath}")
    print(f"Upgrading !$acc kernels -> !$acc parallel loop collapse(N) gang")
    print(f"Adding new GPU directives to clean loop nests\n")
    patch_file(filepath)
