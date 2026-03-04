#!/usr/bin/env python3
"""
Patch module_em.f90 (preprocessed) to add OpenACC directives to loop nests
in the RK loop orchestrator routines.

Operates on: $WRF_DIR/dyn_em/module_em.f90

=== PORTED (new !$acc parallel loop directives) ===
  - rk_addtend_dry: 5 loop nests (ru_tend, rv_tend, rw_tend+ph_tend, t_tend, mu_tend)
  - q_diabatic_add: 2 loop nests (qv, qc)
  - q_diabatic_subtr: 2 loop nests (qv, qc)
  - calculate_phy_tend: ~20 loop nests (radiation, cumulus, shallow cu, PBL, FDDA,
                         scalar_tend, tracer_tend, IAU)
  - positive_definite_filter: 1 loop nest
  - bound_tke: 1 loop nest
  - bound_qna: 1 loop nest

=== SKIPPED (pure orchestrators — no loop nests, only CALL statements) ===
  - rk_step_prep: calls calculate_full, calc_mu_uv, couple_momentum, etc.
  - rk_tendency: calls zero_tend, advect_*, rhs_ph, coriolis, etc.
  - rk_scalar_tend: calls zero_tend, advect_scalar_*, horizontal_diffusion, etc.
  - init_zero_tendency: calls zero_tend/zero_tend2d in loop

=== SKIPPED (automatic arrays / OPTIONAL args / j-outer with i-only inner) ===
  - rk_update_scalar: uses automatic array `tendency(its:ite,kts:kte,jts:jte)`,
                       j-outer loops with separate i-inner muold/munew computation,
                       OPTIONAL advh_t/advz_t. Too complex for simple acc parallel.
  - rk_update_scalar_pd: same automatic array pattern
  - trajectory: complex scalar code with domain type, MPI calls
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

    # =========================================================================
    # 1. rk_addtend_dry — 5 loop nests, all simple array operations
    #    All arrays are already present on GPU from prior kernels.
    #    rk_step is a scalar passed by value.
    # =========================================================================

    # 1a. ru_tend loop (collapse(3))
    old = (
        "   DO j = jts,MIN(jte,jde-1)\n"
        "   DO k = kts,kte-1\n"
        "   DO i = its,ite\n"
        "     \n"
        "     IF(rk_step == 1)ru_tendf(i,k,j) = ru_tendf(i,k,j) +  u_save(i,k,j)*msfuy(i,j)\n"
        "     \n"
        "     ru_tend(i,k,j) = ru_tend(i,k,j) + ru_tendf(i,k,j)/msfuy(i,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO"
    )
    new = (
        "   !$acc parallel loop collapse(3) gang present(ru_tend, ru_tendf, u_save, msfuy)\n"
        "   DO j = jts,MIN(jte,jde-1)\n"
        "   DO k = kts,kte-1\n"
        "   DO i = its,ite\n"
        "     \n"
        "     IF(rk_step == 1)ru_tendf(i,k,j) = ru_tendf(i,k,j) +  u_save(i,k,j)*msfuy(i,j)\n"
        "     \n"
        "     ru_tend(i,k,j) = ru_tend(i,k,j) + ru_tendf(i,k,j)/msfuy(i,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO"
    )
    src, ok = do_replace(src, old, new, "rk_addtend_dry: ru_tend loop")
    if ok: count += 1

    # 1b. rv_tend loop (collapse(3))
    old = (
        "   DO j = jts,jte\n"
        "   DO k = kts,kte-1\n"
        "   DO i = its,MIN(ite,ide-1)\n"
        "     \n"
        "     IF(rk_step == 1)rv_tendf(i,k,j) = rv_tendf(i,k,j) +  v_save(i,k,j)*msfvx(i,j)\n"
        "     \n"
        "     rv_tend(i,k,j) = rv_tend(i,k,j) + rv_tendf(i,k,j)*msfvx_inv(i,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO"
    )
    new = (
        "   !$acc parallel loop collapse(3) gang present(rv_tend, rv_tendf, v_save, msfvx, msfvx_inv)\n"
        "   DO j = jts,jte\n"
        "   DO k = kts,kte-1\n"
        "   DO i = its,MIN(ite,ide-1)\n"
        "     \n"
        "     IF(rk_step == 1)rv_tendf(i,k,j) = rv_tendf(i,k,j) +  v_save(i,k,j)*msfvx(i,j)\n"
        "     \n"
        "     rv_tend(i,k,j) = rv_tend(i,k,j) + rv_tendf(i,k,j)*msfvx_inv(i,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO"
    )
    src, ok = do_replace(src, old, new, "rk_addtend_dry: rv_tend loop")
    if ok: count += 1

    # 1c. rw_tend + ph_tend loop (collapse(3))
    old = (
        "   DO j = jts,MIN(jte,jde-1)\n"
        "   DO k = kts,kte\n"
        "   DO i = its,MIN(ite,ide-1)\n"
        "     \n"
        "     IF(rk_step == 1)rw_tendf(i,k,j) = rw_tendf(i,k,j) +  w_save(i,k,j)*msfty(i,j)\n"
        "     \n"
        "     rw_tend(i,k,j) = rw_tend(i,k,j) + rw_tendf(i,k,j)/msfty(i,j)\n"
        "     IF(rk_step == 1)ph_tendf(i,k,j) = ph_tendf(i,k,j) +  ph_save(i,k,j)\n"
        "     \n"
        "     ph_tend(i,k,j) = ph_tend(i,k,j) + ph_tendf(i,k,j)/msfty(i,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO"
    )
    new = (
        "   !$acc parallel loop collapse(3) gang present(rw_tend, rw_tendf, w_save, ph_tend, ph_tendf, ph_save, msfty)\n"
        "   DO j = jts,MIN(jte,jde-1)\n"
        "   DO k = kts,kte\n"
        "   DO i = its,MIN(ite,ide-1)\n"
        "     \n"
        "     IF(rk_step == 1)rw_tendf(i,k,j) = rw_tendf(i,k,j) +  w_save(i,k,j)*msfty(i,j)\n"
        "     \n"
        "     rw_tend(i,k,j) = rw_tend(i,k,j) + rw_tendf(i,k,j)/msfty(i,j)\n"
        "     IF(rk_step == 1)ph_tendf(i,k,j) = ph_tendf(i,k,j) +  ph_save(i,k,j)\n"
        "     \n"
        "     ph_tend(i,k,j) = ph_tend(i,k,j) + ph_tendf(i,k,j)/msfty(i,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO"
    )
    src, ok = do_replace(src, old, new, "rk_addtend_dry: rw_tend+ph_tend loop")
    if ok: count += 1

    # 1d. t_tend loop (collapse(3))
    old = (
        "   DO j = jts,MIN(jte,jde-1)\n"
        "   DO k = kts,kte-1\n"
        "   DO i = its,MIN(ite,ide-1)\n"
        "     IF(rk_step == 1)t_tendf(i,k,j) = t_tendf(i,k,j) +  t_save(i,k,j)\n"
        "     \n"
        "      t_tend(i,k,j) =  t_tend(i,k,j) +  t_tendf(i,k,j)/msfty(i,j)  &\n"
        "                                     +  (c1(k)*mut(i,j)+c2(k))*h_diabatic(i,k,j)/msfty(i,j)\n"
        "     \n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO"
    )
    new = (
        "   !$acc parallel loop collapse(3) gang present(t_tend, t_tendf, t_save, msfty, c1, c2, mut, h_diabatic)\n"
        "   DO j = jts,MIN(jte,jde-1)\n"
        "   DO k = kts,kte-1\n"
        "   DO i = its,MIN(ite,ide-1)\n"
        "     IF(rk_step == 1)t_tendf(i,k,j) = t_tendf(i,k,j) +  t_save(i,k,j)\n"
        "     \n"
        "      t_tend(i,k,j) =  t_tend(i,k,j) +  t_tendf(i,k,j)/msfty(i,j)  &\n"
        "                                     +  (c1(k)*mut(i,j)+c2(k))*h_diabatic(i,k,j)/msfty(i,j)\n"
        "     \n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO"
    )
    src, ok = do_replace(src, old, new, "rk_addtend_dry: t_tend loop")
    if ok: count += 1

    # 1e. mu_tend loop (collapse(2))
    old = (
        "   DO j = jts,MIN(jte,jde-1)\n"
        "   DO i = its,MIN(ite,ide-1)\n"
        "\n"
        "      MU_TEND(i,j) =  MU_TEND(i,j) +  MU_TENDF(i,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE rk_addtend_dry"
    )
    new = (
        "   !$acc parallel loop collapse(2) gang present(mu_tend, mu_tendf)\n"
        "   DO j = jts,MIN(jte,jde-1)\n"
        "   DO i = its,MIN(ite,ide-1)\n"
        "\n"
        "      MU_TEND(i,j) =  MU_TEND(i,j) +  MU_TENDF(i,j)\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE rk_addtend_dry"
    )
    src, ok = do_replace(src, old, new, "rk_addtend_dry: mu_tend loop")
    if ok: count += 1

    # =========================================================================
    # 2. q_diabatic_add — 2 loop nests inside scalar_loop
    #    These are inside IF branches (im==p_qv, im==p_qc) so we can't collapse
    #    the outer im loop, but each inner j/k/i nest is independent.
    # =========================================================================

    # 2a. qv_diabatic loop
    old = (
        "     IF( im.eq.p_qv )THEN\n"
        "\n"
        "       DO j = jts,MIN(jte,jde-1)\n"
        "       DO k = kts,kte-1\n"
        "       DO i = its,MIN(ite,ide-1)\n"
        "         scalar_tends(i,k,j,im) = scalar_tends(i,k,j,im) + qv_diabatic(i,k,j)*(c1(k)*mut(I,J)+c2(k))\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "     ENDIF\n"
        "     IF( im.eq.p_qc )THEN\n"
        "\n"
        "       DO j = jts,MIN(jte,jde-1)\n"
        "       DO k = kts,kte-1\n"
        "       DO i = its,MIN(ite,ide-1)\n"
        "         scalar_tends(i,k,j,im) = scalar_tends(i,k,j,im) + qc_diabatic(i,k,j)*(c1(k)*mut(I,J)+c2(k))\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "     ENDIF\n"
        "\n"
        "   END DO scalar_loop\n"
        "\n"
        "END SUBROUTINE q_diabatic_add"
    )
    new = (
        "     IF( im.eq.p_qv )THEN\n"
        "\n"
        "       !$acc parallel loop collapse(3) gang present(scalar_tends, qv_diabatic, c1, c2, mut)\n"
        "       DO j = jts,MIN(jte,jde-1)\n"
        "       DO k = kts,kte-1\n"
        "       DO i = its,MIN(ite,ide-1)\n"
        "         scalar_tends(i,k,j,im) = scalar_tends(i,k,j,im) + qv_diabatic(i,k,j)*(c1(k)*mut(I,J)+c2(k))\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "     ENDIF\n"
        "     IF( im.eq.p_qc )THEN\n"
        "\n"
        "       !$acc parallel loop collapse(3) gang present(scalar_tends, qc_diabatic, c1, c2, mut)\n"
        "       DO j = jts,MIN(jte,jde-1)\n"
        "       DO k = kts,kte-1\n"
        "       DO i = its,MIN(ite,ide-1)\n"
        "         scalar_tends(i,k,j,im) = scalar_tends(i,k,j,im) + qc_diabatic(i,k,j)*(c1(k)*mut(I,J)+c2(k))\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "     ENDIF\n"
        "\n"
        "   END DO scalar_loop\n"
        "\n"
        "END SUBROUTINE q_diabatic_add"
    )
    src, ok = do_replace(src, old, new, "q_diabatic_add: qv + qc loops")
    if ok: count += 1

    # =========================================================================
    # 3. q_diabatic_subtr — same pattern
    # =========================================================================
    old = (
        "     IF( im.eq.p_qv )THEN\n"
        "\n"
        "       DO j = jts,MIN(jte,jde-1)\n"
        "       DO k = kts,kte-1\n"
        "       DO i = its,MIN(ite,ide-1)\n"
        "         scalar(i,k,j,im) = scalar(i,k,j,im) - dt*qv_diabatic(i,k,j)\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "     ENDIF\n"
        "     IF( im.eq.p_qc )THEN\n"
        "\n"
        "       DO j = jts,MIN(jte,jde-1)\n"
        "       DO k = kts,kte-1\n"
        "       DO i = its,MIN(ite,ide-1)\n"
        "         scalar(i,k,j,im) = scalar(i,k,j,im) - dt*qc_diabatic(i,k,j)\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "     ENDIF\n"
        "\n"
        "   END DO scalar_loop\n"
        "\n"
        "END SUBROUTINE q_diabatic_subtr"
    )
    new = (
        "     IF( im.eq.p_qv )THEN\n"
        "\n"
        "       !$acc parallel loop collapse(3) gang present(scalar, qv_diabatic)\n"
        "       DO j = jts,MIN(jte,jde-1)\n"
        "       DO k = kts,kte-1\n"
        "       DO i = its,MIN(ite,ide-1)\n"
        "         scalar(i,k,j,im) = scalar(i,k,j,im) - dt*qv_diabatic(i,k,j)\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "     ENDIF\n"
        "     IF( im.eq.p_qc )THEN\n"
        "\n"
        "       !$acc parallel loop collapse(3) gang present(scalar, qc_diabatic)\n"
        "       DO j = jts,MIN(jte,jde-1)\n"
        "       DO k = kts,kte-1\n"
        "       DO i = its,MIN(ite,ide-1)\n"
        "         scalar(i,k,j,im) = scalar(i,k,j,im) - dt*qc_diabatic(i,k,j)\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "       ENDDO\n"
        "     ENDIF\n"
        "\n"
        "   END DO scalar_loop\n"
        "\n"
        "END SUBROUTINE q_diabatic_subtr"
    )
    src, ok = do_replace(src, old, new, "q_diabatic_subtr: qv + qc loops")
    if ok: count += 1

    # =========================================================================
    # 4. positive_definite_filter — single loop nest
    # =========================================================================
    old = (
        "  DO j=jts,min(jte,jde-1)\n"
        "  DO k=kts,kte-1\n"
        "  DO i=its,min(ite,ide-1)\n"
        "\n"
        "    a(i,k,j) = min(1000.,max(a(i,k,j),0.))\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "\n"
        "  END SUBROUTINE positive_definite_filter"
    )
    new = (
        "  !$acc parallel loop collapse(3) gang present(a)\n"
        "  DO j=jts,min(jte,jde-1)\n"
        "  DO k=kts,kte-1\n"
        "  DO i=its,min(ite,ide-1)\n"
        "\n"
        "    a(i,k,j) = min(1000.,max(a(i,k,j),0.))\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "\n"
        "  END SUBROUTINE positive_definite_filter"
    )
    src, ok = do_replace(src, old, new, "positive_definite_filter: loop")
    if ok: count += 1

    # =========================================================================
    # 5. bound_tke — single loop nest
    # =========================================================================
    old = (
        "  DO j=jts,min(jte,jde-1)\n"
        "  DO k=kts,kte-1\n"
        "  DO i=its,min(ite,ide-1)\n"
        "    tke(i,k,j) = min(tke_upper_bound,max(tke(i,k,j),0.))\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "\n"
        "  END SUBROUTINE bound_tke"
    )
    new = (
        "  !$acc parallel loop collapse(3) gang present(tke)\n"
        "  DO j=jts,min(jte,jde-1)\n"
        "  DO k=kts,kte-1\n"
        "  DO i=its,min(ite,ide-1)\n"
        "    tke(i,k,j) = min(tke_upper_bound,max(tke(i,k,j),0.))\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "\n"
        "  END SUBROUTINE bound_tke"
    )
    src, ok = do_replace(src, old, new, "bound_tke: loop")
    if ok: count += 1

    # =========================================================================
    # 6. bound_qna — single loop nest
    # =========================================================================
    old = (
        "  DO j=jts,min(jte,jde-1)\n"
        "  DO k=kts,kte-1\n"
        "  DO i=its,min(ite,ide-1)\n"
        "    qna(i,k,j) = max(qna(i,k,j),0.)\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "\n"
        "  END SUBROUTINE bound_qna"
    )
    new = (
        "  !$acc parallel loop collapse(3) gang present(qna)\n"
        "  DO j=jts,min(jte,jde-1)\n"
        "  DO k=kts,kte-1\n"
        "  DO i=its,min(ite,ide-1)\n"
        "    qna(i,k,j) = max(qna(i,k,j),0.)\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "  ENDDO\n"
        "\n"
        "  END SUBROUTINE bound_qna"
    )
    src, ok = do_replace(src, old, new, "bound_qna: loop")
    if ok: count += 1

    # =========================================================================
    # 7. calculate_phy_tend — many loop nests coupling physics tendencies
    #    All follow the pattern: ARRAY(I,K,J) = (c1(k)*mu(I,J)+c2(k)) * ARRAY(I,K,J)
    #    These are inside IF branches (config_flags checks) that execute on CPU,
    #    but the loop bodies are GPU-safe.
    # =========================================================================

    # 7a. Radiation: RTHRATEN
    old = (
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
        "   ENDIF"
    )
    new = (
        "   IF (config_flags%ra_lw_physics .gt. 0 .or. config_flags%ra_sw_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) gang present(RTHRATEN, c1, c2, mut)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RTHRATEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHRATEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "   ENDIF"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: radiation RTHRATEN")
    if ok: count += 1

    # 7b. Cumulus: RUCUTEN, RVCUTEN, RTHCUTEN, RQVCUTEN
    old = (
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
        "      ENDDO"
    )
    new = (
        "   IF (config_flags%cu_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) gang present(RUCUTEN, RVCUTEN, RTHCUTEN, RQVCUTEN, c1, c2, mut)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUCUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUCUTEN(I,K,J)\n"
        "         RVCUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RVCUTEN(I,K,J)\n"
        "         RTHCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHCUTEN(I,K,J)\n"
        "         RQVCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVCUTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: cumulus main")
    if ok: count += 1

    # 7c. Cumulus RQCCUTEN
    old = (
        "      IF (P_QC .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQCCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQCCUTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QR .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQRCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQRCUTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QI .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQICUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQICUTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QS .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQSCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQSCUTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "   ENDIF"
    )
    new = (
        "      IF (P_QC .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQCCUTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQCCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQCCUTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QR .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQRCUTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQRCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQRCUTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QI .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQICUTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQICUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQICUTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QS .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQSCUTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQSCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQSCUTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "   ENDIF"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: cumulus QC/QR/QI/QS")
    if ok: count += 1

    # 7d. Shallow cumulus: RUSHTEN etc.
    old = (
        "   IF (config_flags%shcu_physics .gt. 0) THEN\n"
        "\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUSHTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUSHTEN(I,K,J)\n"
        "         RVSHTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RVSHTEN(I,K,J)\n"
        "         RTHSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHSHTEN(I,K,J)\n"
        "         RQVSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVSHTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO"
    )
    new = (
        "   IF (config_flags%shcu_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) gang present(RUSHTEN, RVSHTEN, RTHSHTEN, RQVSHTEN, c1, c2, mut)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUSHTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUSHTEN(I,K,J)\n"
        "         RVSHTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RVSHTEN(I,K,J)\n"
        "         RTHSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHSHTEN(I,K,J)\n"
        "         RQVSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVSHTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: shallow cu main")
    if ok: count += 1

    # 7e. Shallow cumulus QC/QR/QI/QS/QG
    old = (
        "      IF (P_QC .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQCSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQCSHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QR .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQRSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQRSHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QI .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQISHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQISHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QS .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQSSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQSSHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QG .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQGSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQGSHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "   ENDIF"
    )
    new = (
        "      IF (P_QC .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQCSHTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQCSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQCSHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QR .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQRSHTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQRSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQRSHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QI .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQISHTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQISHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQISHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QS .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQSSHTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQSSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQSSHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QG .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQGSHTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQGSHTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQGSHTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "   ENDIF"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: shallow cu QC/QR/QI/QS/QG")
    if ok: count += 1

    # 7f. PBL: RUBLTEN, RVBLTEN, RTHBLTEN
    old = (
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
        "      ENDDO"
    )
    new = (
        "   IF (config_flags%bl_pbl_physics .gt. 0) THEN\n"
        "\n"
        "      !$acc parallel loop collapse(3) gang present(RUBLTEN, RVBLTEN, RTHBLTEN, c1, c2, mut)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUBLTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RUBLTEN(I,K,J)\n"
        "         RVBLTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*RVBLTEN(I,K,J)\n"
        "         RTHBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHBLTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: PBL main")
    if ok: count += 1

    # 7g. PBL RQVBLTEN
    old = (
        "      IF (P_QV .ge. PARAM_FIRST_SCALAR) THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQVBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVBLTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QC .ge. PARAM_FIRST_SCALAR) THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "           RQCBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQCBLTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QI .ge. PARAM_FIRST_SCALAR) THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQIBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQIBLTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "    ENDIF"
    )
    new = (
        "      IF (P_QV .ge. PARAM_FIRST_SCALAR) THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQVBLTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQVBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVBLTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QC .ge. PARAM_FIRST_SCALAR) THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQCBLTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "           RQCBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQCBLTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QI .ge. PARAM_FIRST_SCALAR) THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQIBLTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQIBLTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQIBLTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "    ENDIF"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: PBL QV/QC/QI")
    if ok: count += 1

    # 7h. FDDA: RUNDGDTEN (u-staggered)
    # Need to match the FDDA block. The u-tend has itsu start, v-tend has jtsv start.
    # Note: preprocessor strips comments, leaving extra blank lines.
    old = (
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=itsu,itf\n"
        "\n"
        "\n"
        "         RUNDGDTEN(I,K,J) =(c1(k)*muu(I,J)+c2(k))*RUNDGDTEN(I,K,J)\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      DO J=jtsv,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RVNDGDTEN(I,K,J) =(c1(k)*muv(I,J)+c2(k))*RVNDGDTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "\n"
        "\n"
        "         RTHNDGDTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHNDGDTEN(I,K,J)\n"
        "\n"
        "\n"
        "\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO"
    )
    new = (
        "      !$acc parallel loop collapse(3) gang present(RUNDGDTEN, c1, c2, muu)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=itsu,itf\n"
        "\n"
        "\n"
        "         RUNDGDTEN(I,K,J) =(c1(k)*muu(I,J)+c2(k))*RUNDGDTEN(I,K,J)\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      !$acc parallel loop collapse(3) gang present(RVNDGDTEN, c1, c2, muv)\n"
        "      DO J=jtsv,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RVNDGDTEN(I,K,J) =(c1(k)*muv(I,J)+c2(k))*RVNDGDTEN(I,K,J)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      !$acc parallel loop collapse(3) gang present(RTHNDGDTEN, c1, c2, mut)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "\n"
        "\n"
        "         RTHNDGDTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHNDGDTEN(I,K,J)\n"
        "\n"
        "\n"
        "\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: FDDA u/v/th")
    if ok: count += 1

    # 7i. FDDA RQVNDGDTEN
    old = (
        "      IF (P_QV .ge. PARAM_FIRST_SCALAR) THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQVNDGDTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVNDGDTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "    ENDIF"
    )
    new = (
        "      IF (P_QV .ge. PARAM_FIRST_SCALAR) THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQVNDGDTEN, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQVNDGDTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RQVNDGDTEN(I,K,J)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "    ENDIF"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: FDDA QV")
    if ok: count += 1

    # 7j. scalar_tend coupling
    old = (
        "   DO im = PARAM_FIRST_SCALAR,num_scalar\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            scalar_tend(I,K,J,im)=(c1(k)*mut(I,J)+c2(k))*scalar_tend(I,K,J,im)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "   ENDDO\n"
        "\n"
        "   DO im = PARAM_FIRST_SCALAR,num_tracer\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            tracer_tend(I,K,J,im)=(c1(k)*mut(I,J)+c2(k))*tracer_tend(I,K,J,im)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "   ENDDO"
    )
    new = (
        "   DO im = PARAM_FIRST_SCALAR,num_scalar\n"
        "         !$acc parallel loop collapse(3) gang present(scalar_tend, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            scalar_tend(I,K,J,im)=(c1(k)*mut(I,J)+c2(k))*scalar_tend(I,K,J,im)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "   ENDDO\n"
        "\n"
        "   DO im = PARAM_FIRST_SCALAR,num_tracer\n"
        "         !$acc parallel loop collapse(3) gang present(tracer_tend, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            tracer_tend(I,K,J,im)=(c1(k)*mut(I,J)+c2(k))*tracer_tend(I,K,J,im)\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "   ENDDO"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: scalar_tend + tracer_tend coupling")
    if ok: count += 1

    # 7k. IAU main loop (RUIAUTEN etc)
    old = (
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUIAUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*U_IAU(I,K,J)*WGT_IAU\n"
        "         RVIAUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*V_IAU(I,K,J)*WGT_IAU\n"
        "         RTHIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*T_IAU(I,K,J)*WGT_IAU\n"
        "         RQVIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QV_IAU(I,K,J)*WGT_IAU\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf+1\n"
        "      DO I=its,itf\n"
        "         RPHIAUTEN(I,K,J)=PH_IAU(I,K,J)*WGT_IAU\n"
        "      ENDDO \n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      DO J=jts,jtf\n"
        "      DO I=its,itf\n"
        "         RMUIAUTEN(I,J) =MU_IAU(I,J)*WGT_IAU\n"
        "      ENDDO\n"
        "      ENDDO"
    )
    new = (
        "      !$acc parallel loop collapse(3) gang present(RUIAUTEN, RVIAUTEN, RTHIAUTEN, RQVIAUTEN, U_IAU, V_IAU, T_IAU, QV_IAU, c1, c2, mut)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf\n"
        "      DO I=its,itf\n"
        "         RUIAUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*U_IAU(I,K,J)*WGT_IAU\n"
        "         RVIAUTEN(I,K,J) =(c1(k)*mut(I,J)+c2(k))*V_IAU(I,K,J)*WGT_IAU\n"
        "         RTHIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*T_IAU(I,K,J)*WGT_IAU\n"
        "         RQVIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QV_IAU(I,K,J)*WGT_IAU\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      !$acc parallel loop collapse(3) gang present(RPHIAUTEN, PH_IAU)\n"
        "      DO J=jts,jtf\n"
        "      DO K=kts,ktf+1\n"
        "      DO I=its,itf\n"
        "         RPHIAUTEN(I,K,J)=PH_IAU(I,K,J)*WGT_IAU\n"
        "      ENDDO \n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      !$acc parallel loop collapse(2) gang present(RMUIAUTEN, MU_IAU)\n"
        "      DO J=jts,jtf\n"
        "      DO I=its,itf\n"
        "         RMUIAUTEN(I,J) =MU_IAU(I,J)*WGT_IAU\n"
        "      ENDDO\n"
        "      ENDDO"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: IAU main (u/v/th/qv/ph/mu)")
    if ok: count += 1

    # 7l. IAU QC/QR/QI/QS/QG
    old = (
        "      IF (P_QC .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQCIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QC_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QR .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQRIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QR_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QI .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQIIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QI_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QS .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQSIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QS_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QG .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQGIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QG_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "   ENDIF\n"
        "\n"
        "END SUBROUTINE calculate_phy_tend"
    )
    new = (
        "      IF (P_QC .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQCIAUTEN, QC_IAU, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQCIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QC_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QR .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQRIAUTEN, QR_IAU, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQRIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QR_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF (P_QI .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQIIAUTEN, QI_IAU, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQIIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QI_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QS .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQSIAUTEN, QS_IAU, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQSIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QS_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "      IF(P_QG .ge. PARAM_FIRST_SCALAR)THEN\n"
        "         !$acc parallel loop collapse(3) gang present(RQGIAUTEN, QG_IAU, c1, c2, mut)\n"
        "         DO J=jts,jtf\n"
        "         DO K=kts,ktf\n"
        "         DO I=its,itf\n"
        "            RQGIAUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*QG_IAU(I,K,J)*WGT_IAU\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ENDIF\n"
        "\n"
        "   ENDIF\n"
        "\n"
        "END SUBROUTINE calculate_phy_tend"
    )
    src, ok = do_replace(src, old, new, "calculate_phy_tend: IAU QC/QR/QI/QS/QG")
    if ok: count += 1

    # =========================================================================
    # Write output
    # =========================================================================
    if src == original:
        print("\nNo changes made!")
        return False

    with open(filepath, 'w') as f:
        f.write(src)

    print(f"\nDone: {count} patches applied to {filepath}")
    return True


if __name__ == "__main__":
    WRF_DIR = os.environ.get("WRF_DIR", None)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    elif WRF_DIR:
        filepath = os.path.join(WRF_DIR, "dyn_em", "module_em.f90")
    else:
        print("ERROR: Set WRF_DIR environment variable or pass file path as argument")
        sys.exit(1)

    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        sys.exit(1)

    success = patch_file(filepath)
    sys.exit(0 if success else 1)
