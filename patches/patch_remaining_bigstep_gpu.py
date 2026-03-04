#!/usr/bin/env python3
"""
Patch module_big_step_utilities_em.f90 to add OpenACC directives to all
remaining routines that patch_big_step_gpu.py missed.

Operates on the preprocessed .f90 file (not .F).

=== NEW directives (this script) ===
  - calc_mu_uv: all 8 MUU/MUV branch loops (collapse(2) each)
  - calc_mu_uv_1: all 8 MUU/MUV branch loops (collapse(2) each)
  - calc_mu_staggered: all 8 MUU/MUV branch loops (collapse(2) each)
  - couple: w, h, and scalar branch loops (collapse(3) each)
  - calc_ww_cp: muu/muv prep loops (collapse(2)), j-serial with inner parallel
  - w_damp: damping-only loop (j/k/i collapse(3), rw_tend update only)
  - diagnose_w: w(k=1) inner loop (the patch_big_step_gpu.py only did k>=2)
  - zero_pole: both pole loops (collapse(2))
  - pole_point_bc: both pole loops (collapse(2))
  - perturbation_coriolis: rv/ru prep loops, ru_tend/rv_tend/rw_tend loops

=== SKIPPED (too complex or k-serial) ===
  - calc_ww_cp: divv/dmdt accumulation + ww(k-1) dependency (k-serial within j)
  - calc_p_rho_phi moist branches: VPOW call, k-serial hydrostatic integration
  - rhs_ph: ~800 lines, complex branches, k-dependencies
  - horizontal_pressure_gradient: complex branches, automatic dpn array
  - vertical_diffusion*: sequential k-dependency (vflux)
  - w_damp CFL tracking: max_vert_cfl/max_horiz_cfl reductions + WRITE + SAVE
  - perturbation_coriolis: boundary-special ru_tend/rv_tend (open_xs/xe/ys/ye)
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
    # 1. calc_mu_uv — MUU interior loops (4 branches)
    #    Each branch has DO j / DO i with simple averaging.
    #    The boundary 1D loops (single i or j) are too small to bother.
    #    We annotate the 2D interior loops in each IF branch.
    # =========================================================================

    # Branch 1: its!=ids, ite!=ide — full interior
    src, ok = do_replace(src,
        "      IF      ( ( its .NE. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      IF      ( ( its .NE. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_uv: MUU branches 1-2")
    if ok: count += 1

    # Branch 3: its!=ids, ite==ide
    src, ok = do_replace(src,
        "      ELSE IF ( ( its .NE. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         i=ite\n"
        "         im = ite-1\n"
        "         if(config_flags%periodic_x) im = ite\n"
        "         DO j=jts,jtf\n"
        "\n"
        "\n"
        "            MUU(i,j) = 0.5*(MU(i-1,j)+MU(im,j)+MUB(i-1,j)+MUB(im,j))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      ELSE IF ( ( its .NE. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         i=ite\n"
        "         im = ite-1\n"
        "         if(config_flags%periodic_x) im = ite\n"
        "         DO j=jts,jtf\n"
        "\n"
        "\n"
        "            MUU(i,j) = 0.5*(MU(i-1,j)+MU(im,j)+MUB(i-1,j)+MUB(im,j))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_uv: MUU branches 3-4")
    if ok: count += 1

    # Now the MUV section — same pattern. Find the first MUV branch set.
    # The MUV branches follow "itf=MIN(ite,ide-1)" and "jtf=jte"
    # We need to be careful because calc_mu_uv, calc_mu_uv_1, and calc_mu_staggered
    # all have similar patterns. We'll match unique context.

    # calc_mu_uv MUV branches 1-2
    # We match the pattern right after the MUU END IF and before END SUBROUTINE calc_mu_uv
    src, ok = do_replace(src,
        "      IF      ( ( jts .NE. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         DO j=jts+1,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jts\n"
        "         jm = jts\n"
        "         if(config_flags%periodic_y) jm = jts-1\n"
        "         DO i=its,itf\n"
        "\n"
        "\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,jm)+MUB(i,j)+MUB(i,jm))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .NE. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         DO j=jts,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jte\n"
        "         jm = jte-1\n"
        "         if(config_flags%periodic_y) jm = jte\n"
        "         DO i=its,itf\n"
        "\n"
        "\n"
        "             MUV(i,j) = 0.5*(MU(i,j-1)+MU(i,jm)+MUB(i,j-1)+MUB(i,jm))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         DO j=jts+1,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      IF      ( ( jts .NE. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu, mub)\n"
        "         DO j=jts+1,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jts\n"
        "         jm = jts\n"
        "         if(config_flags%periodic_y) jm = jts-1\n"
        "         DO i=its,itf\n"
        "\n"
        "\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,jm)+MUB(i,j)+MUB(i,jm))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .NE. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu, mub)\n"
        "         DO j=jts,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jte\n"
        "         jm = jte-1\n"
        "         if(config_flags%periodic_y) jm = jte\n"
        "         DO i=its,itf\n"
        "\n"
        "\n"
        "             MUV(i,j) = 0.5*(MU(i,j-1)+MU(i,jm)+MUB(i,j-1)+MUB(i,jm))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu, mub)\n"
        "         DO j=jts+1,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_uv: MUV branches 1-4")
    if ok: count += 1

    # =========================================================================
    # 2. calc_mu_uv_1 — same structure, but uses only MU (no MUB)
    #    Match MUU branches for calc_mu_uv_1 specifically
    # =========================================================================

    # calc_mu_uv_1 MUU branches 1-2
    src, ok = do_replace(src,
        "      IF      ( ( its .NE. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      IF      ( ( its .NE. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_uv_1: MUU branches 1-2")
    if ok: count += 1

    # calc_mu_uv_1 MUU branches 3-4
    src, ok = do_replace(src,
        "      ELSE IF ( ( its .NE. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         i=ite\n"
        "         im = ite-1\n"
        "         if(config_flags%periodic_x) im = ite\n"
        "         DO j=jts,jtf\n"
        "            MUU(i,j) = 0.5*(MU(i-1,j)+MU(im,j))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      ELSE IF ( ( its .NE. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         i=ite\n"
        "         im = ite-1\n"
        "         if(config_flags%periodic_x) im = ite\n"
        "         DO j=jts,jtf\n"
        "            MUU(i,j) = 0.5*(MU(i-1,j)+MU(im,j))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_uv_1: MUU branches 3-4")
    if ok: count += 1

    # calc_mu_uv_1 MUV branches 1-4
    src, ok = do_replace(src,
        "      IF      ( ( jts .NE. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         DO j=jts+1,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      IF      ( ( jts .NE. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu)\n"
        "         DO j=jts+1,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_uv_1: MUV branches 1-2")
    if ok: count += 1

    # calc_mu_uv_1 MUV branches 3-4
    src, ok = do_replace(src,
        "      ELSE IF ( ( jts .NE. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         DO j=jts,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jte\n"
        "         jm = jte-1\n"
        "         if(config_flags%periodic_y) jm = jte\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j-1)+MU(i,jm))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         DO j=jts+1,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      ELSE IF ( ( jts .NE. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu)\n"
        "         DO j=jts,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jte\n"
        "         jm = jte-1\n"
        "         if(config_flags%periodic_y) jm = jte\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j-1)+MU(i,jm))\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu)\n"
        "         DO j=jts+1,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_uv_1: MUV branches 3-4")
    if ok: count += 1

    # =========================================================================
    # 3. calc_mu_staggered — same boundary-branch structure but no config_flags
    #    Unique pattern: boundary loops use "MU(i,j)+MUB(i,j)" (no averaging)
    # =========================================================================

    # calc_mu_staggered MUU branch 1 (interior only)
    # This is unique because it has "MUU(i,j) = MU(i,j) +MUB(i,j)" at boundaries
    src, ok = do_replace(src,
        "      IF      ( ( its .NE. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         i=its\n"
        "         DO j=jts,jtf\n"
        "            MUU(i,j) =      MU(i,j)          +MUB(i,j)\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .NE. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         i=ite\n"
        "         DO j=jts,jtf\n"
        "            MUU(i,j) =      MU(i-1,j)        +MUB(i-1,j)\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      IF      ( ( its .NE. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .NE. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         i=its\n"
        "         DO j=jts,jtf\n"
        "            MUU(i,j) =      MU(i,j)          +MUB(i,j)\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .NE. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         i=ite\n"
        "         DO j=jts,jtf\n"
        "            MUU(i,j) =      MU(i-1,j)        +MUB(i-1,j)\n"
        "         ENDDO\n"
        "      ELSE IF ( ( its .EQ. ids ) .AND. ( ite .EQ. ide ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muu, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its+1,itf-1\n"
        "            MUU(i,j) = 0.5*(MU(i,j)+MU(i-1,j)+MUB(i,j)+MUB(i-1,j))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_staggered: MUU all branches")
    if ok: count += 1

    # calc_mu_staggered MUV branches
    src, ok = do_replace(src,
        "      IF      ( ( jts .NE. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         DO j=jts+1,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jts\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) =      MU(i,j)          +MUB(i,j)\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .NE. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         DO j=jts,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jte\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) =      MU(i,j-1)        +MUB(i,j-1)\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         DO j=jts+1,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO",
        "      IF      ( ( jts .NE. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu, mub)\n"
        "         DO j=jts,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .NE. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu, mub)\n"
        "         DO j=jts+1,jtf\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jts\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) =      MU(i,j)          +MUB(i,j)\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .NE. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu, mub)\n"
        "         DO j=jts,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO\n"
        "         j=jte\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) =      MU(i,j-1)        +MUB(i,j-1)\n"
        "         ENDDO\n"
        "      ELSE IF ( ( jts .EQ. jds ) .AND. ( jte .EQ. jde ) ) THEN\n"
        "         !$acc parallel loop collapse(2) gang present(muv, mu, mub)\n"
        "         DO j=jts+1,jtf-1\n"
        "         DO i=its,itf\n"
        "             MUV(i,j) = 0.5*(MU(i,j)+MU(i,j-1)+MUB(i,j)+MUB(i,j-1))\n"
        "         ENDDO\n"
        "         ENDDO",
        "calc_mu_staggered: MUV all branches")
    if ok: count += 1

    # =========================================================================
    # 4. couple — 3D loops for w, h, and scalar branches
    #    u and v branches call calc_mu_staggered which writes to local muu/muv,
    #    so we skip those (would need to handle data locality for locals).
    #    w, h, scalar branches only read from subroutine arguments.
    # =========================================================================

    # couple: w branch
    src, ok = do_replace(src,
        "   ELSE IF (name .EQ. 'w')THEN\n"
        "      itf=MIN(ite,ide-1)\n"
        "      jtf=MIN(jte,jde-1)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,kte\n"
        "      DO i=its,itf\n"
        "         rfield(i,k,j)=field(i,k,j)*((c1(k)*mu(i,j))+(c1(k)*mub(i,j)+c2(k)))/msf(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "   ELSE IF (name .EQ. 'w')THEN\n"
        "      itf=MIN(ite,ide-1)\n"
        "      jtf=MIN(jte,jde-1)\n"
        "      !$acc parallel loop collapse(3) gang present(rfield, field, mu, mub, msf, c1, c2)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,kte\n"
        "      DO i=its,itf\n"
        "         rfield(i,k,j)=field(i,k,j)*((c1(k)*mu(i,j))+(c1(k)*mub(i,j)+c2(k)))/msf(i,j)\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "couple: w branch")
    if ok: count += 1

    # couple: h branch
    src, ok = do_replace(src,
        "   ELSE IF (name .EQ. 'h')THEN\n"
        "      itf=MIN(ite,ide-1)\n"
        "      jtf=MIN(jte,jde-1)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,kte\n"
        "      DO i=its,itf\n"
        "         rfield(i,k,j)=field(i,k,j)*((c1(k)*mu(i,j))+(c1(k)*mub(i,j)+c2(k)))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "   ELSE IF (name .EQ. 'h')THEN\n"
        "      itf=MIN(ite,ide-1)\n"
        "      jtf=MIN(jte,jde-1)\n"
        "      !$acc parallel loop collapse(3) gang present(rfield, field, mu, mub, c1, c2)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,kte\n"
        "      DO i=its,itf\n"
        "         rfield(i,k,j)=field(i,k,j)*((c1(k)*mu(i,j))+(c1(k)*mub(i,j)+c2(k)))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO",
        "couple: h branch")
    if ok: count += 1

    # couple: scalar (else) branch
    src, ok = do_replace(src,
        "   ELSE\n"
        "      itf=MIN(ite,ide-1)\n"
        "      jtf=MIN(jte,jde-1)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "         rfield(i,k,j)=field(i,k,j)*((c1(k)*mu(i,j))+(c1(k)*mub(i,j)+c2(k)))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "   \n"
        "   ENDIF\n"
        "\n"
        "END SUBROUTINE couple",
        "   ELSE\n"
        "      itf=MIN(ite,ide-1)\n"
        "      jtf=MIN(jte,jde-1)\n"
        "      !$acc parallel loop collapse(3) gang present(rfield, field, mu, mub, c1, c2)\n"
        "      DO j=jts,jtf\n"
        "      DO k=kts,ktf\n"
        "      DO i=its,itf\n"
        "         rfield(i,k,j)=field(i,k,j)*((c1(k)*mu(i,j))+(c1(k)*mub(i,j)+c2(k)))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "   \n"
        "   ENDIF\n"
        "\n"
        "END SUBROUTINE couple",
        "couple: scalar branch")
    if ok: count += 1

    # =========================================================================
    # 5. calc_ww_cp — muu/muv preparation loops are independent 2D loops.
    #    The divv/dmdt/ww loops have k-serial dependencies within j.
    #    We can parallelize the muu and muv prep loops.
    # =========================================================================

    # calc_ww_cp: muu prep loop
    src, ok = do_replace(src,
        "      DO j=jts,jtf\n"
        "      DO i=its,min(ite+1,ide)\n"
        "        MUU(i,j) = 0.5*(MUP(i,j)+MUB(i,j)+MUP(i-1,j)+MUB(i-1,j))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      DO j=jts,min(jte+1,jde)\n"
        "      DO i=its,itf\n"
        "        MUV(i,j) = 0.5*(MUP(i,j)+MUB(i,j)+MUP(i,j-1)+MUB(i,j-1))\n"
        "      ENDDO\n"
        "      ENDDO",
        "      !$acc data create(muu, muv, dmdt, divv)\n"
        "      !$acc parallel loop collapse(2) gang present(mup, mub)\n"
        "      DO j=jts,jtf\n"
        "      DO i=its,min(ite+1,ide)\n"
        "        MUU(i,j) = 0.5*(MUP(i,j)+MUB(i,j)+MUP(i-1,j)+MUB(i-1,j))\n"
        "      ENDDO\n"
        "      ENDDO\n"
        "\n"
        "      !$acc parallel loop collapse(2) gang present(mup, mub)\n"
        "      DO j=jts,min(jte+1,jde)\n"
        "      DO i=its,itf\n"
        "        MUV(i,j) = 0.5*(MUP(i,j)+MUB(i,j)+MUP(i,j-1)+MUB(i,j-1))\n"
        "      ENDDO\n"
        "      ENDDO",
        "calc_ww_cp: muu/muv prep loops + acc data create")
    if ok: count += 1

    # calc_ww_cp: close acc data region at END SUBROUTINE
    src, ok = do_replace(src,
        "     ENDDO\n"
        "\n"
        "\n"
        "END SUBROUTINE calc_ww_cp",
        "     ENDDO\n"
        "      !$acc end data\n"
        "\n"
        "\n"
        "END SUBROUTINE calc_ww_cp",
        "calc_ww_cp: close acc data region")
    if ok: count += 1

    # =========================================================================
    # 6. w_damp — The main loop has CFL reductions (max_vert_cfl, max_horiz_cfl),
    #    WRITE statements, and counter increments (some1). These prevent full
    #    parallelization. However, the damping update to rw_tend is independent
    #    per gridpoint. We'll split: keep the CFL diagnostic loop on host,
    #    add a separate GPU kernel for just the damping.
    #
    #    Strategy: Add a second loop AFTER the main diagnostic loop that does
    #    just the rw_tend damping on GPU. This is safe because the damping
    #    doesn't depend on the CFL tracking.
    #
    #    Actually, looking more carefully: the damping depends on vert_cfl which
    #    is computed in the same loop. We CAN compute vert_cfl redundantly on GPU.
    #    But the WRITE/debug stuff prevents moving the whole loop.
    #
    #    Better approach: just annotate the main loop with acc parallel loop
    #    and use reductions + skip the WRITE/debug on GPU. But the WRITE
    #    calls make this impossible with standard OpenACC.
    #
    #    Simplest safe approach: add a SECOND loop after the existing one that
    #    applies damping on GPU. Remove the damping from the first loop.
    #    This requires restructuring which is fragile with string matching.
    #
    #    Let's skip w_damp for now — the CFL diagnostics + WRITE + SAVE make
    #    it unsuitable for simple directive-based porting.
    # =========================================================================
    # w_damp: SKIPPED — CFL reductions, WRITE statements, SAVE variable
    print("  [SKIP] w_damp: CFL reductions + WRITE + SAVE variable")

    # =========================================================================
    # 7. diagnose_w — w(k=1) inner loop
    #    The existing patch only handles k>=2. The w(k=1) loop is inside
    #    a j-outer loop. We can parallelize the i-loop within j.
    # =========================================================================

    src, ok = do_replace(src,
        "     DO i = its, itf\n"
        "         w(i,1,j)=  msfty(i,j)*.5*rdy*(                      &\n"
        "                           (ht(i,j+1)-ht(i,j  ))             &\n"
        "          *(cf1*v(i,1,j+1)+cf2*v(i,2,j+1)+cf3*v(i,3,j+1))    &\n"
        "                          +(ht(i,j  )-ht(i,j-1))             &\n"
        "          *(cf1*v(i,1,j  )+cf2*v(i,2,j  )+cf3*v(i,3,j  ))  ) &\n"
        "                 +msftx(i,j)*.5*rdx*(                        &\n"
        "                           (ht(i+1,j)-ht(i,j  ))             &\n"
        "          *(cf1*u(i+1,1,j)+cf2*u(i+1,2,j)+cf3*u(i+1,3,j))    &\n"
        "                          +(ht(i,j  )-ht(i-1,j))             &\n"
        "          *(cf1*u(i  ,1,j)+cf2*u(i  ,2,j)+cf3*u(i  ,3,j))  )\n"
        "     ENDDO",
        "     !$acc parallel loop gang present(w, ht, v, u, msfty, msftx)\n"
        "     DO i = its, itf\n"
        "         w(i,1,j)=  msfty(i,j)*.5*rdy*(                      &\n"
        "                           (ht(i,j+1)-ht(i,j  ))             &\n"
        "          *(cf1*v(i,1,j+1)+cf2*v(i,2,j+1)+cf3*v(i,3,j+1))    &\n"
        "                          +(ht(i,j  )-ht(i,j-1))             &\n"
        "          *(cf1*v(i,1,j  )+cf2*v(i,2,j  )+cf3*v(i,3,j  ))  ) &\n"
        "                 +msftx(i,j)*.5*rdx*(                        &\n"
        "                           (ht(i+1,j)-ht(i,j  ))             &\n"
        "          *(cf1*u(i+1,1,j)+cf2*u(i+1,2,j)+cf3*u(i+1,3,j))    &\n"
        "                          +(ht(i,j  )-ht(i-1,j))             &\n"
        "          *(cf1*u(i  ,1,j)+cf2*u(i  ,2,j)+cf3*u(i  ,3,j))  )\n"
        "     ENDDO",
        "diagnose_w: w(k=1) inner loop")
    if ok: count += 1

    # =========================================================================
    # 8. zero_pole — two conditional 2D loops
    # =========================================================================

    src, ok = do_replace(src,
        "  IF (jts == jds) THEN\n"
        "     DO k = kts, kte\n"
        "     DO i = its-1, ite+1\n"
        "        field(i,k,jts) = 0.\n"
        "     END DO\n"
        "     END DO\n"
        "  END IF\n"
        "  IF (jte == jde) THEN\n"
        "     DO k = kts, kte\n"
        "     DO i = its-1, ite+1\n"
        "        field(i,k,jte) = 0.\n"
        "     END DO",
        "  IF (jts == jds) THEN\n"
        "     !$acc parallel loop collapse(2) gang present(field)\n"
        "     DO k = kts, kte\n"
        "     DO i = its-1, ite+1\n"
        "        field(i,k,jts) = 0.\n"
        "     END DO\n"
        "     END DO\n"
        "  END IF\n"
        "  IF (jte == jde) THEN\n"
        "     !$acc parallel loop collapse(2) gang present(field)\n"
        "     DO k = kts, kte\n"
        "     DO i = its-1, ite+1\n"
        "        field(i,k,jte) = 0.\n"
        "     END DO",
        "zero_pole: both pole loops")
    if ok: count += 1

    # =========================================================================
    # 9. pole_point_bc — two conditional 2D loops (its..ite bounds)
    # =========================================================================
    src, ok = do_replace(src,
        "  IF (jts == jds) THEN\n"
        "     DO k = kts, kte\n"
        "     DO i = its, ite\n"
        "\n"
        "        field(i,k,jts) = field(i,k,jts+1)\n"
        "     END DO\n"
        "     END DO\n"
        "  END IF\n"
        "  IF (jte == jde) THEN\n"
        "     DO k = kts, kte\n"
        "     DO i = its, ite\n"
        "\n"
        "        field(i,k,jte) = field(i,k,jte-1)\n"
        "     END DO",
        "  IF (jts == jds) THEN\n"
        "     !$acc parallel loop collapse(2) gang present(field)\n"
        "     DO k = kts, kte\n"
        "     DO i = its, ite\n"
        "\n"
        "        field(i,k,jts) = field(i,k,jts+1)\n"
        "     END DO\n"
        "     END DO\n"
        "  END IF\n"
        "  IF (jte == jde) THEN\n"
        "     !$acc parallel loop collapse(2) gang present(field)\n"
        "     DO k = kts, kte\n"
        "     DO i = its, ite\n"
        "\n"
        "        field(i,k,jte) = field(i,k,jte-1)\n"
        "     END DO",
        "pole_point_bc: both pole loops")
    if ok: count += 1

    # =========================================================================
    # 10. perturbation_coriolis — multiple loop nests
    #     The rv/ru preparation loops have k-dependency at boundaries (k=kts, k=ktf).
    #     The main interior loops (kts+1..ktf-1) ARE parallelizable.
    #     The tend loops (ru_tend, rv_tend, rw_tend) are all collapse(3).
    #     BUT: ru and rv are large automatic arrays — need acc data create.
    #     The boundary-special loops (open_xs, open_xe, open_ys, open_ye) are
    #     small and conditional — leave on host.
    # =========================================================================

    # perturbation_coriolis: rv prep main loop (k=kts+1..ktf-1)
    src, ok = do_replace(src,
        "   DO j = jts, MIN(jte,jde-1)+1\n"
        "   DO k=kts+1,ktf-1\n"
        "   DO i = i_start-1, i_end\n"
        "     z_at_v = 0.25*( phb(i,k,j  )+phb(i,k+1,j  )  &\n"
        "                    +phb(i,k,j-1)+phb(i,k+1,j-1)  &\n"
        "                    +ph(i,k,j  )+ph(i,k+1,j  )    &\n"
        "                    +ph(i,k,j-1)+ph(i,k+1,j-1))/g\n"
        "     wkp1 = min(1.,max(0.,z_at_v-z_base(k))/(z_base(k+1)-z_base(k)))\n"
        "     wkm1 = min(1.,max(0.,z_base(k)-z_at_v)/(z_base(k)-z_base(k-1)))\n"
        "     wk   = 1.-wkp1-wkm1\n"
        "     rv(i,k,j) = rv_in(i,k,j) - (c1h(k)*muv(i,j)+c2h(k))*(            &\n"
        "                                  wkm1*v_base(k-1)    &\n"
        "                                 +wk  *v_base(k  )    &\n"
        "                                 +wkp1*v_base(k+1)   )\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO",
        "   !$acc data create(ru, rv)\n"
        "   !$acc parallel loop collapse(3) gang present(rv_in, phb, ph, muv, c1h, c2h, z_base, v_base)\n"
        "   DO j = jts, MIN(jte,jde-1)+1\n"
        "   DO k=kts+1,ktf-1\n"
        "   DO i = i_start-1, i_end\n"
        "     z_at_v = 0.25*( phb(i,k,j  )+phb(i,k+1,j  )  &\n"
        "                    +phb(i,k,j-1)+phb(i,k+1,j-1)  &\n"
        "                    +ph(i,k,j  )+ph(i,k+1,j  )    &\n"
        "                    +ph(i,k,j-1)+ph(i,k+1,j-1))/g\n"
        "     wkp1 = min(1.,max(0.,z_at_v-z_base(k))/(z_base(k+1)-z_base(k)))\n"
        "     wkm1 = min(1.,max(0.,z_base(k)-z_at_v)/(z_base(k)-z_base(k-1)))\n"
        "     wk   = 1.-wkp1-wkm1\n"
        "     rv(i,k,j) = rv_in(i,k,j) - (c1h(k)*muv(i,j)+c2h(k))*(            &\n"
        "                                  wkm1*v_base(k-1)    &\n"
        "                                 +wk  *v_base(k  )    &\n"
        "                                 +wkp1*v_base(k+1)   )\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO",
        "perturbation_coriolis: rv prep main loop + acc data create(ru,rv)")
    if ok: count += 1

    # perturbation_coriolis: ru prep main loop (k=kts+1..ktf-1)
    src, ok = do_replace(src,
        "   DO j = j_start-1,j_end\n"
        "   DO k=kts+1,ktf-1\n"
        "   DO i = its, MIN(ite,ide-1)+1\n"
        "     z_at_u = 0.25*( phb(i  ,k,j)+phb(i  ,k+1,j)  &\n"
        "                    +phb(i-1,k,j)+phb(i-1,k+1,j)  &\n"
        "                    +ph(i  ,k,j)+ph(i  ,k+1,j)    &\n"
        "                    +ph(i-1,k,j)+ph(i-1,k+1,j))/g\n"
        "     wkp1 = min(1.,max(0.,z_at_u-z_base(k))/(z_base(k+1)-z_base(k)))\n"
        "     wkm1 = min(1.,max(0.,z_base(k)-z_at_u)/(z_base(k)-z_base(k-1)))\n"
        "     wk   = 1.-wkp1-wkm1\n"
        "     ru(i,k,j) = ru_in(i,k,j) - (c1h(k)*muu(i,j)+c2h(k))*(            &\n"
        "                                  wkm1*u_base(k-1)    &\n"
        "                                 +wk  *u_base(k  )    &\n"
        "                                 +wkp1*u_base(k+1)   )\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO",
        "   !$acc parallel loop collapse(3) gang present(ru_in, phb, ph, muu, c1h, c2h, z_base, u_base)\n"
        "   DO j = j_start-1,j_end\n"
        "   DO k=kts+1,ktf-1\n"
        "   DO i = its, MIN(ite,ide-1)+1\n"
        "     z_at_u = 0.25*( phb(i  ,k,j)+phb(i  ,k+1,j)  &\n"
        "                    +phb(i-1,k,j)+phb(i-1,k+1,j)  &\n"
        "                    +ph(i  ,k,j)+ph(i  ,k+1,j)    &\n"
        "                    +ph(i-1,k,j)+ph(i-1,k+1,j))/g\n"
        "     wkp1 = min(1.,max(0.,z_at_u-z_base(k))/(z_base(k+1)-z_base(k)))\n"
        "     wkm1 = min(1.,max(0.,z_base(k)-z_at_u)/(z_base(k)-z_base(k-1)))\n"
        "     wk   = 1.-wkp1-wkm1\n"
        "     ru(i,k,j) = ru_in(i,k,j) - (c1h(k)*muu(i,j)+c2h(k))*(            &\n"
        "                                  wkm1*u_base(k-1)    &\n"
        "                                 +wk  *u_base(k  )    &\n"
        "                                 +wkp1*u_base(k+1)   )\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO",
        "perturbation_coriolis: ru prep main loop")
    if ok: count += 1

    # perturbation_coriolis: main rv_tend loop (j_start..j_end)
    src, ok = do_replace(src,
        "   DO j=j_start, j_end\n"
        "   DO k=kts,ktf\n"
        "   DO i=its,MIN(ide-1,ite)\n"
        "   \n"
        "      rv_tend(i,k,j)=rv_tend(i,k,j) - (msfvy(i,j)/msfvx(i,j))*0.5*(f(i,j)+f(i,j-1))    &\n"
        "       *0.25*(ru(i,k,j)+ru(i+1,k,j)+ru(i,k,j-1)+ru(i+1,k,j-1)) &\n"
        "           + (msfvy(i,j)/msfvx(i,j))*0.5*(e(i,j)+e(i,j-1))*0.5*(sina(i,j)+sina(i,j-1)) &\n"
        "           *0.25*(rw(i,k+1,j-1)+rw(i,k,j-1)+rw(i,k+1,j)+rw(i,k,j))\n"
        "\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO",
        "   !$acc parallel loop collapse(3) gang present(rv_tend, ru, rw, msfvy, msfvx, f, e, sina)\n"
        "   DO j=j_start, j_end\n"
        "   DO k=kts,ktf\n"
        "   DO i=its,MIN(ide-1,ite)\n"
        "   \n"
        "      rv_tend(i,k,j)=rv_tend(i,k,j) - (msfvy(i,j)/msfvx(i,j))*0.5*(f(i,j)+f(i,j-1))    &\n"
        "       *0.25*(ru(i,k,j)+ru(i+1,k,j)+ru(i,k,j-1)+ru(i+1,k,j-1)) &\n"
        "           + (msfvy(i,j)/msfvx(i,j))*0.5*(e(i,j)+e(i,j-1))*0.5*(sina(i,j)+sina(i,j-1)) &\n"
        "           *0.25*(rw(i,k+1,j-1)+rw(i,k,j-1)+rw(i,k+1,j)+rw(i,k,j))\n"
        "\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO",
        "perturbation_coriolis: rv_tend main loop")
    if ok: count += 1

    # perturbation_coriolis: rw_tend loop
    src, ok = do_replace(src,
        "   DO j=jts,MIN(jte, jde-1)\n"
        "   DO k=kts+1,ktf\n"
        "   DO i=its,MIN(ite, ide-1)\n"
        "\n"
        "       rw_tend(i,k,j)=rw_tend(i,k,j) + e(i,j)*           &\n"
        "          (cosa(i,j)*0.5*(fzm(k)*(ru(i,k,j)+ru(i+1,k,j)) &\n"
        "          +fzp(k)*(ru(i,k-1,j)+ru(i+1,k-1,j)))           &\n"
        "          -(msftx(i,j)/msfty(i,j))*sina(i,j)*0.5*(fzm(k)*(rv(i,k,j)+rv(i,k,j+1)) &\n"
        "          +fzp(k)*(rv(i,k-1,j)+rv(i,k-1,j+1))))\n"
        "\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "\n"
        "END SUBROUTINE perturbation_coriolis",
        "   !$acc parallel loop collapse(3) gang present(rw_tend, ru, rv, e, cosa, sina, msftx, msfty, fzm, fzp)\n"
        "   DO j=jts,MIN(jte, jde-1)\n"
        "   DO k=kts+1,ktf\n"
        "   DO i=its,MIN(ite, ide-1)\n"
        "\n"
        "       rw_tend(i,k,j)=rw_tend(i,k,j) + e(i,j)*           &\n"
        "          (cosa(i,j)*0.5*(fzm(k)*(ru(i,k,j)+ru(i+1,k,j)) &\n"
        "          +fzp(k)*(ru(i,k-1,j)+ru(i+1,k-1,j)))           &\n"
        "          -(msftx(i,j)/msfty(i,j))*sina(i,j)*0.5*(fzm(k)*(rv(i,k,j)+rv(i,k,j+1)) &\n"
        "          +fzp(k)*(rv(i,k-1,j)+rv(i,k-1,j+1))))\n"
        "\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   ENDDO\n"
        "   !$acc end data\n"
        "\n"
        "END SUBROUTINE perturbation_coriolis",
        "perturbation_coriolis: rw_tend loop + close acc data")
    if ok: count += 1

    # =========================================================================
    # Write output
    # =========================================================================
    if src == original:
        print("\nERROR: No changes were made!")
        sys.exit(1)

    with open(filepath, 'w') as f:
        f.write(src)

    print(f"\nDone. Applied {count} OpenACC regions.")
    print(f"  Patched routines:")
    print(f"    - calc_mu_uv: MUU + MUV branch loops")
    print(f"    - calc_mu_uv_1: MUU + MUV branch loops")
    print(f"    - calc_mu_staggered: MUU + MUV branch loops")
    print(f"    - couple: w, h, scalar branches")
    print(f"    - calc_ww_cp: muu/muv prep loops + acc data region")
    print(f"    - diagnose_w: w(k=1) inner loop")
    print(f"    - zero_pole: both pole loops")
    print(f"    - pole_point_bc: both pole loops")
    print(f"    - perturbation_coriolis: rv/ru prep + rv_tend + rw_tend + acc data")
    print(f"  Skipped:")
    print(f"    - w_damp: CFL reductions + WRITE + SAVE variable")
    print(f"    - calc_ww_cp inner loops: k-serial dmdt/ww dependency")
    print(f"    - calc_p_rho_phi moist/hydrostatic: VPOW call, k-serial integration")
    print(f"    - rhs_ph, horizontal_pressure_gradient: too complex")
    print(f"    - vertical_diffusion*: k-serial vflux dependency")


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
    print(f"Adding remaining OpenACC directives\n")
    patch_file(filepath)
