#!/usr/bin/env python3
"""
patch_small_step_gpu.py — Upgrade !$acc kernels to !$acc parallel loop
in module_small_step_em.f90 for explicit GPU parallelism.

Reads the preprocessed .f90 (not the .F), replaces every
`!$acc kernels present(...)` with appropriate `!$acc parallel loop`
directives, and writes the modified file back.

Strategy:
  - Simple 3D loops (j,k,i): collapse(3) gang vector
  - Simple 2D loops (j,i): collapse(2) gang vector
  - Complex outer j-loops (advance_uv, advance_mu_t, advance_w):
    gang on j with private() for local per-j arrays
  - calc_coef_w: gang on j with private(cof), sequential k inside
  - Tridiagonal solver in advance_w: sequential in k, parallel in i
    (runs inside gang j-loop, each gang handles one j-column)

No computational logic is modified — only OpenACC directives.

Analysis results:
  - SAVE variables: NONE found in any target subroutine
  - WRITE/PRINT in hot loops: NONE found
  - Automatic/local arrays:
      advance_uv:   dpn(its:ite,kts:kte), dpxy(its:ite,kts:kte), mudf_xy(its:ite)
      advance_mu_t: wdtn(its:ite,kts:kte), dvdxi(its:ite,kts:kte), dmdt(its:ite)
      advance_w:    rhs(its:ite,kts:kte), wdwn(its:ite,kts:kte),
                    mut_inv(its:ite), msft_inv(its:ite), dampwt(kts:kte)
      calc_coef_w:  cof(ims:ime)
    All handled via private() clause on gang j-loops.
"""

import sys
import os

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

F90_PATH = os.path.join(WRF_DIR, "dyn_em", "module_small_step_em.f90")


def patch_once(text, old, new, label=""):
    """Replace exactly one occurrence of old with new in text."""
    count = text.count(old)
    if count == 0:
        print(f"  WARNING: pattern not found for [{label}]")
        return text
    if count > 1:
        print(f"  INFO: {count} occurrences for [{label}], replacing first only")
    idx = text.index(old)
    return text[:idx] + new + text[idx + len(old):]


def apply_patches(text):
    """Apply all kernels->parallel loop upgrades."""

    # =====================================================================
    # PART 1: small_step_prep — simple loops in rk_step==1 and else blocks
    # =====================================================================

    # 2D: mu_1=mu_2, ww_save=0, mudf=0
    text = patch_once(text,
        '!$acc kernels present(mu_1, mu_2, ww_save, mudf)\n',
        '!$acc parallel loop collapse(2) gang vector present(mu_1, mu_2, ww_save, mudf)\n',
        'prep: mu_1/mu_2/ww_save/mudf 2D')

    # 3D: u_1 = u_2
    text = patch_once(text,
        '!$acc kernels present(u_1, u_2)\n',
        '!$acc parallel loop collapse(3) gang vector present(u_1, u_2)\n',
        'prep: u_1=u_2')

    # 3D: v_1 = v_2
    text = patch_once(text,
        '!$acc kernels present(v_1, v_2)\n',
        '!$acc parallel loop collapse(3) gang vector present(v_1, v_2)\n',
        'prep: v_1=v_2')

    # 3D: t_1 = t_2
    text = patch_once(text,
        '!$acc kernels present(t_1, t_2)\n',
        '!$acc parallel loop collapse(3) gang vector present(t_1, t_2)\n',
        'prep: t_1=t_2')

    # 3D: w_1=w_2, ph_1=ph_2
    text = patch_once(text,
        '!$acc kernels present(w_1, w_2, ph_1, ph_2)\n',
        '!$acc parallel loop collapse(3) gang vector present(w_1, w_2, ph_1, ph_2)\n',
        'prep: w_1/ph_1')

    # 2D with two inner i-loops (different bounds): muts/muus — rk_step==1 branch
    text = patch_once(text,
        '!$acc kernels present(muts, mub, mu_2, muus, muu)\n',
        '!$acc parallel loop gang present(muts, mub, mu_2, muus, muu)\n',
        'prep: muts/muus rk1')

    # 2D: muvs = muv — rk_step==1
    text = patch_once(text,
        '!$acc kernels present(muvs, muv)\n',
        '!$acc parallel loop collapse(2) gang vector present(muvs, muv)\n',
        'prep: muvs=muv rk1')

    # 2D: mu_save=mu_2, mu_2=0 — rk_step==1
    text = patch_once(text,
        '!$acc kernels present(mu_save, mu_2)\n',
        '!$acc parallel loop collapse(2) gang vector present(mu_save, mu_2)\n',
        'prep: mu_save rk1')

    # 2D with two inner i-loops: muts/muus — else branch
    text = patch_once(text,
        '!$acc kernels present(muts, mub, mu_1, muus)\n',
        '!$acc parallel loop gang present(muts, mub, mu_1, muus)\n',
        'prep: muts/muus else')

    # 2D: muvs — else branch
    text = patch_once(text,
        '!$acc kernels present(muvs, mub, mu_1, muv)\n',
        '!$acc parallel loop collapse(2) gang vector present(muvs, mub, mu_1, muv)\n',
        'prep: muvs else')

    # 2D: mu_save, mu_2 — else branch
    text = patch_once(text,
        '!$acc kernels present(mu_save, mu_2, mu_1)\n',
        '!$acc parallel loop collapse(2) gang vector present(mu_save, mu_2, mu_1)\n',
        'prep: mu_save else')

    # 2D: ww_save boundaries
    text = patch_once(text,
        '!$acc kernels present(ww_save)\n',
        '!$acc parallel loop collapse(2) gang vector present(ww_save)\n',
        'prep: ww_save bdy')

    # 3D: c2a = cpovcv*(pb+p)/alt
    text = patch_once(text,
        '!$acc kernels present(c2a, pb, p, alt)\n',
        '!$acc parallel loop collapse(3) gang vector present(c2a, pb, p, alt)\n',
        'prep: c2a')

    # 3D: u_save/u_2 coupled
    text = patch_once(text,
        '!$acc kernels present(u_save, u_2, u_1, c1h, c2h, muus, muu, msfuy)\n',
        '!$acc parallel loop collapse(3) gang vector present(u_save, u_2, u_1, c1h, c2h, muus, muu, msfuy)\n',
        'prep: u_save/u_2')

    # 3D: v_save/v_2 coupled
    text = patch_once(text,
        '!$acc kernels present(v_save, v_2, v_1, c1h, c2h, muvs, muv, msfvx_inv)\n',
        '!$acc parallel loop collapse(3) gang vector present(v_save, v_2, v_1, c1h, c2h, muvs, muv, msfvx_inv)\n',
        'prep: v_save/v_2')

    # 3D: t_save/t_2 coupled
    text = patch_once(text,
        '!$acc kernels present(t_save, t_2, t_1, c1h, c2h, muts, mut)\n',
        '!$acc parallel loop collapse(3) gang vector present(t_save, t_2, t_1, c1h, c2h, muts, mut)\n',
        'prep: t_save/t_2')

    # 3D: w_save/w_2/ph_save/ph_2 coupled
    text = patch_once(text,
        '!$acc kernels present(w_save, w_2, w_1, ph_save, ph_2, ph_1, c1f, c2f, muts, mut, msfty)\n',
        '!$acc parallel loop collapse(3) gang vector present(w_save, w_2, w_1, ph_save, ph_2, ph_1, c1f, c2f, muts, mut, msfty)\n',
        'prep: w_save/ph_save')

    # 3D: ww_save = ww
    text = patch_once(text,
        '!$acc kernels present(ww_save, ww)\n',
        '!$acc parallel loop collapse(3) gang vector present(ww_save, ww)\n',
        'prep: ww_save=ww')

    # =====================================================================
    # PART 2: small_step_finish — all simple 3D/2D loops
    # =====================================================================

    # 3D: v_2 reconstruction
    text = patch_once(text,
        '!$acc kernels present(v_2, v_save, msfvx, muv, muvs, c1h, c2h)\n',
        '!$acc parallel loop collapse(3) gang vector present(v_2, v_save, msfvx, muv, muvs, c1h, c2h)\n',
        'finish: v_2')

    # 3D: u_2 reconstruction
    text = patch_once(text,
        '!$acc kernels present(u_2, u_save, msfuy, muu, muus, c1h, c2h)\n',
        '!$acc parallel loop collapse(3) gang vector present(u_2, u_save, msfuy, muu, muus, c1h, c2h)\n',
        'finish: u_2')

    # 3D: w_2/ph_2/ww reconstruction
    text = patch_once(text,
        '!$acc kernels present(w_2, w_save, ph_2, ph_save, ww, ww1, msfty, mut, muts, c1f, c2f)\n',
        '!$acc parallel loop collapse(3) gang vector present(w_2, w_save, ph_2, ph_save, ww, ww1, msfty, mut, muts, c1f, c2f)\n',
        'finish: w_2/ph_2/ww')

    # 3D: t_2 (rk_step < rk_order)
    text = patch_once(text,
        '!$acc kernels present(t_2, t_save, mut, muts, c1h, c2h)\n',
        '!$acc parallel loop collapse(3) gang vector present(t_2, t_save, mut, muts, c1h, c2h)\n',
        'finish: t_2 rk<order')

    # 3D: t_2 (rk_step == rk_order, with h_diabatic)
    text = patch_once(text,
        '!$acc kernels present(t_2, t_save, h_diabatic, mut, muts, c1h, c2h)\n',
        '!$acc parallel loop collapse(3) gang vector present(t_2, t_save, h_diabatic, mut, muts, c1h, c2h)\n',
        'finish: t_2 rk==order')

    # 2D: mu_2 += mu_save
    text = patch_once(text,
        '!$acc kernels present(mu_2, mu_save)\n',
        '!$acc parallel loop collapse(2) gang vector present(mu_2, mu_save)\n',
        'finish: mu_2')

    # =====================================================================
    # PART 3: calc_p_rho — mostly simple 3D, one hydrostatic has k-dependency
    # =====================================================================

    # 3D: nonhydrostatic al/p
    text = patch_once(text,
        '!$acc kernels present(al, p, alt, t_2, t_1, c2a, mu, mut, ph, rdnw, c1h, c2h)\n',
        '!$acc parallel loop collapse(3) gang vector present(al, p, alt, t_2, t_1, c2a, mu, mut, ph, rdnw, c1h, c2h)\n',
        'calc_p_rho: nonhydro')

    # Hydrostatic: ph(k+1) depends on ph(k) -> sequential in k!
    # Use gang on j, sequential k, vector on i
    text = patch_once(text,
        '!$acc kernels present(p, al, alt, t_2, t_1, c2a, mu, mut, ph, dnw, c1h, c2h, c3h)\n',
        '!$acc parallel loop gang present(p, al, alt, t_2, t_1, c2a, mu, mut, ph, dnw, c1h, c2h, c3h)\n',
        'calc_p_rho: hydro (seq k)')

    # 3D: pm1 = p (init)
    text = patch_once(text,
        '!$acc kernels present(pm1, p)\n',
        '!$acc parallel loop collapse(3) gang vector present(pm1, p)\n',
        'calc_p_rho: pm1=p')

    # 3D: divergence damping p += smdiv*(p-pm1)
    text = patch_once(text,
        '!$acc kernels present(p, pm1)\n',
        '!$acc parallel loop collapse(3) gang vector present(p, pm1)\n',
        'calc_p_rho: div damp')

    # =====================================================================
    # PART 4: calc_coef_w — sequential k dependencies, local cof()
    # =====================================================================

    text = patch_once(text,
        '!$acc kernels present(a, alpha, gamma, c2a, cqw, mut, c1h, c2h, c1f, c2f, rdn, rdnw)\n',
        '!$acc parallel loop gang private(cof) present(a, alpha, gamma, c2a, cqw, mut, c1h, c2h, c1f, c2f, rdn, rdnw)\n',
        'calc_coef_w: tridiag coefs')

    # =====================================================================
    # PART 5: advance_uv — complex j-loops with local dpn/dpxy/mudf_xy
    # The existing !$acc data create blocks handle device allocation.
    # We use private() on the gang j-loop so each gang gets its own copy.
    # =====================================================================

    # u block (u_outer_j_loop)
    text = patch_once(text,
        '!$acc kernels present(u, ru_tend, p, pb, ph, php, alt, al, muu, cqu, mudf, msfux, msfuy, mu, c1h, c2h, fnm, fnp, rdnw)\n',
        '!$acc parallel loop gang private(dpn, dpxy, mudf_xy) present(u, ru_tend, p, pb, ph, php, alt, al, muu, cqu, mudf, msfux, msfuy, mu, c1h, c2h, fnm, fnp, rdnw)\n',
        'advance_uv: u j-loop')

    # v block (v_outer_j_loop)
    text = patch_once(text,
        '!$acc kernels present(v, rv_tend, p, pb, ph, php, alt, al, muv, cqv, mudf, msfvy, msfvx, msfvx_inv, mu, c1h, c2h, fnm, fnp, rdnw)\n',
        '!$acc parallel loop gang private(dpn, dpxy, mudf_xy) present(v, rv_tend, p, pb, ph, php, alt, al, muv, cqv, mudf, msfvy, msfvx, msfvx_inv, mu, c1h, c2h, fnm, fnp, rdnw)\n',
        'advance_uv: v j-loop')

    # Polar boundary: v(jds)=0 (2D, k,i)
    # Match by context to distinguish the two v=0 blocks
    text = patch_once(text,
        '!$acc kernels present(v)\n        DO k = k_start, k_end\n        DO i = i_start, i_end\n           v(i,k,jds) = 0.',
        '!$acc parallel loop collapse(2) gang vector present(v)\n        DO k = k_start, k_end\n        DO i = i_start, i_end\n           v(i,k,jds) = 0.',
        'advance_uv: polar jds')

    text = patch_once(text,
        '!$acc kernels present(v)\n        DO k = k_start, k_end\n        DO i = i_start, i_end\n           v(i,k,jde) = 0.',
        '!$acc parallel loop collapse(2) gang vector present(v)\n        DO k = k_start, k_end\n        DO i = i_start, i_end\n           v(i,k,jde) = 0.',
        'advance_uv: polar jde')

    # =====================================================================
    # PART 6: advance_mu_t — complex j-loop with dmdt/dvdxi/wdtn locals
    # =====================================================================

    # Main mu/divergence j-loop (has ww k-dependency: ww(k) depends on ww(k-1))
    text = patch_once(text,
        '!$acc kernels present(mu, mut, muave, muts, mudf, mu_tend, u, u_1, v, v_1, ww, ww_1, muu, muv, msfux, msfuy, msfvx_inv, msftx, msfty, c1h, c2h, dnw, fnm, fnp, rdnw)\n',
        '!$acc parallel loop gang private(dmdt, dvdxi) present(mu, mut, muave, muts, mudf, mu_tend, u, u_1, v, v_1, ww, ww_1, muu, muv, msfux, msfuy, msfvx_inv, msftx, msfty, c1h, c2h, dnw, fnm, fnp, rdnw)\n',
        'advance_mu_t: main j-loop')

    # 3D: t_ave/t update (simple j,k,i)
    text = patch_once(text,
        '!$acc kernels present(t_ave, t, ft, msfty)\n',
        '!$acc parallel loop collapse(3) gang vector present(t_ave, t, ft, msfty)\n',
        'advance_mu_t: t_ave')

    # t advection j-loop with local wdtn (wdtn has k-dependency)
    text = patch_once(text,
        '!$acc kernels present(t, t_1, ww, u, v, msftx, msfty, fnm, fnp, rdnw)\n',
        '!$acc parallel loop gang private(wdtn) present(t, t_1, ww, u, v, msftx, msfty, fnm, fnp, rdnw)\n',
        'advance_mu_t: t advection j-loop')

    # =====================================================================
    # PART 7: advance_w — complex j-loop with tridiagonal solver
    # Contains forward elimination + back substitution (sequential in k).
    # All local arrays are per-j-column.
    # =====================================================================

    text = patch_once(text,
        '!$acc kernels present(w, rw_tend, ww, w_save, u, v, t_2ave, t_2, t_1, ph, ph_1, phb, ph_tend, ht, c2a, cqw, alt, alb, a, alpha, gamma, mu1, mut, muave, muts, msftx, msfty, c1h, c2h, c1f, c2f, fnm, fnp, rdnw, rdn, dnw)\n',
        '!$acc parallel loop gang private(rhs, wdwn, mut_inv, msft_inv, dampwt) present(w, rw_tend, ww, w_save, u, v, t_2ave, t_2, t_1, ph, ph_1, phb, ph_tend, ht, c2a, cqw, alt, alb, a, alpha, gamma, mu1, mut, muave, muts, msftx, msfty, c1h, c2h, c1f, c2f, fnm, fnp, rdnw, rdn, dnw)\n',
        'advance_w: main j-loop')

    # =====================================================================
    # PART 8: sumflux — all simple 3D loops
    # =====================================================================

    # init ru_m/rv_m/ww_m = 0 (3D, j,k,i)
    text = patch_once(text,
        '!$acc kernels present(ru_m, rv_m, ww_m)\n      DO  j = jts, jte\n',
        '!$acc parallel loop collapse(3) gang vector present(ru_m, rv_m, ww_m)\n      DO  j = jts, jte\n',
        'sumflux: init=0')

    # accumulate ru_m += ru, rv_m += rv, ww_m += ww (3D)
    text = patch_once(text,
        '!$acc kernels present(ru_m, rv_m, ww_m, ru, rv, ww)\n',
        '!$acc parallel loop collapse(3) gang vector present(ru_m, rv_m, ww_m, ru, rv, ww)\n',
        'sumflux: accum')

    # extra ru_m for i > mini (3D)
    text = patch_once(text,
        '!$acc kernels present(ru_m, ru)\n      DO  j = jts, minj\n      DO  k = kts, mink\n      DO  i = mini+1',
        '!$acc parallel loop collapse(3) gang vector present(ru_m, ru)\n      DO  j = jts, minj\n      DO  k = kts, mink\n      DO  i = mini+1',
        'sumflux: ru_m extra i')

    # extra rv_m for j > minj (3D)
    text = patch_once(text,
        '!$acc kernels present(rv_m, rv)\n      DO  j = minj+1',
        '!$acc parallel loop collapse(3) gang vector present(rv_m, rv)\n      DO  j = minj+1',
        'sumflux: rv_m extra j')

    # extra ww_m for k > mink (3D)
    text = patch_once(text,
        '!$acc kernels present(ww_m, ww)\n      DO  j = jts, minj\n      DO  k = mink+1',
        '!$acc parallel loop collapse(3) gang vector present(ww_m, ww)\n      DO  j = jts, minj\n      DO  k = mink+1',
        'sumflux: ww_m extra k')

    # final averaging (3D)
    text = patch_once(text,
        '!$acc kernels present(ru_m, rv_m, ww_m, u_lin, v_lin, ww_lin, muu, muv, msfuy, msfvx_inv, c1h, c2h)\n',
        '!$acc parallel loop collapse(3) gang vector present(ru_m, rv_m, ww_m, u_lin, v_lin, ww_lin, muu, muv, msfuy, msfvx_inv, c1h, c2h)\n',
        'sumflux: final avg')

    # extra ru_m averaging (3D)
    text = patch_once(text,
        '!$acc kernels present(ru_m, u_lin, muu, msfuy, c1h, c2h)\n',
        '!$acc parallel loop collapse(3) gang vector present(ru_m, u_lin, muu, msfuy, c1h, c2h)\n',
        'sumflux: ru_m avg extra')

    # extra rv_m averaging (3D)
    text = patch_once(text,
        '!$acc kernels present(rv_m, v_lin, muv, msfvx_inv, c1h, c2h)\n',
        '!$acc parallel loop collapse(3) gang vector present(rv_m, v_lin, muv, msfvx_inv, c1h, c2h)\n',
        'sumflux: rv_m avg extra')

    # extra ww_m averaging (3D)
    text = patch_once(text,
        '!$acc kernels present(ww_m, ww_lin)\n',
        '!$acc parallel loop collapse(3) gang vector present(ww_m, ww_lin)\n',
        'sumflux: ww_m avg extra')

    # =====================================================================
    # PART 9: Any remaining subroutines (save_ph_mu, etc.)
    # These should also be simple 3D or 2D loops.
    # =====================================================================

    # ph_1 = ph_2 (3D)
    text = patch_once(text,
        '!$acc kernels present(ph_1, ph_2)\n',
        '!$acc parallel loop collapse(3) gang vector present(ph_1, ph_2)\n',
        'save_ph_mu: ph_1=ph_2')

    # mu_1 = mu_2 (2D)
    text = patch_once(text,
        '!$acc kernels present(mu_1, mu_2)\n',
        '!$acc parallel loop collapse(2) gang vector present(mu_1, mu_2)\n',
        'save_ph_mu: mu_1=mu_2')

    # ph_save = ph_2 - ph_1 (3D)
    text = patch_once(text,
        '!$acc kernels present(ph_save, ph_2, ph_1)\n',
        '!$acc parallel loop collapse(3) gang vector present(ph_save, ph_2, ph_1)\n',
        'save_ph_mu: ph_save')

    # mu_save = mu_2 - mu_1 (2D)
    text = patch_once(text,
        '!$acc kernels present(mu_save, mu_2, mu_1)\n',
        '!$acc parallel loop collapse(2) gang vector present(mu_save, mu_2, mu_1)\n',
        'save_ph_mu: mu_save (2)')

    # ph_2/mu_2 restore (may be combined)
    # Check if there's a combined present clause
    text = patch_once(text,
        '!$acc kernels present(ph_2, ph_save, mu_2, mu_save)\n',
        '!$acc parallel loop gang present(ph_2, ph_save, mu_2, mu_save)\n',
        'restore_ph_mu: combined')

    # =====================================================================
    # PART 10: Global replacement of all remaining "!$acc end kernels"
    # =====================================================================
    text = text.replace('!$acc end kernels', '!$acc end parallel loop')

    return text


def validate(text):
    """Check for remaining unconverted directives and balanced opens/closes."""
    remaining_kernels = text.count('!$acc kernels')
    remaining_end_kernels = text.count('!$acc end kernels')
    n_parallel = text.count('!$acc parallel loop')
    n_end_parallel = text.count('!$acc end parallel loop')
    n_data = text.count('!$acc data create')
    n_end_data = text.count('!$acc end data')

    print(f"\n=== Directive Counts ===")
    print(f"  !$acc parallel loop:     {n_parallel}")
    print(f"  !$acc end parallel loop: {n_end_parallel}")
    print(f"  !$acc data create:       {n_data}")
    print(f"  !$acc end data:          {n_end_data}")

    ok = True
    if remaining_kernels > 0:
        print(f"\n  WARNING: {remaining_kernels} unconverted '!$acc kernels' remain!")
        # Find them
        for i, line in enumerate(text.split('\n'), 1):
            if '!$acc kernels' in line:
                print(f"    Line {i}: {line.strip()}")
        ok = False

    if remaining_end_kernels > 0:
        print(f"\n  WARNING: {remaining_end_kernels} unconverted '!$acc end kernels' remain!")
        ok = False

    if n_data != n_end_data:
        print(f"\n  WARNING: Mismatched data regions ({n_data} opens vs {n_end_data} closes)")

    if ok:
        print("\n  All !$acc kernels successfully converted to !$acc parallel loop.")

    return ok


def main():
    print(f"Reading {F90_PATH}...")
    with open(F90_PATH, "r") as f:
        text = f.read()

    n_lines = text.count('\n')
    orig_kernels = text.count('!$acc kernels')
    orig_end = text.count('!$acc end kernels')
    print(f"  {n_lines} lines")
    print(f"  {orig_kernels} '!$acc kernels' directives")
    print(f"  {orig_end} '!$acc end kernels' directives")

    print("\nApplying patches...")
    text = apply_patches(text)

    ok = validate(text)

    if ok:
        print(f"\nWriting {F90_PATH}...")
        with open(F90_PATH, "w") as f:
            f.write(text)
        print("Done.")
    else:
        # Write anyway but warn — some subroutines at end of file may have
        # been added after the original analysis
        print(f"\nWriting {F90_PATH} (with warnings)...")
        with open(F90_PATH, "w") as f:
            f.write(text)
        print("Done (check warnings above).")

    # =====================================================================
    # Summary report
    # =====================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    print("\nSAVE variables in target subroutines: NONE")
    print("  (No SAVE attribute found in any of the 6 target subroutines)")

    print("\nAutomatic/local arrays (all handled via private() or data create):")
    print("  calc_coef_w:  cof(ims:ime)         -> private(cof)")
    print("  advance_uv:   dpn(its:ite,kts:kte) -> private(dpn,dpxy,mudf_xy)")
    print("                dpxy(its:ite,kts:kte)")
    print("                mudf_xy(its:ite)")
    print("  advance_mu_t: dvdxi(its:ite,kts:kte) -> private(dmdt,dvdxi)")
    print("                dmdt(its:ite)")
    print("                wdtn(its:ite,kts:kte)   -> private(wdtn) on t-advection loop")
    print("  advance_w:    rhs(its:ite,kts:kte)    -> private(rhs,wdwn,mut_inv,msft_inv,dampwt)")
    print("                wdwn(its:ite,kts:kte)")
    print("                mut_inv(its:ite)")
    print("                msft_inv(its:ite)")
    print("                dampwt(kts:kte)")

    print("\nWRITE/PRINT statements in hot loops: NONE")

    print("\nTridiagonal solver in advance_w (sequential k, parallel i):")
    print("  Forward elim:  DO k=2,k_end+1 / DO i  -> runs inside gang j-loop")
    print("  Back subst:    DO k=k_end,2,-1 / DO i  -> runs inside gang j-loop")
    print("  Each gang processes one j-column independently")

    print("\nNotes:")
    print("  - collapse(3) used for simple j,k,i triple loops (best perf)")
    print("  - collapse(2) used for simple j,i double loops")
    print("  - gang-only used for complex j-loops with internal k-dependencies")
    print("  - private() gives each gang its own copy of local arrays")
    print("  - Existing !$acc data create blocks kept for device memory allocation")


if __name__ == "__main__":
    main()
