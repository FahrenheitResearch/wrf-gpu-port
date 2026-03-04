#!/usr/bin/env python3
"""
Comprehensive GPU data management patch for module_diffusion_em.f90.

Adds !$acc data create() directives for ALL local arrays in EVERY subroutine
that has stack-allocated work arrays. This prevents per-kernel implicit
copy-in/copy-out of large temporary arrays, which is the main bottleneck
for GPU execution of diffusion code.

Previous patch (patch_diffusion_locals.py) only covered 9 "top-level" subs.
This patch covers ALL 18 subroutines that need it, including:
  - Deep call-tree leaves (cal_titau_12_21, cal_titau_13_31, cal_titau_23_32)
  - TKE subroutines (tke_km, tke_dissip, tke_shear)
  - Turbulence closures (smag_km, smag2d_km, cal_dampkm)
  - Top-level dispatchers (vertical_diffusion_2)
  - Long-scale subs (meso_length_scale, free_atmos_length)
  - Misc (cal_deform_and_div, calculate_N2, nonlocal_flux, cal_helicity)
  - vertical_diffusion_implicit (23 arrays!)

Special cases handled:
  - TRIDIAG: has !$acc routine seq — local array q(n) lives on thread stack,
    NOT patched (data create would be wrong for routine seq)
  - NONLOCAL_FLUX: includes LOGICAL arrays (pblflg, sfcflg, stable)
  - CAL_HELICITY: includes LOGICAL array (use_column)
  - Already-patched subs: skipped (idempotent)
  - Fixes missing rravg in vertical_diffusion_s data create (old patch bug)

Call tree ordering (leaves first, then callers):
  LEAF: cal_deform_and_div, cal_dampkm, calculate_N2, smag_km, smag2d_km,
        cal_titau_12_21, cal_titau_13_31, cal_titau_23_32, nonlocal_flux,
        tke_shear, cal_helicity, meso_length_scale, free_atmos_length
  calc_l_scale <- tke_km, tke_dissip
  cal_titau_* <- horizontal_diffusion_{u,v,w}_2 (already patched)
  vertical_diffusion_{u,v,w,s}_2 <- vertical_diffusion_2
  tridiag <- vertical_diffusion_implicit

Author: GPU port automation
Date: 2026-03-03
"""

import re
import sys
import os

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

TARGET = os.path.join(WRF_DIR, "dyn_em", "module_diffusion_em.f90")

# Each entry: (sub_name, anchor_pattern, [local_array_names])
# anchor_pattern: whitespace-insensitive match for the first executable statement.
# Arrays are listed in declaration order for readability.
#
# NOTE: TRIDIAG is deliberately excluded — it has !$acc routine seq,
# so its local array q(n) is thread-private stack memory.

SUBROUTINES = [
    # =========================================================================
    # GROUP 1: Leaf subroutines (no internal calls)
    # =========================================================================

    # cal_deform_and_div: 6 local arrays (3x 2D, 3x 3D)
    # Called by: solve_em (external)
    ("cal_deform_and_div",
     "ktes1   = kte-1",
     ["mm", "zzavg", "zeta_zd12",    # 2D: (its:ite, jts:jte)
      "tmp1", "hat", "hatavg"]),       # 3D: (its-2:ite+2, kts:kte, jts-2:jte+2)

    # cal_dampkm: 3 local arrays
    # Called by: calculate_km_kh
    ("cal_dampkm",
     "ktf = min(kte,kde-1)",
     ["deltaz",                        # 1D: (its:ite)
      "dampk", "dampkv"]),             # 3D: (its:ite, kts:kte, jts:jte)

    # calculate_N2: 5 local arrays
    # Called by: calculate_km_kh
    ("calculate_N2",
     "qc_cr   = 0.00001",
     ["tmp1sfc", "tmp1top",            # 2D: (its:ite, jts:jte)
      "tmp1", "qvs", "qctmp"]),        # 3D: (its:ite, kts:kte, jts:jte)

    # smag_km: 1 local array
    # Called by: calculate_km_kh
    ("smag_km",
     "ktf = min(kte,kde-1)",
     ["def2"]),                         # 3D: (its:ite, kts:kte, jts:jte)

    # smag2d_km: 1 local array
    # Called by: calculate_km_kh
    ("smag2d_km",
     "ktf = min(kte,kde-1)",
     ["def2"]),                         # 3D: (its:ite, kts:kte, jts:jte)

    # cal_titau_12_21: 2 local arrays
    # Called by: horizontal_diffusion_u_2, horizontal_diffusion_v_2
    ("cal_titau_12_21",
     "ktf = MIN( kte, kde-1 )",
     ["xkxavg", "rhoavg"]),            # 3D: (its:ite, kts:kte, jts:jte)

    # cal_titau_13_31: 2 local arrays
    # Called by: horizontal_diffusion_w_2, vertical_diffusion_u_2
    ("cal_titau_13_31",
     "ktf = MIN( kte, kde-1 )",
     ["xkxavg", "rhoavg"]),            # 3D: (its:ite, kts:kte, jts:jte)

    # cal_titau_23_32: 2 local arrays
    # Called by: horizontal_diffusion_w_2, vertical_diffusion_v_2
    ("cal_titau_23_32",
     "ktf = MIN( kte, kde-1 )",
     ["xkxavg", "rhoavg"]),            # 3D: (its:ite, kts:kte, jts:jte)

    # meso_length_scale: 10 local arrays (7x 3D, 3x 2D)
    # Called by: external (module_bl_mynn)
    ("meso_length_scale",
     "ktf     = MIN( kte, kde-1 )",
     ["zfull", "za", "elb", "qtke",    # 3D
      "els", "elf", "dthrdn",          # 3D
      "sflux", "elt", "vsc"]),         # 2D: (its:ite, jts:jte)

    # free_atmos_length: 5 local arrays (all 3D)
    # Called by: external
    ("free_atmos_length",
     "ktf     = MIN( kte, kde-1 )",
     ["zfull", "za",                   # 3D
      "dlg", "dlu", "dld"]),           # 3D

    # nonlocal_flux: 19 REAL + 3 LOGICAL local arrays = 22 total
    # Called by: vertical_diffusion_implicit (indirectly via calculate_km_kh)
    # Has dense loop nests, benefits hugely from GPU data create
    ("nonlocal_flux",
     "ktf=MIN(kte,kde-1)",
     ["zq", "za", "thv",              # 3D
      "zfacmf", "entfacmf",           # 3D
      "govrth", "sflux", "wstar3",    # 2D
      "wstar", "rigs",                # 2D
      "pblflg", "sfcflg", "stable",   # 2D LOGICAL
      "deltaoh", "we", "enlfrac2",    # 2D
      "hfxpbl", "bfxpbl",             # 2D
      "dthv", "wm2",                  # 2D
      "wscale", "thermal"]),           # 2D

    # cal_helicity: 6 REAL + 1 LOGICAL local arrays
    # Called by: external (solve_em)
    ("cal_helicity",
     "ktes1   = kte-1",
     ["mm",                            # 2D: (its-3:ite+2, jts-3:jte+2)
      "tmp1", "hat", "hatavg",         # 3D
      "wavg", "rvort",                 # 3D
      "use_column"]),                  # 2D LOGICAL

    # =========================================================================
    # GROUP 2: Subroutines that call other module subs
    # =========================================================================

    # tke_km: 5 local arrays. Calls calc_l_scale (no local arrays).
    # Called by: calculate_km_kh
    ("tke_km",
     "ktf     = MIN( kte, kde-1 )",
     ["l_scale", "dthrdn",            # 3D
      "def2", "xkmh_s", "xkhh_s"]),   # 3D

    # tke_dissip: 4 local arrays. Calls calc_l_scale.
    # Called by: tke_rhs
    ("tke_dissip",
     "c_k = config_flags%c_k",
     ["dthrdn", "l_scale",            # 3D
      "sumtke", "sumtkez"]),           # 2D: (its:ite, jts:jte)

    # tke_shear: 7 local arrays. Leaf.
    # Called by: tke_rhs
    ("tke_shear",
     "ktf    = MIN( kte, kde-1 )",
     ["avg", "titau",                  # 3D
      "tmp2", "titau12",              # 3D
      "tmp1",                          # 3D
      "zxavg", "zyavg"]),             # 3D

    # vertical_diffusion_2: 1 local array. Dispatcher — calls vert_diff_{u,v,w,s}_2.
    # Called by: solve_em
    ("vertical_diffusion_2",
     "i_start = its",
     ["var_mix"]),                     # 3D: (ims:ime, kms:kme, jms:jme)

    # vertical_diffusion_implicit: 23 local arrays. Calls tridiag (!$acc routine seq).
    # This is the biggest single sub — 4 groups of arrays with different bounds.
    # Called by: solve_em
    ("vertical_diffusion_implicit",
     "ktf=MIN(kte,kde-1)",
     ["var_mix",                       # (ims:ime, kms:kme, jms:jme)
      "xkxavg_m", "xkxavg_s",         # (its-1:ite+1, kts:kte, jts-1:jte+1)
      "xkxavg", "xkxavg_w",           # (its:ite, kts:kte, jts:jte)
      "rhoavg", "nlflux_rho",         # (its:ite, kts:kte, jts:jte)
      "gamvavg", "gamuavg",            # (its-1:ite+1, jts-1:jte+1)
      "tao_xz", "tao_yz",             # (its-1:ite+1, jts-1:jte+1)
      "muavg_u", "muavg_v",           # (its-1:ite+1, kts:kte, jts-1:jte+1)
      "rdz_u", "rdz_v",               # (its-1:ite+1, kts:kte, jts-1:jte+1)
      "a", "b", "c", "d",             # 1D: (kts:kte)
      "a1", "b1", "c1", "d1"]),       # 1D: (kts:kte-1)
]

# Fix for existing patch: vertical_diffusion_s is missing rravg in its data create.
# We'll fix this separately since it already has a data create directive.
FIX_EXISTING = [
    # (sub_name, old_create_pattern, new_create_line)
    ("vertical_diffusion_s",
     "!$acc data create(H3, xkxavg, tmptendf)",
     "   !$acc data create(H3, xkxavg, rravg, tmptendf)"),
]


def find_subroutine_bounds(lines, sub_name):
    """Find start and end line indices for a subroutine."""
    sub_start = None
    sub_end = None
    target = "SUBROUTINE " + sub_name.upper()

    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if target in stripped and not stripped.startswith("END"):
            if sub_start is None:
                sub_start = i
        if sub_start is not None and sub_end is None:
            if stripped.startswith("END SUBROUTINE"):
                # Match either END SUBROUTINE or END SUBROUTINE name
                if sub_name.upper() in stripped or stripped == "END SUBROUTINE":
                    sub_end = i
                    break

    return sub_start, sub_end


def find_anchor(lines, sub_start, sub_end, anchor_text):
    """Find the first executable statement matching anchor_text (whitespace-insensitive)."""
    anchor_clean = anchor_text.replace(" ", "")
    for i in range(sub_start, sub_end):
        if anchor_clean in lines[i].replace(" ", ""):
            return i
    return None


def find_end_data_line(lines, sub_start, sub_end, anchor_line):
    """Find the right place for !$acc end data (before RETURN or END SUBROUTINE)."""
    # Walk backwards from end to find RETURN or END SUBROUTINE
    for i in range(sub_end - 1, anchor_line, -1):
        stripped = lines[i].strip().upper()
        if stripped == "RETURN" or stripped.startswith("END SUBROUTINE"):
            return i
    return sub_end


def build_create_directive(local_arrays, indent="   "):
    """Build !$acc data create() directive, with continuation lines if needed."""
    if not local_arrays:
        return ""

    vars_str = ", ".join(local_arrays)
    full_line = "{}!$acc data create({})".format(indent, vars_str)

    if len(full_line) <= 80:
        return full_line + "\n"

    # Need continuation lines — split at ~60 chars per line
    parts = []
    current = []
    current_len = 0
    for v in local_arrays:
        if current_len + len(v) + 2 > 50 and current:
            parts.append(", ".join(current))
            current = [v]
            current_len = len(v)
        else:
            current.append(v)
            current_len += len(v) + 2
    if current:
        parts.append(", ".join(current))

    result = "{}!$acc data create({}, &\n".format(indent, parts[0])
    for p in parts[1:-1]:
        result += "{}!$acc&  {}, &\n".format(indent, p)
    result += "{}!$acc&  {})\n".format(indent, parts[-1])
    return result


def main():
    with open(TARGET) as f:
        lines = f.readlines()

    text = "".join(lines)

    # =========================================================================
    # Step 0: Idempotency check
    # =========================================================================
    # Check for a distinctive marker from this comprehensive patch.
    # We use cal_titau_12_21 as marker since the old patch never touched it.
    marker = "cal_titau_12_21"
    marker_found = False
    for entry in SUBROUTINES:
        if entry[0] == marker:
            # Check if this sub already has data create
            s, e = find_subroutine_bounds(lines, marker)
            if s is not None and e is not None:
                sub_text = "".join(lines[s:e])
                if "!$acc data create" in sub_text.lower():
                    marker_found = True
            break

    if marker_found:
        print("Comprehensive patch already applied (found data create in {}). Skipping.".format(marker))
        return

    # =========================================================================
    # Step 1: Fix existing patches (missing variables)
    # =========================================================================
    for sub_name, old_pattern, new_line in FIX_EXISTING:
        found = False
        for i, line in enumerate(lines):
            if old_pattern in line:
                lines[i] = new_line + "\n"
                found = True
                print("FIXED: {} — added missing rravg to existing data create".format(sub_name))
                break
        if not found:
            # Check if already fixed
            for i, line in enumerate(lines):
                if "rravg" in line and "data create" in line.lower():
                    sub_s, sub_e = find_subroutine_bounds(lines, sub_name)
                    if sub_s is not None and sub_s <= i <= (sub_e or len(lines)):
                        print("SKIP FIX: {} — rravg already present".format(sub_name))
                        found = True
                        break
            if not found:
                print("WARNING: Could not find old pattern for fix in {}".format(sub_name))

    # =========================================================================
    # Step 2: Apply new data create directives
    # =========================================================================
    # Process in REVERSE order by line number so insertions don't shift
    # subsequent line numbers.

    # First, collect all patch points
    patches = []  # (anchor_line, end_data_line, sub_name, create_directive)

    for sub_name, anchor_text, local_arrays in SUBROUTINES:
        sub_start, sub_end = find_subroutine_bounds(lines, sub_name)
        if sub_start is None or sub_end is None:
            print("WARNING: Could not find subroutine {} (start={}, end={})".format(
                sub_name, sub_start, sub_end))
            continue

        # Check if already has data create
        sub_text = "".join(lines[sub_start:sub_end])
        if "!$acc data create" in sub_text.lower():
            print("SKIP: {} — already has !$acc data create".format(sub_name))
            continue

        # Find anchor
        anchor_line = find_anchor(lines, sub_start, sub_end, anchor_text)
        if anchor_line is None:
            print("WARNING: Could not find anchor '{}' in {}".format(anchor_text, sub_name))
            continue

        # Verify arrays exist in sub text (sanity check)
        existing = [v for v in local_arrays if re.search(r'\b' + v + r'\b', sub_text)]
        missing = [v for v in local_arrays if v not in existing]
        if missing:
            print("WARNING: {} — arrays not found in source: {}".format(sub_name, ", ".join(missing)))
            # Only include existing arrays
            local_arrays = existing

        if not local_arrays:
            print("SKIP: {} — no matching local arrays".format(sub_name))
            continue

        # Find end data placement
        end_data_line = find_end_data_line(lines, sub_start, sub_end, anchor_line)

        # Build directive
        create_dir = build_create_directive(local_arrays)

        patches.append((anchor_line, end_data_line, sub_name, create_dir, local_arrays))

    if not patches:
        print("\nNo patches to apply.")
        return

    # Sort by anchor_line DESCENDING so we patch from bottom to top
    patches.sort(key=lambda x: x[0], reverse=True)

    patched_count = 0
    for anchor_line, end_data_line, sub_name, create_dir, local_arrays in patches:
        # Insert !$acc end data before the end point
        lines.insert(end_data_line, "   !$acc end data\n")
        # Insert !$acc data create before the anchor (with blank line before)
        lines.insert(anchor_line, create_dir)
        lines.insert(anchor_line, "\n")

        patched_count += 1
        print("PATCHED: {} — {} arrays: {}".format(
            sub_name, len(local_arrays), ", ".join(local_arrays)))

    # =========================================================================
    # Step 3: Write output
    # =========================================================================
    with open(TARGET, "w") as f:
        f.writelines(lines)

    print("\n" + "=" * 60)
    print("DONE: {} subroutines patched".format(patched_count))
    print("Output: {}".format(TARGET))
    print("=" * 60)


if __name__ == "__main__":
    main()
