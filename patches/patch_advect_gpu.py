#!/usr/bin/env python3
"""
Patch module_advect_em.f90 with OpenACC directives for high-order horizontal advection.

Targets the preprocessed .f90 file (what nvfortran actually compiles).
The file already has !$acc directives on:
  - Local array data regions (!$acc data create)
  - 2nd-order horizontal loops
  - All vertical advection loops
  - advect_scalar_pd PD-limiter and tendency-update loops

What this script ADDS:
  1. advect_scalar_pd: !$acc kernels on 5th/6th order y-flux and x-flux loops
     (these use full 3D fqy/fqx arrays, so all j-iterations are independent)
  2. advect_u/v: !$acc kernels on the x-flux inner (k,i) loops within the
     high-order sections (the j-loop stays sequential due to 2D fqx buffer,
     but inner loops can run on GPU)

Strategy:
  - advect_scalar_pd: The y-flux and x-flux loops write to fqy(i,k,j) and
    fqx(i,k,j) which are full 3D. The DO j loop with IF/ELSEIF branches is
    parallelizable because each j writes to different memory. Wrap entire
    j-loop in !$acc kernels.
  - advect_u/v: fqx(i,k) is 2D, rewritten each j. Cannot parallelize j.
    Instead wrap the inner DO k / DO i flux loops and the tendency update
    loop within each j iteration.

Rules:
  - All dummy argument arrays use present() clause
  - Local arrays (fqx, fqy, etc.) already in !$acc data create regions
  - Do NOT change computational logic
  - Skip loops that already have !$acc directives nearby
"""

import re
import sys
import shutil
import os

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

INPUT_FILE = os.path.join(WRF_DIR, "dyn_em", "module_advect_em.f90")
BACKUP_FILE = INPUT_FILE + ".bak_before_horz_acc"


def read_file(path):
    with open(path, 'r') as f:
        return f.readlines()


def write_file(path, lines):
    with open(path, 'w') as f:
        f.writelines(lines)


def find_line(lines, text, start=0, end=None):
    """Find first line containing text. Returns index or -1."""
    if end is None:
        end = len(lines)
    for i in range(start, min(end, len(lines))):
        if text in lines[i]:
            return i
    return -1


def find_line_re(lines, pattern, start=0, end=None):
    """Find first line matching regex pattern. Returns index or -1."""
    if end is None:
        end = len(lines)
    for i in range(start, min(end, len(lines))):
        if re.search(pattern, lines[i]):
            return i
    return -1


def find_subroutine_range(lines, name):
    """Find (start_line, end_line) for SUBROUTINE name ... END SUBROUTINE name."""
    name_upper = name.upper()
    start = None
    for i, line in enumerate(lines):
        s = line.strip().upper()
        if start is None:
            # Match SUBROUTINE name but not SUBROUTINE name_pd etc
            if re.match(r'SUBROUTINE\s+' + re.escape(name_upper) + r'\s*[\((\s]', s):
                start = i
        else:
            if re.match(r'END\s*SUBROUTINE\s+' + re.escape(name_upper) + r'\s*$', s):
                return start, i
    return start, None


def get_indent(line):
    """Return leading whitespace."""
    return line[:len(line) - len(line.lstrip())]


def has_acc_nearby(lines, line_idx, window=3):
    """Check if there's an !$acc directive within window lines of line_idx."""
    start = max(0, line_idx - window)
    end = min(len(lines), line_idx + window + 1)
    for i in range(start, end):
        if '!$acc' in lines[i]:
            return True
    return False


def find_enddo(lines, do_line):
    """Find the ENDDO matching DO at do_line, handling nesting."""
    depth = 0
    for i in range(do_line, len(lines)):
        s = lines[i].strip().upper()
        # Match DO with loop variable or named DO
        if re.match(r'^DO\s+[A-Z_]\w*\s*=', s) or re.match(r'^\w+\s*:\s*DO\s', s):
            depth += 1
        if s.startswith('ENDDO') or s.startswith('END DO'):
            depth -= 1
            if depth == 0:
                return i
    return -1


def insert_lines_at(lines, idx, new_lines):
    """Insert new_lines before lines[idx]. Returns count inserted."""
    for i, nl in enumerate(new_lines):
        lines.insert(idx + i, nl)
    return len(new_lines)


class Patcher:
    """Accumulates insertions and applies them bottom-up to preserve indices."""

    def __init__(self, lines):
        self.lines = lines
        self.insertions = []  # list of (line_idx, new_lines_list, label)

    def add_before(self, line_idx, new_lines, label=""):
        """Schedule insertion of new_lines before lines[line_idx]."""
        self.insertions.append((line_idx, new_lines, label))

    def add_after(self, line_idx, new_lines, label=""):
        """Schedule insertion of new_lines after lines[line_idx]."""
        self.insertions.append((line_idx + 1, new_lines, label))

    def apply(self):
        """Apply all insertions in reverse order. Returns total lines inserted."""
        # Sort by position descending so later insertions don't shift earlier ones
        self.insertions.sort(key=lambda x: x[0], reverse=True)
        total = 0
        for idx, new_lines, label in self.insertions:
            n = insert_lines_at(self.lines, idx, new_lines)
            total += n
            if label:
                print(f"    {label} @ line {idx+1} ({n} lines)")
        return total


def acc_kernels_present(present_vars, indent="      "):
    """Generate !$acc kernels present(...) line."""
    present_str = ', '.join(present_vars)
    return f"{indent}!$acc kernels present({present_str})\n"


def acc_end_kernels(indent="      "):
    """Generate !$acc end kernels line."""
    return f"{indent}!$acc end kernels\n"


# ============================================================
# advect_scalar_pd: High-order horizontal flux loops
# ============================================================

def patch_scalar_pd_horz(lines):
    """Add !$acc kernels to high-order horizontal flux loops in advect_scalar_pd.

    These loops use full 3D fqy(i,k,j), fqx(i,k,j), fqyl, fqxl arrays,
    so all j-iterations are independent and can be parallelized.
    """
    print("\n  advect_scalar_pd high-order horizontal fluxes:")
    sub_start, sub_end = find_subroutine_range(lines, 'advect_scalar_pd')
    if sub_start is None:
        print("    ERROR: subroutine not found")
        return 0

    patcher = Patcher(lines)

    # Present clause for scalar_pd
    pd_present = ['field', 'field_old', 'tendency', 'h_tendency', 'z_tendency',
                  'ru', 'rv', 'rom', 'mut', 'mub', 'mu_old',
                  'msftx', 'msfty', 'rdzw', 'c1', 'c2']

    # --- 6th-order y-flux ---
    # Look for "j_loop_y_flux_6 : DO j = j_start, j_end+1"
    # This loop writes fqy(i,k,j) and fqyl(i,k,j) for each j
    idx = find_line(lines, 'j_loop_y_flux_6 : DO j = j_start, j_end+1', sub_start, sub_end)
    if idx >= 0 and not has_acc_nearby(lines, idx):
        enddo = find_enddo(lines, idx)
        if enddo > 0:
            indent = get_indent(lines[idx])
            patcher.add_after(enddo, [acc_end_kernels(indent)], "6th-order y-flux END")
            patcher.add_before(idx, [acc_kernels_present(pd_present, indent)], "6th-order y-flux START")

    # --- 6th-order x-flux ---
    # After the y-flux loop, there's "DO j = j_start, j_end" containing x-flux
    # Find it by looking for the x-flux body marker after the y-flux enddo
    search_from = enddo + 1 if (idx >= 0 and enddo > 0) else sub_start
    # The x-flux section starts with recalculation of i_start, i_end, then "DO j = j_start, j_end"
    # body contains "fqx( i,k,j ) = vel*flux6("
    xflux_body = find_line(lines, 'vel*flux6( field(i-3,k,j)', search_from, sub_end)
    if xflux_body >= 0:
        # Walk backwards to find the enclosing "DO j = j_start, j_end"
        xflux_j_loop = -1
        for i in range(xflux_body, search_from, -1):
            if re.search(r'DO j\s*=\s*j_start\s*,\s*j_end\s*$', lines[i].strip(), re.IGNORECASE):
                xflux_j_loop = i
                break
        if xflux_j_loop >= 0 and not has_acc_nearby(lines, xflux_j_loop):
            # Find "ENDDO" comment "! x-flux" or just the matching enddo
            xflux_enddo = find_enddo(lines, xflux_j_loop)
            if xflux_enddo > 0:
                indent = get_indent(lines[xflux_j_loop])
                patcher.add_after(xflux_enddo, [acc_end_kernels(indent)], "6th-order x-flux END")
                patcher.add_before(xflux_j_loop, [acc_kernels_present(pd_present, indent)], "6th-order x-flux START")

    # --- 5th-order y-flux ---
    idx5y = find_line(lines, 'j_loop_y_flux_5 : DO j = j_start, j_end+1', sub_start, sub_end)
    if idx5y >= 0 and not has_acc_nearby(lines, idx5y):
        enddo5y = find_enddo(lines, idx5y)
        if enddo5y > 0:
            indent = get_indent(lines[idx5y])
            patcher.add_after(enddo5y, [acc_end_kernels(indent)], "5th-order y-flux END")
            patcher.add_before(idx5y, [acc_kernels_present(pd_present, indent)], "5th-order y-flux START")

    # --- 5th-order x-flux ---
    search_from5 = enddo5y + 1 if (idx5y >= 0 and enddo5y > 0) else sub_start
    xflux5_body = find_line(lines, 'vel*flux5( field(i-3,k,j)', search_from5, sub_end)
    if xflux5_body >= 0:
        xflux5_j_loop = -1
        for i in range(xflux5_body, search_from5, -1):
            if re.search(r'DO j\s*=\s*j_start\s*,\s*j_end\s*$', lines[i].strip(), re.IGNORECASE):
                xflux5_j_loop = i
                break
        if xflux5_j_loop >= 0 and not has_acc_nearby(lines, xflux5_j_loop):
            xflux5_enddo = find_enddo(lines, xflux5_j_loop)
            if xflux5_enddo > 0:
                indent = get_indent(lines[xflux5_j_loop])
                patcher.add_after(xflux5_enddo, [acc_end_kernels(indent)], "5th-order x-flux END")
                patcher.add_before(xflux5_j_loop, [acc_kernels_present(pd_present, indent)], "5th-order x-flux START")

    # --- 4th-order and 3rd-order y-flux and x-flux ---
    # These follow the same pattern but use j_loop naming or just DO j
    # 4th order y-flux: search for the flux4 pattern within advect_scalar_pd
    # Let's find the "ELSE IF( horz_order == 4 )" section
    horz4_start = find_line(lines, "horz_order == 4 ) THEN", sub_start, sub_end)
    if horz4_start >= 0:
        horz3_start = find_line(lines, "horz_order == 3 ) THEN", horz4_start + 1, sub_end)
        horz4_end = horz3_start if horz3_start >= 0 else sub_end

        # 4th-order y-flux: named j_loop_y_flux_4 or just DO j
        idx4y = find_line(lines, 'j_loop_y_flux_4', horz4_start, horz4_end)
        if idx4y < 0:
            # Search for DO j = j_start, j_end+1 after horz4_start
            idx4y = find_line_re(lines, r'DO j\s*=\s*j_start\s*,\s*j_end\s*\+\s*1', horz4_start, horz4_end)
        if idx4y >= 0 and not has_acc_nearby(lines, idx4y):
            enddo4y = find_enddo(lines, idx4y)
            if enddo4y > 0:
                indent = get_indent(lines[idx4y])
                patcher.add_after(enddo4y, [acc_end_kernels(indent)], "4th-order y-flux END")
                patcher.add_before(idx4y, [acc_kernels_present(pd_present, indent)], "4th-order y-flux START")

        # 4th-order x-flux: DO j = j_start, j_end containing flux4 on field
        search4x = enddo4y + 1 if (idx4y >= 0 and enddo4y > 0) else horz4_start
        xflux4_body = find_line(lines, 'vel*flux4(', search4x, horz4_end)
        if xflux4_body >= 0:
            xflux4_j = -1
            for i in range(xflux4_body, search4x, -1):
                if re.search(r'DO j\s*=\s*j_start\s*,\s*j_end\s*$', lines[i].strip(), re.IGNORECASE):
                    xflux4_j = i
                    break
            if xflux4_j >= 0 and not has_acc_nearby(lines, xflux4_j):
                xflux4_enddo = find_enddo(lines, xflux4_j)
                if xflux4_enddo > 0:
                    indent = get_indent(lines[xflux4_j])
                    patcher.add_after(xflux4_enddo, [acc_end_kernels(indent)], "4th-order x-flux END")
                    patcher.add_before(xflux4_j, [acc_kernels_present(pd_present, indent)], "4th-order x-flux START")

    # 3rd-order: similar pattern
    if horz3_start and horz3_start >= 0:
        horz2_start = find_line(lines, "horz_order == 2 ) THEN", horz3_start + 1, sub_end)
        horz3_end = horz2_start if horz2_start >= 0 else sub_end

        idx3y = find_line(lines, 'j_loop_y_flux_3', horz3_start, horz3_end)
        if idx3y < 0:
            idx3y = find_line_re(lines, r'DO j\s*=\s*j_start\s*,\s*j_end\s*\+\s*1', horz3_start, horz3_end)
        if idx3y >= 0 and not has_acc_nearby(lines, idx3y):
            enddo3y = find_enddo(lines, idx3y)
            if enddo3y > 0:
                indent = get_indent(lines[idx3y])
                patcher.add_after(enddo3y, [acc_end_kernels(indent)], "3rd-order y-flux END")
                patcher.add_before(idx3y, [acc_kernels_present(pd_present, indent)], "3rd-order y-flux START")

        # 3rd-order x-flux
        search3x = enddo3y + 1 if (idx3y >= 0 and enddo3y > 0) else horz3_start
        xflux3_body = find_line(lines, 'vel*flux3(', search3x, horz3_end)
        if xflux3_body >= 0:
            xflux3_j = -1
            for i in range(xflux3_body, search3x, -1):
                if re.search(r'DO j\s*=\s*j_start\s*,\s*j_end\s*$', lines[i].strip(), re.IGNORECASE):
                    xflux3_j = i
                    break
            if xflux3_j >= 0 and not has_acc_nearby(lines, xflux3_j):
                xflux3_enddo = find_enddo(lines, xflux3_j)
                if xflux3_enddo > 0:
                    indent = get_indent(lines[xflux3_j])
                    patcher.add_after(xflux3_enddo, [acc_end_kernels(indent)], "3rd-order x-flux END")
                    patcher.add_before(xflux3_j, [acc_kernels_present(pd_present, indent)], "3rd-order x-flux START")

    # --- 2nd-order y-flux and x-flux (in advect_scalar_pd) ---
    # These are the DO j / DO k / DO i loops in the horz_order==2 branch
    horz2_start_pd = find_line(lines, "horz_order == 2 ) THEN", sub_start, sub_end)
    if horz2_start_pd >= 0:
        horz2_end_pd = find_line(lines, "ENDIF horizontal_order_test", horz2_start_pd, sub_end)
        if horz2_end_pd < 0:
            horz2_end_pd = sub_end

        # y-flux: "DO j = j_start, j_end+1" with fqy writes
        idx2y = find_line_re(lines, r'DO j\s*=\s*j_start\s*,\s*j_end\s*\+\s*1', horz2_start_pd, horz2_end_pd)
        if idx2y >= 0 and not has_acc_nearby(lines, idx2y):
            enddo2y = find_enddo(lines, idx2y)
            if enddo2y > 0:
                indent = get_indent(lines[idx2y])
                patcher.add_after(enddo2y, [acc_end_kernels(indent)], "2nd-order y-flux END")
                patcher.add_before(idx2y, [acc_kernels_present(pd_present, indent)], "2nd-order y-flux START")

        # x-flux: "DO j = j_start, j_end" with fqx writes
        search2x = enddo2y + 1 if (idx2y >= 0 and enddo2y > 0) else horz2_start_pd
        idx2x = find_line_re(lines, r'DO j\s*=\s*j_start\s*,\s*j_end\s*$', search2x, horz2_end_pd)
        if idx2x >= 0 and not has_acc_nearby(lines, idx2x):
            enddo2x = find_enddo(lines, idx2x)
            if enddo2x > 0:
                indent = get_indent(lines[idx2x])
                patcher.add_after(enddo2x, [acc_end_kernels(indent)], "2nd-order x-flux END")
                patcher.add_before(idx2x, [acc_kernels_present(pd_present, indent)], "2nd-order x-flux START")

    return patcher.apply()


# ============================================================
# advect_scalar: High-order horizontal flux loops
# ============================================================

def patch_scalar_horz(lines):
    """Add !$acc kernels to high-order horizontal flux loops in advect_scalar.

    advect_scalar uses similar fqy(i,k,2) flip-flop pattern as advect_u/v
    for y-flux. But the x-flux uses fqx(i,k) 2D within DO j.
    Actually let me check...
    """
    print("\n  advect_scalar high-order horizontal fluxes:")
    sub_start, sub_end = find_subroutine_range(lines, 'advect_scalar')
    if sub_start is None:
        print("    ERROR: subroutine not found")
        return 0

    # Check local array declarations to understand structure
    # advect_scalar uses fqx(its:ite+1, kts:kte) and fqy(its:ite, kts:kte, 2)
    # So it has the same jp0/jp1 flip-flop as advect_u/v for y-flux
    # The x-flux DO j loop has fqx(i,k) as 2D, rewritten each j
    # So same situation as advect_u/v: only inner loops parallelizable

    # For advect_scalar, the x-flux inner loops and tendency update loops
    # within each j-iteration can be wrapped.
    # But this is the same approach as advect_u/v, so let's skip for now
    # since advect_scalar_pd is the primary scalar advection routine used.
    print("    Skipping (advect_scalar_pd is the active scalar routine)")
    return 0


# ============================================================
# advect_u: High-order horizontal x-flux inner loops
# ============================================================

def patch_u_horz(lines):
    """Add !$acc kernels to inner (k,i) loops of high-order x-flux in advect_u.

    The x-flux section structure (for order 5 and 6):
      DO j = j_start, j_end       ! sequential (fqx is 2D)
        DO k=kts,ktf               ! <-- parallelize this
        DO i = i_start_f, i_end_f
          fqx(i,k) = ...flux5/6...
        ENDDO
        ENDDO
        IF(degrade_xs) ... boundary fixup (small, leave serial)
        IF(degrade_xe) ... boundary fixup (small, leave serial)
        DO k=kts,ktf               ! <-- parallelize this
        DO i = i_start, i_end
          tendency(i,k,j) = tendency(i,k,j) - mrdx*(fqx(i+1,k)-fqx(i,k))
        ENDDO
        ENDDO
      ENDDO

    We wrap the main flux computation and the tendency update with !$acc kernels.
    """
    print("\n  advect_u high-order horizontal x-flux:")
    sub_start, sub_end = find_subroutine_range(lines, 'advect_u')
    if sub_start is None:
        print("    ERROR: subroutine not found")
        return 0

    patcher = Patcher(lines)
    u_present = ['u', 'u_old', 'tendency', 'ru', 'rv', 'rom',
                 'mut', 'msfux', 'msfuy', 'msfvx', 'msfvy', 'msftx', 'msfty',
                 'fzm', 'fzp', 'rdzw', 'c1', 'c2']

    # Find the horizontal order sections
    for order, flux_fn in [(6, 'flux6'), (5, 'flux5')]:
        if order == 6:
            section_start = find_line(lines, 'horizontal_order_test : IF( horz_order == 6', sub_start, sub_end)
        else:
            section_start = find_line(lines, 'ELSE IF( horz_order == 5', sub_start, sub_end)

        if section_start < 0:
            continue

        # Find the next section boundary
        if order == 6:
            section_end = find_line(lines, 'ELSE IF( horz_order == 5', section_start + 1, sub_end)
        else:
            section_end = find_line(lines, 'ELSE IF( horz_order == 4', section_start + 1, sub_end)
        if section_end < 0:
            section_end = sub_end

        # Find the x-flux section (second DO j = j_start, j_end in this order block)
        # The first j-loop is the y-flux (j_loop_y_flux_N)
        # The second is the x-flux
        y_flux_label = f'j_loop_y_flux_{order}'
        y_flux_start = find_line(lines, y_flux_label, section_start, section_end)
        if y_flux_start < 0:
            continue
        y_flux_end = find_enddo(lines, y_flux_start)
        if y_flux_end < 0:
            continue

        # x-flux j-loop starts after y-flux ends
        xflux_j = find_line_re(lines, r'DO j\s*=\s*j_start\s*,\s*j_end\s*$', y_flux_end + 1, section_end)
        if xflux_j < 0:
            continue
        xflux_j_end = find_enddo(lines, xflux_j)
        if xflux_j_end < 0:
            continue

        # Inside this j-loop, find the main flux DO k / DO i loop
        # Pattern: "DO k=kts,ktf" followed by "DO i = i_start_f, i_end_f" with flux computation
        flux_k_start = find_line_re(lines, r'DO k\s*=\s*kts\s*,\s*ktf', xflux_j + 1, xflux_j_end)
        if flux_k_start >= 0 and not has_acc_nearby(lines, flux_k_start):
            flux_k_end = find_enddo(lines, flux_k_start)
            if flux_k_end > 0:
                indent = get_indent(lines[flux_k_start])
                patcher.add_after(flux_k_end, [acc_end_kernels(indent)],
                                  f"advect_u order={order} x-flux compute END")
                patcher.add_before(flux_k_start, [acc_kernels_present(u_present, indent)],
                                   f"advect_u order={order} x-flux compute START")

        # Find the tendency update loop (last DO k / DO i before the j-loop ENDDO)
        # Search backwards from xflux_j_end for "tendency(i,k,j) = tendency(i,k,j) - mrdx"
        tend_marker = find_line(lines, 'tendency(i,k,j) = tendency(i,k,j) - mrdx*(fqx(i+1,k)-fqx(i,k))',
                                xflux_j + 1, xflux_j_end)
        if tend_marker >= 0:
            # Walk backwards to find the DO k
            for i in range(tend_marker, xflux_j, -1):
                if re.search(r'DO k\s*=\s*kts\s*,\s*ktf', lines[i]):
                    tend_k_start = i
                    break
            else:
                tend_k_start = -1

            if tend_k_start >= 0 and not has_acc_nearby(lines, tend_k_start):
                tend_k_end = find_enddo(lines, tend_k_start)
                if tend_k_end > 0:
                    indent = get_indent(lines[tend_k_start])
                    patcher.add_after(tend_k_end, [acc_end_kernels(indent)],
                                      f"advect_u order={order} x-tendency END")
                    patcher.add_before(tend_k_start, [acc_kernels_present(u_present, indent)],
                                       f"advect_u order={order} x-tendency START")

    return patcher.apply()


# ============================================================
# advect_v: High-order horizontal x-flux inner loops
# ============================================================

def patch_v_horz(lines):
    """Add !$acc kernels to inner loops of high-order x-flux in advect_v.
    Same structure as advect_u but with v-specific arrays.
    """
    print("\n  advect_v high-order horizontal x-flux:")
    sub_start, sub_end = find_subroutine_range(lines, 'advect_v')
    if sub_start is None:
        print("    ERROR: subroutine not found")
        return 0

    patcher = Patcher(lines)
    v_present = ['v', 'v_old', 'tendency', 'ru', 'rv', 'rom',
                 'mut', 'msfux', 'msfuy', 'msfvx', 'msfvy', 'msftx', 'msfty',
                 'fzm', 'fzp', 'rdzw', 'c1', 'c2']

    for order, flux_fn in [(6, 'flux6'), (5, 'flux5')]:
        if order == 6:
            section_start = find_line(lines, 'horizontal_order_test : IF( horz_order == 6', sub_start, sub_end)
        else:
            section_start = find_line(lines, 'ELSE IF( horz_order == 5', sub_start, sub_end)

        if section_start < 0:
            continue

        if order == 6:
            section_end = find_line(lines, 'ELSE IF( horz_order == 5', section_start + 1, sub_end)
        else:
            section_end = find_line(lines, 'ELSE IF( horz_order == 4', section_start + 1, sub_end)
        if section_end < 0:
            section_end = sub_end

        # Find y-flux loop end first
        y_flux_label = f'j_loop_y_flux_{order}'
        y_flux_start = find_line(lines, y_flux_label, section_start, section_end)
        if y_flux_start < 0:
            continue
        y_flux_end = find_enddo(lines, y_flux_start)
        if y_flux_end < 0:
            continue

        # x-flux j-loop
        xflux_j = find_line_re(lines, r'DO j\s*=\s*j_start\s*,\s*j_end\s*$', y_flux_end + 1, section_end)
        if xflux_j < 0:
            continue
        xflux_j_end = find_enddo(lines, xflux_j)
        if xflux_j_end < 0:
            continue

        # Main flux k,i loop
        flux_k_start = find_line_re(lines, r'DO k\s*=\s*kts\s*,\s*ktf', xflux_j + 1, xflux_j_end)
        if flux_k_start >= 0 and not has_acc_nearby(lines, flux_k_start):
            flux_k_end = find_enddo(lines, flux_k_start)
            if flux_k_end > 0:
                indent = get_indent(lines[flux_k_start])
                patcher.add_after(flux_k_end, [acc_end_kernels(indent)],
                                  f"advect_v order={order} x-flux compute END")
                patcher.add_before(flux_k_start, [acc_kernels_present(v_present, indent)],
                                   f"advect_v order={order} x-flux compute START")

        # Tendency update loop
        tend_marker = find_line(lines, 'tendency(i,k,j) = tendency(i,k,j) - mrdx*(fqx(i+1,k)-fqx(i,k))',
                                xflux_j + 1, xflux_j_end)
        if tend_marker < 0:
            # advect_v uses mrdy for some, try both
            tend_marker = find_line(lines, 'tendency(i,k,j) = tendency(i,k,j) - mrdy*(fqx(i+1,k)-fqx(i,k))',
                                    xflux_j + 1, xflux_j_end)
        if tend_marker >= 0:
            for i in range(tend_marker, xflux_j, -1):
                if re.search(r'DO k\s*=\s*kts\s*,\s*ktf', lines[i]):
                    tend_k_start = i
                    break
            else:
                tend_k_start = -1

            if tend_k_start >= 0 and not has_acc_nearby(lines, tend_k_start):
                tend_k_end = find_enddo(lines, tend_k_start)
                if tend_k_end > 0:
                    indent = get_indent(lines[tend_k_start])
                    patcher.add_after(tend_k_end, [acc_end_kernels(indent)],
                                      f"advect_v order={order} x-tendency END")
                    patcher.add_before(tend_k_start, [acc_kernels_present(v_present, indent)],
                                       f"advect_v order={order} x-tendency START")

    return patcher.apply()


# ============================================================
# Main
# ============================================================

def main():
    print(f"Reading {INPUT_FILE}")
    lines = read_file(INPUT_FILE)
    orig_len = len(lines)
    print(f"  Original: {orig_len} lines")

    # Count existing directives
    existing_kernels = sum(1 for l in lines if '!$acc kernels' in l and '!$acc end' not in l)
    existing_end = sum(1 for l in lines if '!$acc end kernels' in l)
    print(f"  Existing !$acc kernels: {existing_kernels}, !$acc end kernels: {existing_end}")

    total = 0
    total += patch_scalar_pd_horz(lines)
    total += patch_u_horz(lines)
    total += patch_v_horz(lines)
    total += patch_scalar_horz(lines)

    if total > 0:
        # Backup
        if not os.path.exists(BACKUP_FILE):
            shutil.copy2(INPUT_FILE, BACKUP_FILE)
            print(f"\n  Backup saved to {BACKUP_FILE}")

        write_file(INPUT_FILE, lines)
        print(f"\n  Wrote {len(lines)} lines (+{total} lines inserted)")
    else:
        print("\n  No patches applied (targets not found or already patched)")

    # Verification
    print("\n--- Verification ---")
    final_kernels = sum(1 for l in lines if '!$acc kernels' in l and '!$acc end' not in l)
    final_end = sum(1 for l in lines if '!$acc end kernels' in l)
    final_data = sum(1 for l in lines if '!$acc data' in l and '!$acc end' not in l)
    final_data_end = sum(1 for l in lines if '!$acc end data' in l)

    print(f"  !$acc kernels:     {final_kernels} (was {existing_kernels}, +{final_kernels - existing_kernels})")
    print(f"  !$acc end kernels: {final_end} (was {existing_end}, +{final_end - existing_end})")
    print(f"  !$acc data:        {final_data}")
    print(f"  !$acc end data:    {final_data_end}")

    if final_kernels != final_end:
        print(f"  WARNING: Mismatch between kernels ({final_kernels}) and end kernels ({final_end})!")
    else:
        print(f"  OK: {final_kernels} matched kernels pairs")

    if final_data != final_data_end:
        print(f"  WARNING: Mismatch between data ({final_data}) and end data ({final_data_end})!")
    else:
        print(f"  OK: {final_data} matched data pairs")


if __name__ == '__main__':
    main()
