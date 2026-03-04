#!/usr/bin/env python3
"""
patch_solve_em_gpu.py — Add !$acc enter/exit data for solve_em local arrays

Parses local 3D and 2D array declarations in solve_em and adds UNSTRUCTURED
data management (!$acc enter data create / !$acc exit data delete) so that
local arrays are available on GPU for dynamics kernels.

IMPORTANT: Uses unstructured data directives (enter data / exit data) instead
of structured (data create / end data) because structured regions cause NVHPC
to try auto-offloading ALL code inside the region, including physics
subroutine calls that are not GPU-ready (e.g., Thompson microphysics).
Unstructured directives only manage memory — they don't affect offloading.

Idempotent: skips patching if '!$acc enter data create' already present.
"""

import re
import sys
import os
from pathlib import Path

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

SOLVE_EM = Path(WRF_DIR) / "dyn_em" / "solve_em.f90"

# ── Dimension patterns for LOCAL arrays ────────────────────────────
# 3D: DIMENSION(grid%sm31:grid%em31, grid%sm32:grid%em32, grid%sm33:grid%em33)
# Captures everything after :: (may be multi-var with continuations)
RE_3D = re.compile(
    r",DIMENSION\(grid%sm31:grid%em31,grid%sm32:grid%em32,grid%sm33:grid%em33\)\s*::\s*(.+)",
    re.IGNORECASE,
)
# 2D: DIMENSION(grid%sm31:grid%em31, grid%sm33:grid%em33)
RE_2D = re.compile(
    r",DIMENSION\(grid%sm31:grid%em31,grid%sm33:grid%em33\)\s*::\s*(.+)",
    re.IGNORECASE,
)
# 4D (has a 4th dimension after the 3D part)
RE_4D = re.compile(
    r",DIMENSION\(grid%sm31:grid%em31,grid%sm32:grid%em32,grid%sm33:grid%em33,\s*\w+\)\s*::\s*(.+)",
    re.IGNORECASE,
)

# Variables that are subroutine ARGUMENTS (not locals).  These are passed in
# and may already be on the GPU — we skip them entirely from create() since
# they belong to the caller.
ARGUMENT_VARS = {
    # The big 4D argument arrays and their boundary/tendency copies
    "moist", "moist_bxs", "moist_bxe", "moist_bys", "moist_bye",
    "moist_btxs", "moist_btxe", "moist_btys", "moist_btye",
    "dfi_moist", "dfi_moist_bxs", "dfi_moist_bxe", "dfi_moist_bys",
    "dfi_moist_bye", "dfi_moist_btxs", "dfi_moist_btxe", "dfi_moist_btys",
    "dfi_moist_btye",
    "scalar", "scalar_bxs", "scalar_bxe", "scalar_bys", "scalar_bye",
    "scalar_btxs", "scalar_btxe", "scalar_btys", "scalar_btye",
    "dfi_scalar", "dfi_scalar_bxs", "dfi_scalar_bxe", "dfi_scalar_bys",
    "dfi_scalar_bye", "dfi_scalar_btxs", "dfi_scalar_btxe", "dfi_scalar_btys",
    "dfi_scalar_btye",
    "aerod", "aerocu", "ozmixm", "aerosolc_1", "aerosolc_2",
    "fdda3d", "fdda2d", "advh_t", "advz_t",
    "tracer", "tracer_bxs", "tracer_bxe", "tracer_bys", "tracer_bye",
    "tracer_btxs", "tracer_btxe", "tracer_btys", "tracer_btye",
    "pert3d", "nba_mij", "nba_rij", "sbmradar", "chem",
}

# Logical arrays can't go in a numeric data region usefully; skip them
SKIP_TYPES = {"logical"}


def main():
    if not SOLVE_EM.exists():
        print(f"ERROR: {SOLVE_EM} not found", file=sys.stderr)
        sys.exit(1)

    text = SOLVE_EM.read_text()
    lines = text.split("\n")

    # ── Idempotency check ──────────────────────────────────────────
    if "!$acc enter data create" in text:
        print("solve_em.f90 already contains '!$acc enter data create' — skipping.")
        return

    # ── Pass 1: collect local array names ──────────────────────────
    local_3d = []   # local 3-D real arrays
    local_2d = []   # local 2-D real arrays
    local_4d = []   # local 4-D real arrays (the _tend, _old locals)

    # We only scan the declaration block (first ~340 lines contain all decls).
    # Stop scanning at the first executable statement.
    decl_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # First executable statement after declarations
        if stripped == "feedback_is_ready = .false.":
            decl_end = i
            break

    if decl_end == 0:
        print("ERROR: could not find 'feedback_is_ready = .false.' marker", file=sys.stderr)
        sys.exit(1)

    def extract_varnames(match_text, lines, line_idx):
        """Extract variable names from a declaration, handling continuations.

        match_text is everything after '::' on the declaration line.
        If it ends with '&', read continuation lines.
        """
        full_text = match_text
        idx = line_idx
        while full_text.rstrip().endswith("&"):
            full_text = full_text.rstrip().rstrip("&")
            idx += 1
            if idx < len(lines):
                full_text += " " + lines[idx].strip()
        # Extract identifiers (variable names)
        names = [n.strip() for n in full_text.split(",") if n.strip()]
        # Filter out non-identifiers (safety)
        names = [n for n in names if re.match(r"^\w+$", n)]
        return names

    arg_lower = {v.lower() for v in ARGUMENT_VARS}

    for i in range(decl_end):
        line = lines[i]
        stripped = line.strip().lower()

        # Skip logical arrays
        if stripped.startswith("logical"):
            continue

        # Check for 4D match first (superset pattern of 3D)
        m4 = RE_4D.search(line)
        if m4:
            for varname in extract_varnames(m4.group(1), lines, i):
                if varname.lower() not in arg_lower:
                    local_4d.append(varname)
            continue

        # 3D match
        m3 = RE_3D.search(line)
        if m3:
            for varname in extract_varnames(m3.group(1), lines, i):
                if varname.lower() not in arg_lower:
                    local_3d.append(varname)
            continue

        # 2D match
        m2 = RE_2D.search(line)
        if m2:
            for varname in extract_varnames(m2.group(1), lines, i):
                if varname.lower() not in arg_lower:
                    local_2d.append(varname)
            continue

    print(f"Found {len(local_3d)} local 3D arrays: {', '.join(local_3d)}")
    print(f"Found {len(local_2d)} local 2D arrays: {', '.join(local_2d)}")
    print(f"Found {len(local_4d)} local 4D arrays: {', '.join(local_4d)}")

    # ── Build the !$acc enter data create(...) directive ────────────
    # All locals go in create() — they are uninitialized scratch space.
    # Using UNSTRUCTURED data management (enter data / exit data) so that
    # NVHPC doesn't try to auto-offload physics code inside the region.
    all_locals = local_3d + local_2d + local_4d
    if not all_locals:
        print("WARNING: no local arrays found — nothing to patch.")
        return

    # Format: split across continuation lines, ~6 vars per line
    chunk_size = 6

    # Build enter data create block
    enter_lines = []
    enter_lines.append("!$acc enter data create( &")
    for start in range(0, len(all_locals), chunk_size):
        chunk = all_locals[start : start + chunk_size]
        names = ", ".join(chunk)
        if start + chunk_size < len(all_locals):
            enter_lines.append(f"!$acc   {names}, &")
        else:
            enter_lines.append(f"!$acc   {names} )")
    # Indent to match surrounding code (3 spaces)
    enter_block = "\n".join("   " + l for l in enter_lines)

    # Build exit data delete block
    exit_lines = []
    exit_lines.append("!$acc exit data delete( &")
    for start in range(0, len(all_locals), chunk_size):
        chunk = all_locals[start : start + chunk_size]
        names = ", ".join(chunk)
        if start + chunk_size < len(all_locals):
            exit_lines.append(f"!$acc   {names}, &")
        else:
            exit_lines.append(f"!$acc   {names} )")
    exit_block = "\n".join("   " + l for l in exit_lines)

    # ── Pass 2: insert the directives ──────────────────────────────
    # Insert !$acc enter data create(...) right before the first executable stmt
    insert_open = decl_end

    # Find the RETURN at end of subroutine
    return_line = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "RETURN":
            return_line = i
            break

    if return_line is None:
        print("ERROR: could not find RETURN statement", file=sys.stderr)
        sys.exit(1)

    # ── Build gpu_init one-time call ─────────────────────────────────
    # gpu_init_domain_data copies all grid% arrays to GPU.
    # It must run once before any !$acc present() directives.
    # We use a SAVE variable to ensure it only runs on the first call.
    gpu_init_block = (
        "   ! --- GPU init: copy grid arrays to device (one-time) ---\n"
        "   IF (.NOT. gpu_data_initialized) THEN\n"
        "     CALL gpu_init_domain_data(grid)\n"
        "     gpu_data_initialized = .TRUE.\n"
        "   END IF"
    )

    # Find the declaration block to add the SAVE variable
    # Look for the last LOGICAL declaration or a good spot before decl_end
    save_decl = "   LOGICAL, SAVE :: gpu_data_initialized = .FALSE."

    # Build new file content
    new_lines = []
    save_inserted = False
    for i, line in enumerate(lines):
        # Insert SAVE declaration near start of declarations
        if not save_inserted and i > 0 and i < decl_end:
            stripped = line.strip().lower()
            # Insert after first LOGICAL declaration we find
            if stripped.startswith("logical"):
                new_lines.append(line)
                new_lines.append(save_decl)
                save_inserted = True
                continue
        if i == insert_open:
            # If we haven't inserted the SAVE decl yet, do it before exec stmts
            if not save_inserted:
                new_lines.append(save_decl)
                save_inserted = True
            new_lines.append("")
            new_lines.append(gpu_init_block)
            new_lines.append("")
            new_lines.append(enter_block)
            new_lines.append("")
        if i == return_line:
            new_lines.append("")
            new_lines.append(exit_block)
            new_lines.append("")
        new_lines.append(line)

    SOLVE_EM.write_text("\n".join(new_lines))
    print(f"\nPatched {SOLVE_EM}")
    print(f"  - Inserted CALL gpu_init_domain_data(grid) one-time init")
    print(f"  - Inserted !$acc enter data create({len(all_locals)} vars) at line ~{insert_open + 1}")
    print(f"  - Inserted !$acc exit data delete before RETURN at line ~{return_line + 1}")


if __name__ == "__main__":
    main()
