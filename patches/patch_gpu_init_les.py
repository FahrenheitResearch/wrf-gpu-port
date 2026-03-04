#!/usr/bin/env python3
"""
patch_gpu_init_les.py — Add LES diffusion fields to gpu_init/finalize_domain_data.

Adds ~20 3D grid% fields needed for LES turbulence (km_opt=2/3, diff_opt=2):
  - Deformation tensor: defor11-33
  - TKE: tke_1, tke_2
  - Eddy viscosity/diffusivity: xkmv, xkhv
  - Brunt-Vaisala, divergence, mixing length: bn2, div, dlk
  - Terrain metrics: zx, zy, rdz, rdzw
  - Physics tendencies (may be accessed even if zero): rthcuten, rqvcuten, rushten, rvshten

Safe to run multiple times — skips fields already registered.

Usage:
    python patch_gpu_init_les.py [WRF_DIR]
    # default: reads from WRF_DIR env var
"""
import sys
import re
import os

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)
WRF = WRF_DIR
ACC = "!" + "$" + "acc"

# Fields to add, grouped by purpose. All are "create" (computed on device).
LES_FIELDS = [
    # --- LES deformation tensor (3D) ---
    ("grid%defor11", "grid%defor12", "grid%defor13"),
    ("grid%defor22", "grid%defor23", "grid%defor33"),
    # --- Brunt-Vaisala, divergence (3D) ---
    ("grid%bn2", "grid%div"),
    # --- TKE (3D, two time levels) ---
    ("grid%tke_1", "grid%tke_2"),
    # --- Eddy viscosity/diffusivity vertical (3D) ---
    ("grid%xkmv", "grid%xkhv"),
    # --- Mixing length (3D) ---
    ("grid%dlk",),
    # --- Terrain metrics (3D) ---
    ("grid%zx", "grid%zy"),
    ("grid%rdz", "grid%rdzw"),
    # --- Physics tendencies (3D, zero for LES but may be accessed) ---
    ("grid%rthcuten", "grid%rqvcuten"),
    ("grid%rushten", "grid%rvshten"),
]

# Flatten for duplicate checking
ALL_LES_FIELD_NAMES = []
for group in LES_FIELDS:
    for f in group:
        ALL_LES_FIELD_NAMES.append(f)


def find_existing_fields(text):
    """Extract all grid%xxx field names already in acc enter/exit data lines."""
    # Match grid%word patterns in acc directives
    return set(re.findall(r'grid%\w+', text))


def build_init_lines(existing):
    """Build acc enter data create(...) lines for fields not already registered."""
    lines = []
    lines.append(f"      ! --- LES diffusion fields (added by patch_gpu_init_les.py) ---")
    for group in LES_FIELDS:
        new_fields = [f for f in group if f not in existing]
        if not new_fields:
            continue
        field_str = ", ".join(new_fields)
        lines.append(f"      {ACC} enter data create({field_str})")
    return lines


def build_finalize_lines(existing):
    """Build acc exit data delete(...) lines for fields not already registered."""
    lines = []
    lines.append(f"      ! --- LES diffusion fields ---")
    # Group into chunks of 4 for readability
    new_fields = [f for f in ALL_LES_FIELD_NAMES if f not in existing]
    for i in range(0, len(new_fields), 4):
        chunk = new_fields[i:i+4]
        field_str = ", ".join(chunk)
        lines.append(f"      {ACC} exit data delete({field_str})")
    return lines


def patch_subroutine(text, sub_name, builder_fn):
    """Insert new acc lines just before the WRITE(*,*) in the given subroutine."""
    # Find the subroutine
    pattern = rf'(SUBROUTINE\s+{sub_name}\s*\(grid\))'
    match = re.search(pattern, text)
    if not match:
        print(f"  ERROR: Could not find {sub_name} in file")
        sys.exit(1)

    sub_start = match.start()

    # Find END SUBROUTINE for this sub
    end_pattern = rf'END\s+SUBROUTINE\s+{sub_name}'
    end_match = re.search(end_pattern, text[sub_start:])
    if not end_match:
        print(f"  ERROR: Could not find END SUBROUTINE {sub_name}")
        sys.exit(1)

    sub_end = sub_start + end_match.end()
    sub_text = text[sub_start:sub_end]

    # Find existing fields in this subroutine
    existing = find_existing_fields(sub_text)

    # Build new lines
    new_lines = builder_fn(existing)
    if not new_lines or len(new_lines) <= 1:
        # Only the comment line, no actual fields to add
        print(f"  {sub_name}: all LES fields already present, nothing to add")
        return text

    # Count how many fields we're adding
    added = [f for f in ALL_LES_FIELD_NAMES if f not in existing]
    print(f"  {sub_name}: adding {len(added)} fields: {', '.join(added)}")

    # Insert before the WRITE line
    write_match = re.search(r"(\n\s*WRITE\s*\(\s*\*\s*,\s*\*\s*\))", sub_text)
    if write_match:
        insert_pos = sub_start + write_match.start()
        insert_block = "\n" + "\n".join(new_lines) + "\n"
        text = text[:insert_pos] + insert_block + text[insert_pos:]
    else:
        # No WRITE found, insert before END SUBROUTINE
        insert_pos = sub_start + end_match.start()
        insert_block = "\n".join(new_lines) + "\n\n"
        text = text[:insert_pos] + insert_block + text[insert_pos:]

    return text


def main():
    path = f"{WRF}/frame/module_domain.f90"
    print(f"Patching {path} for LES fields...")

    with open(path) as f:
        text = f.read()

    # Patch gpu_init_domain_data
    text = patch_subroutine(text, "gpu_init_domain_data", build_init_lines)

    # Patch gpu_finalize_domain_data
    text = patch_subroutine(text, "gpu_finalize_domain_data", build_finalize_lines)

    with open(path, "w") as f:
        f.write(text)

    print(f"Done. Written to {path}")
    print(f"Remember: touch the .o/.F files to avoid __FILE__ recompile issues")


if __name__ == "__main__":
    main()
