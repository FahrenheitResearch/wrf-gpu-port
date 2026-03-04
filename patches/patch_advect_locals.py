#!/usr/bin/env python3
"""
Add !$acc data create() regions for local arrays in advect subroutines.
This prevents per-kernel upload/download of large work arrays.

Only targets subroutines that have ACC kernels referencing local arrays:
- advect_scalar_pd: fqx, fqy, fqz, fqxl, fqyl, fqzl, flux_out, ph_low, scale_out
- advect_scalar_wenopd: same locals
- advect_w: vflux, fqx, fqy (handled by fix_advect_w_create.py)

advect_u, advect_v, advect_scalar only reference subroutine arguments in
present() clauses, so they don't need data create for locals.
"""
import re
import os
import sys

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

TARGET = os.path.join(WRF_DIR, "dyn_em", "module_advect_em.f90")

# Local arrays to create on device for each subroutine
# Format: (subroutine_name, anchor_text, local_arrays)
SUBROUTINES = [
    ("advect_scalar_pd",
     "horz_order = config_flags%h_sca_adv_order",
     ["fqx", "fqy", "fqz", "fqxl", "fqyl", "fqzl", "flux_out", "ph_low", "scale_out"]),

    ("advect_scalar_wenopd",
     "horz_order = config_flags%h_sca_adv_order",
     ["fqx", "fqy", "fqz", "fqxl", "fqyl", "fqzl", "flux_out", "ph_low", "scale_out"]),
]

def main():
    with open(TARGET) as f:
        text = f.read()

    # Idempotency: check for our specific marker
    if 'patch_advect_locals applied' in text:
        print("Already patched — skipping")
        return

    lines = text.split('\n')

    # First, enable any disabled !!$acc data create that contain flux_out
    # (WRF source has these but they're commented out with double !!)
    enabled = 0
    for i, line in enumerate(lines):
        if '!!$acc' in line and 'data create' in line and 'flux_out' in line:
            lines[i] = line.replace('!!$acc', '!$acc', 1)
            enabled += 1
        elif '!!$acc' in line and 'flux_out' in line:
            lines[i] = line.replace('!!$acc', '!$acc', 1)
            enabled += 1
        # Also enable disabled !!$acc end data near flux_out create blocks
        elif '!!$acc end data' in line or '! end data' == line.strip():
            # Check if this is near a PD subroutine END
            for j in range(i, min(i+5, len(lines))):
                if 'END SUBROUTINE advect_scalar_pd' in lines[j] or \
                   'END SUBROUTINE advect_scalar_wenopd' in lines[j]:
                    lines[i] = '   !$acc end data'
                    enabled += 1
                    break

    if enabled > 0:
        print(f"  Enabled {enabled} disabled ACC data directives")

    patched = 0
    # Process each subroutine that needs data create
    for sub_name, anchor, locals_list in SUBROUTINES:
        # Find the subroutine
        sub_start = None
        sub_end = None
        in_sub = False

        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if f'SUBROUTINE {sub_name.upper()}' in stripped and not stripped.startswith('END'):
                if sub_start is None:
                    sub_start = i
                    in_sub = True
            if in_sub and stripped.startswith('END SUBROUTINE'):
                sub_end = i
                in_sub = False
                break

        if sub_start is None or sub_end is None:
            print(f"  WARNING: Could not find {sub_name}")
            continue

        # Check if data create already exists for this subroutine
        sub_text = '\n'.join(lines[sub_start:sub_end])
        if '!$acc data create(' in sub_text and 'flux_out' in sub_text:
            print(f"  {sub_name}: data create already exists")
            # Still check if scale_out is included
            if 'scale_out' not in sub_text.split('data create')[1].split(')')[0]:
                # Need to add scale_out to existing data create
                for i in range(sub_start, sub_end):
                    if '!$acc data create(' in lines[i] and 'flux_out' in lines[i]:
                        if 'scale_out' not in lines[i]:
                            lines[i] = lines[i].rstrip().rstrip(')') + ', scale_out)'
                            print(f"    Added scale_out to existing data create")
                        break
                    # Check continuation lines
                    if '!$acc' in lines[i] and 'flux_out' in lines[i]:
                        if 'scale_out' not in lines[i]:
                            lines[i] = lines[i].rstrip().rstrip(')') + ', scale_out)'
                            print(f"    Added scale_out to existing data create")
                        break
            continue

        # Find the anchor line within the subroutine
        anchor_line = None
        for i in range(sub_start, sub_end):
            if anchor in lines[i]:
                anchor_line = i
                break

        if anchor_line is None:
            print(f"  WARNING: Could not find anchor '{anchor}' in {sub_name}")
            continue

        # Check which locals actually exist in this subroutine
        existing_locals = [v for v in locals_list if re.search(rf'\b{v}\b', sub_text)]

        if not existing_locals:
            print(f"  WARNING: No matching locals in {sub_name}")
            continue

        # Insert !$acc data create(...) before the anchor line
        vars_str = ", ".join(existing_locals)
        create_line = f"   !$acc data create({vars_str})"

        # Insert !$acc end data before END SUBROUTINE
        lines.insert(sub_end, "   !$acc end data")
        # Insert !$acc data create before anchor
        lines.insert(anchor_line, create_line)

        patched += 1
        print(f"  Patched {sub_name}: create({vars_str})")

    # Add idempotency marker
    lines.insert(0, "! patch_advect_locals applied")

    # Write back
    with open(TARGET, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nDone: {patched} subroutines patched, {enabled} directives enabled")


if __name__ == "__main__":
    main()
