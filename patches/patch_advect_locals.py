#!/usr/bin/env python3
"""
Add !$acc data create() regions for local arrays in advect subroutines.
This prevents per-kernel upload/download of large work arrays.
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
     ["fqx", "fqy", "fqz", "fqxl", "fqyl", "fqzl", "flux_out", "ph_low"]),

    ("advect_scalar",
     "horz_order = config_flags%h_sca_adv_order",
     ["vflux", "fqx", "fqy", "fqz"]),

    ("advect_u",
     "horz_order = config_flags%h_mom_adv_order",
     ["vflux", "fqx", "fqy"]),

    ("advect_v",
     "horz_order = config_flags%h_mom_adv_order",
     ["vflux", "fqx", "fqy"]),

    ("advect_w",
     "specified = .false.",
     ["vflux", "fqx", "fqy"]),
]

def main():
    with open(TARGET) as f:
        text = f.read()

    lines = text.split('\n')

    # Check idempotency
    if '!$acc data create(fqx' in text:
        print("Already patched — skipping")
        return

    patched = 0
    # Process each subroutine
    for sub_name, anchor, locals_list in SUBROUTINES:
        # Find the subroutine
        sub_start = None
        sub_end = None
        in_sub = False
        depth = 0

        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if f'SUBROUTINE {sub_name.upper()}' in stripped and not stripped.startswith('END'):
                if sub_start is None:
                    sub_start = i
                    in_sub = True
                    depth = 0
            if in_sub and stripped.startswith('END SUBROUTINE'):
                sub_end = i
                in_sub = False
                break

        if sub_start is None or sub_end is None:
            print(f"  WARNING: Could not find {sub_name}")
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
        sub_text = '\n'.join(lines[sub_start:sub_end])
        existing_locals = [v for v in locals_list if re.search(rf'\b{v}\b', sub_text)]

        if not existing_locals:
            print(f"  WARNING: No matching locals in {sub_name}")
            continue

        # Insert !$acc data create(...) before the anchor line
        vars_str = ", ".join(existing_locals)
        create_line = f"   !$acc data create({vars_str})"

        # Find the last ENDDO/RETURN before END SUBROUTINE for the end data
        end_data_line = None
        for i in range(sub_end - 1, anchor_line, -1):
            stripped = lines[i].strip().upper()
            if stripped == 'RETURN' or stripped == 'END SUBROUTINE':
                end_data_line = i
                break

        if end_data_line is None:
            end_data_line = sub_end

        # Insert !$acc end data before RETURN/END SUBROUTINE
        lines.insert(end_data_line, "   !$acc end data")
        # Insert !$acc data create before anchor
        lines.insert(anchor_line, create_line)
        lines.insert(anchor_line, "")

        patched += 1
        print(f"  Patched {sub_name}: create({vars_str})")

    # Write back
    with open(TARGET, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nDone: {patched} subroutines patched")


if __name__ == "__main__":
    main()
