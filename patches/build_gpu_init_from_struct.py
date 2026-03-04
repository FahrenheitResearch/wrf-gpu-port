#!/usr/bin/env python3
"""
Build comprehensive gpu_init by cross-referencing:
1. grid% fields used in solve_em + first_rk_step_part1/2 + module_em
2. state_struct.inc to identify which are arrays (POINTER with DIMENSION)
3. Existing gpu_init entries

Only adds ARRAY fields (not scalars/strings/derived types).
"""
import re
import os

WRF = "/home/drew/WRF_BUILD_GPU"
DOMAIN_FILE = os.path.join(WRF, "frame/module_domain.f90")
STRUCT_FILE = os.path.join(WRF, "inc/state_struct.inc")

# Files where grid% fields are used and may need GPU presence
SCAN_FILES = [
    "dyn_em/solve_em.f90",
    "dyn_em/module_em.f90",
    "dyn_em/module_first_rk_step_part1.f90",
    "dyn_em/module_first_rk_step_part2.f90",
]

def main():
    # Step 1: Parse state_struct.inc to build field→type map
    print("=== Parsing state_struct.inc ===")
    array_fields = set()  # fields that are POINTER arrays
    scalar_fields = set()  # scalar fields

    with open(STRUCT_FILE) as f:
        for line in f:
            line = line.strip()
            if '::' not in line:
                continue
            # Extract field name (last token after ::)
            parts = line.split('::')
            if len(parts) < 2:
                continue
            fieldname = parts[1].strip().lower()
            # Check if it has DIMENSION
            if 'DIMENSION' in parts[0].upper():
                array_fields.add(fieldname)
            elif 'POINTER' not in parts[0].upper():
                # Scalar (not pointer, not array)
                scalar_fields.add(fieldname)
            else:
                # POINTER without DIMENSION — likely scalar pointer or special
                scalar_fields.add(fieldname)

    print(f"  Arrays in grid type: {len(array_fields)}")
    print(f"  Scalars in grid type: {len(scalar_fields)}")

    # Step 2: Scan dynamics files for grid% references
    print("\n=== Scanning dynamics files for grid% references ===")
    used_fields = set()
    for relpath in SCAN_FILES:
        filepath = os.path.join(WRF, relpath)
        if not os.path.exists(filepath):
            print(f"  SKIP: {relpath}")
            continue
        with open(filepath) as f:
            text = f.read()
        fields = set(m.lower() for m in re.findall(r'grid%(\w+)', text, re.IGNORECASE))
        print(f"  {relpath}: {len(fields)} unique grid% fields")
        used_fields |= fields

    print(f"  Total unique grid% fields across all files: {len(used_fields)}")

    # Step 3: Filter to only arrays that are used
    needed_arrays = used_fields & array_fields
    needed_scalars = used_fields & scalar_fields
    unknown = used_fields - array_fields - scalar_fields

    print(f"\n=== Classification ===")
    print(f"  Arrays referenced in dynamics: {len(needed_arrays)}")
    print(f"  Scalars referenced in dynamics: {len(needed_scalars)} (auto firstprivate)")
    print(f"  Unknown (not in state_struct): {len(unknown)}")
    if unknown:
        print(f"    Examples: {', '.join(sorted(unknown)[:10])}")

    # Step 4: Read existing gpu_init
    print(f"\n=== Reading existing gpu_init ===")
    with open(DOMAIN_FILE) as f:
        domain_lines = f.readlines()

    existing = set()
    in_sub = False
    last_acc_line = -1
    sub_end = -1

    for i, line in enumerate(domain_lines):
        if 'SUBROUTINE gpu_init_domain_data' in line and 'END' not in line.upper().split('!')[0]:
            in_sub = True
            continue
        if in_sub and 'END SUBROUTINE' in line.upper() and 'gpu_init' in line.lower():
            sub_end = i
            in_sub = False
            continue
        if in_sub and '!$acc' in line.lower():
            last_acc_line = i
            for m in re.finditer(r'grid%(\w+)', line, re.IGNORECASE):
                existing.add(m.group(1).lower())

    print(f"  Already in gpu_init: {len(existing)}")

    # Step 5: Find missing arrays
    missing = needed_arrays - existing

    # Exclude known non-data fields
    EXCLUDE = {
        'head_grid', 'tail_grid', 'next_grid', 'nests', 'parents',
        'child_of_parent', 'alarms', 'alarms_created',
        'io_intervals', 'iofields_params', 'return_after',
        'ts_filename', 'nametsloc', 'desctsloc',
        'track_i', 'track_j', 'track_lat_domain', 'track_lon_domain',
        'track_lat_in', 'track_lon_in', 'track_time_domain', 'track_time_in',
    }
    missing -= EXCLUDE

    print(f"  Missing arrays to add: {len(missing)}")

    if not missing:
        print("\nAll needed arrays are already in gpu_init!")
        return

    # Step 6: Classify missing arrays by dimension
    dim_map = {}  # field -> ndims
    with open(STRUCT_FILE) as f:
        for line in f:
            if '::' not in line or 'DIMENSION' not in line.upper():
                continue
            parts = line.split('::')
            fieldname = parts[1].strip().lower()
            if fieldname in missing:
                # Count commas in DIMENSION clause to determine ndims
                dim_match = re.search(r'DIMENSION\(([^)]+)\)', parts[0], re.IGNORECASE)
                if dim_match:
                    ndims = dim_match.group(1).count(',') + 1
                    dim_map[fieldname] = ndims

    # Group by dimension count
    by_dims = {}
    for f in sorted(missing):
        nd = dim_map.get(f, 0)
        by_dims.setdefault(nd, []).append(f)

    print(f"\n=== Missing arrays by dimension ===")
    for nd in sorted(by_dims.keys()):
        print(f"  {nd}D: {len(by_dims[nd])} fields")

    # Step 7: Generate copyin statements
    new_lines = []
    new_lines.append("\n")
    new_lines.append("      ! --- BEGIN auto-generated copyin (build_gpu_init_from_struct.py) ---\n")

    for nd in sorted(by_dims.keys()):
        fields = by_dims[nd]
        new_lines.append(f"      ! {nd}D arrays ({len(fields)})\n")
        for i in range(0, len(fields), 3):
            chunk = fields[i:i+3]
            fields_str = ", ".join(f"grid%{f}" for f in chunk)
            new_lines.append(f"      !$acc enter data copyin({fields_str})\n")

    new_lines.append("      ! --- END auto-generated copyin ---\n")

    # Insert after last !$acc line
    insert_at = last_acc_line + 1
    print(f"\nInserting {len(new_lines)} lines at line {insert_at + 1}")

    updated = domain_lines[:insert_at] + new_lines + domain_lines[insert_at:]
    with open(DOMAIN_FILE, 'w') as f:
        f.writelines(updated)

    print(f"Done! Added copyin for {len(missing)} array fields.")

    # Print all added fields
    print(f"\n=== Added fields ===")
    for f in sorted(missing):
        print(f"  grid%{f} ({dim_map.get(f, '?')}D)")


if __name__ == "__main__":
    main()
