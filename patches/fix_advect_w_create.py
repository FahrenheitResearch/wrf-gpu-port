#!/usr/bin/env python3
"""Add !$acc data create(vflux, fqx, fqy) to advect_w subroutine."""

TARGET = "/home/drew/WRF_BUILD_GPU/dyn_em/module_advect_em.f90"

with open(TARGET) as f:
    lines = f.readlines()

# Check if already patched
for line in lines:
    if 'data create(vflux, fqx, fqy)' in line:
        # Check if this is in advect_w range (not advect_u or advect_v)
        pass  # We'll check range below

# Find advect_w subroutine
sub_start = None
sub_end = None
for i, line in enumerate(lines):
    stripped = line.strip().upper()
    if 'SUBROUTINE ADVECT_W' in stripped and not stripped.startswith('END') and sub_start is None:
        sub_start = i
    if sub_start is not None and sub_end is None and stripped.startswith('END SUBROUTINE ADVECT_W'):
        sub_end = i
        break

if sub_start is None or sub_end is None:
    print(f"ERROR: Could not find advect_w (start={sub_start}, end={sub_end})")
    exit(1)

print(f"advect_w: lines {sub_start+1}-{sub_end+1}")

# Check if already has data create in this range
for i in range(sub_start, sub_end):
    if '!$acc data create' in lines[i]:
        print("Already patched — skipping")
        exit(0)

# Find anchor: specified = .false.
anchor_line = None
for i in range(sub_start, sub_end):
    if 'specified = .false.' in lines[i]:
        anchor_line = i
        break

if anchor_line is None:
    print("ERROR: Could not find anchor 'specified = .false.'")
    exit(1)

# Find RETURN or END SUBROUTINE for end data placement
end_data_line = sub_end
for i in range(sub_end - 1, anchor_line, -1):
    stripped = lines[i].strip().upper()
    if stripped == 'RETURN' or stripped.startswith('END SUBROUTINE'):
        end_data_line = i
        break

print(f"  Anchor at line {anchor_line+1}")
print(f"  End data before line {end_data_line+1}")

# Insert end data first (so line numbers don't shift for anchor)
lines.insert(end_data_line, "   !$acc end data\n")
# Insert create before anchor
lines.insert(anchor_line, "   !$acc data create(vflux, fqx, fqy)\n")
lines.insert(anchor_line, "\n")

with open(TARGET, 'w') as f:
    f.writelines(lines)

print("  Patched advect_w: create(vflux, fqx, fqy)")
