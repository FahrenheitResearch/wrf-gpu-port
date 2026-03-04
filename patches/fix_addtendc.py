#!/usr/bin/env python3
"""Fix duplicate !$acc parallel loop directives in module_physics_addtendc.f90"""
import os, sys

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

path = os.path.join(WRF_DIR, "phys", "module_physics_addtendc.f90")
with open(path) as f:
    lines = f.readlines()

fixed = 0
new_lines = []
i = 0
while i < len(lines):
    stripped = lines[i].strip()
    if stripped.startswith("!$acc parallel loop"):
        new_lines.append(lines[i])
        # Check if next line is also an acc parallel loop directive
        if i + 1 < len(lines) and lines[i + 1].strip().startswith("!$acc parallel loop"):
            # Skip the duplicate
            i += 2
            fixed += 1
            continue
    else:
        new_lines.append(lines[i])
    i += 1

with open(path, 'w') as f:
    f.writelines(new_lines)
print(f"Fixed {fixed} duplicate directives")
