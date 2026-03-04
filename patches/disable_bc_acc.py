#!/usr/bin/env python3
"""Disable all ACC directives in module_bc.f90."""
import os, sys
WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)
path = os.path.join(WRF_DIR, "share", "module_bc.f90")
with open(path) as f:
    lines = f.readlines()
count = 0
out = []
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("!$acc"):
        out.append(line.replace("!$acc", "!!$acc", 1))
        count += 1
    else:
        out.append(line)
with open(path, "w") as f:
    f.writelines(out)
print(f"Disabled {count} ACC directives in module_bc.f90")
