#!/usr/bin/env python3
"""Disable all ACC directives in module_bc.f90."""
path = "/home/drew/WRF_BUILD_GPU/share/module_bc.f90"
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
