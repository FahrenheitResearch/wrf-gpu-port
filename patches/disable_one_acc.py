#!/usr/bin/env python3
"""Disable ACC directives in a single .f90 file (!$acc -> !!$acc).
Usage: python disable_one_acc.py <filename>
"""
import sys, os
os.chdir("/home/drew/WRF_BUILD_GPU/dyn_em")
fname = sys.argv[1]
with open(fname) as f:
    lines = f.readlines()
count = 0
out = []
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("!$acc") and not stripped.startswith("!!$acc"):
        out.append(line.replace("!$acc", "!!$acc", 1))
        count += 1
    else:
        out.append(line)
with open(fname, "w") as f:
    f.writelines(out)
print(f"Disabled {count} ACC directives in {fname}")
