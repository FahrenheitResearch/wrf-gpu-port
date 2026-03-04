#!/usr/bin/env python3
"""Re-enable ACC directives (!!$acc -> !$acc) in the 4 dynamics files."""
import os
os.chdir("/home/drew/WRF_BUILD_GPU/dyn_em")
for fname in ["module_advect_em.f90", "module_big_step_utilities_em.f90",
              "module_diffusion_em.f90", "module_em.f90"]:
    with open(fname) as f:
        lines = f.readlines()
    count = 0
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("!!$acc") and not stripped.startswith("!!!$acc"):
            out.append(line.replace("!!$acc", "!$acc", 1))
            count += 1
        else:
            out.append(line)
    with open(fname, "w") as f:
        f.writelines(out)
    print(f"Re-enabled {count} ACC directives in {fname}")
