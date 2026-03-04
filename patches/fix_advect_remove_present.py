#!/usr/bin/env python3
"""
fix_advect_remove_present.py — Remove present() clauses from advection kernels.

The advection routines receive arrays as arguments from rk_tendency in module_em.
Many of these arrays (ru, rv, rom, etc.) are either:
  - Grid arrays not yet on GPU (grid%ru, grid%rv computed by rk_step_prep on CPU)
  - Local variables computed in rk_tendency on CPU (rom)

Using present() causes runtime failures because these arrays aren't in the
present table. Removing present() lets NVHPC use implicit data management
(copies data as needed). This is slower but correct.

Run AFTER all other advection patches (patch_advect_gpu.py, etc.).
"""

import os
import sys
import re

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

ADVECT_FILE = os.path.join(WRF_DIR, "dyn_em", "module_advect_em.f90")

if not os.path.exists(ADVECT_FILE):
    print(f"ERROR: {ADVECT_FILE} not found")
    sys.exit(1)

text = open(ADVECT_FILE).read()

# Replace "!$acc kernels present(...)" with "!$acc kernels"
# The present() clause can span multiple lines with continuations
count = 0

# Single-line pattern: !$acc kernels present(...)
pattern = re.compile(r'(!\$acc\s+kernels)\s+present\([^)]+\)', re.IGNORECASE)
new_text, n = pattern.subn(r'\1', text)
count += n

# Also handle "!$acc parallel loop ... present(...)"
pattern2 = re.compile(r'(!\$acc\s+parallel\s+loop[^!]*?)\s+present\([^)]+\)', re.IGNORECASE)
new_text, n2 = pattern2.subn(r'\1', new_text)
count += n2

if count == 0:
    print("No present() clauses found in advection kernels — nothing to fix.")
else:
    open(ADVECT_FILE, 'w').write(new_text)
    print(f"Removed present() from {count} ACC directives in {ADVECT_FILE}")
