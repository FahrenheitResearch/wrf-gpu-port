#!/usr/bin/env python3
"""
Patch start_em.f90 to call gpu_init_domain_data(grid) at the end of
start_domain_em, so that all grid arrays are copied to the GPU before
the first timestep.

This is the critical missing piece: without this call, gpu_init_domain_data
exists but never runs, and all !$acc present() clauses fail at runtime.
"""
import os, sys, re

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

TARGET = os.path.join(WRF_DIR, "dyn_em", "start_em.f90")

with open(TARGET) as f:
    text = f.read()

# Idempotency check
if "gpu_init_domain_data" in text:
    print(f"SKIP: gpu_init_domain_data already present in {TARGET}")
    sys.exit(0)

lines = text.split("\n")
patched = False

# Strategy: Find "END SUBROUTINE start_domain_em" and insert the call
# just before the RETURN that precedes it.
#
# We also need to add the USE statement at the top of the subroutine.

# Step 1: Find the subroutine start to add USE statement
sub_start = -1
for i, line in enumerate(lines):
    if re.search(r'SUBROUTINE\s+start_domain_em\s*\(', line, re.IGNORECASE):
        sub_start = i
        break

if sub_start < 0:
    print(f"ERROR: Could not find SUBROUTINE start_domain_em in {TARGET}")
    sys.exit(1)

# Find a USE statement block to insert after (add after last USE in the subroutine)
last_use = sub_start
for i in range(sub_start + 1, min(sub_start + 100, len(lines))):
    stripped = lines[i].strip().upper()
    if stripped.startswith("USE "):
        last_use = i
    elif stripped.startswith("IMPLICIT") or (stripped and not stripped.startswith("!") and not stripped.startswith("#") and not stripped.startswith("USE") and "&" not in lines[i-1] if i > 0 else False):
        # Stop at first non-USE, non-comment, non-continuation line
        if "&" not in lines[max(0, i-1)]:
            break

# Insert USE statement
use_line = "   USE module_domain, ONLY : gpu_init_domain_data"
lines.insert(last_use + 1, use_line)
print(f"  Inserted USE statement at line {last_use + 2}")

# Step 2: Find END SUBROUTINE start_domain_em and insert CALL before the RETURN
end_sub = -1
for i in range(len(lines) - 1, sub_start, -1):
    if re.search(r'END\s+SUBROUTINE\s+start_domain_em', lines[i], re.IGNORECASE):
        end_sub = i
        break

if end_sub < 0:
    print(f"ERROR: Could not find END SUBROUTINE start_domain_em in {TARGET}")
    sys.exit(1)

# Find the RETURN statement before END SUBROUTINE
return_line = -1
for i in range(end_sub - 1, max(end_sub - 20, sub_start), -1):
    if lines[i].strip().upper() == "RETURN":
        return_line = i
        break

if return_line < 0:
    # No RETURN found, insert before END SUBROUTINE
    insert_at = end_sub
else:
    insert_at = return_line

# Insert the call block
call_lines = [
    "",
    "! --- GPU OpenACC: Transfer grid data to device ---",
    "   CALL gpu_init_domain_data(grid)",
    "   WRITE(*,*) 'GPU_OPENACC: Domain data transferred to GPU'",
    "",
]

for j, cl in enumerate(call_lines):
    lines.insert(insert_at + j, cl)

print(f"  Inserted CALL gpu_init_domain_data at line {insert_at + 1}")

# Write back
with open(TARGET, "w") as f:
    f.write("\n".join(lines))

print(f"PATCHED: {TARGET}")
print(f"  Added USE module_domain, ONLY : gpu_init_domain_data")
print(f"  Added CALL gpu_init_domain_data(grid) before end of start_domain_em")
