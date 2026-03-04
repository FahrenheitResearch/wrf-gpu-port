#!/usr/bin/env python3
"""Replace !$acc kernels wrapping j-loops with !$acc parallel loop gang.

This fuses multiple auto-generated CUDA kernels into one, reducing kernel
launch overhead from ~12 launches/call to 1 launch/call.

Target: functions with one !$acc kernels wrapping a j-loop that contains
many inner loop nests (each generating a separate CUDA kernel).
"""

import re
import os, sys

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

def process_file(path):
    with open(path) as f:
        lines = f.readlines()

    changes = 0
    i = 0
    while i < len(lines):
        s = lines[i].strip()

        # Pattern: !$acc kernels followed by a labeled or bare j-loop
        if s == '!$acc kernels':
            # Look at next non-blank line for j-loop
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                j += 1

            if j < len(lines):
                next_line = lines[j].strip()
                # Match patterns like:
                #   j_loop_w: DO j = j_start, j_end
                #   DO j = j_start, j_end
                #   u_outer_j_loop: DO j = j_start, j_end
                #   outer_j_loop: DO j = j_start, j_end
                if re.match(r'(\w+:\s+)?DO\s+j\s*=', next_line, re.IGNORECASE):
                    # Find the matching !$acc end kernels
                    depth = 1
                    for k in range(i+1, len(lines)):
                        sk = lines[k].strip()
                        if sk == '!$acc kernels':
                            depth += 1
                        elif sk == '!$acc end kernels':
                            depth -= 1
                            if depth == 0:
                                # Replace !$acc kernels with !$acc parallel loop gang
                                indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                                lines[i] = indent + '!$acc parallel loop gang\n'
                                # Remove !$acc end kernels
                                lines[k] = ''
                                changes += 1
                                fname = "?"
                                # Find function name
                                for m in range(i, -1, -1):
                                    sm = lines[m].strip().upper()
                                    if sm.startswith('SUBROUTINE '):
                                        fname = re.search(r'SUBROUTINE\s+(\w+)', sm, re.IGNORECASE).group(1)
                                        break
                                print(f"  Line {i+1}: {fname} - fused kernels→parallel loop gang (was {k-i} lines)")
                                break
        i += 1

    if changes:
        with open(path, 'w') as f:
            f.writelines(lines)
        print(f"\n  {changes} kernel regions fused in {path}")
    else:
        print(f"  No fusable regions found in {path}")

    return changes

# Process dynamics files
files = [
    os.path.join(WRF_DIR, "dyn_em", "module_small_step_em.f90"),
    os.path.join(WRF_DIR, "dyn_em", "module_advect_em.f90"),
    os.path.join(WRF_DIR, "dyn_em", "module_big_step_utilities_em.f90"),
]

total = 0
for f in files:
    print(f"\n=== Processing {f.split('/')[-1]} ===")
    total += process_file(f)

print(f"\n=== Total: {total} regions fused ===")
