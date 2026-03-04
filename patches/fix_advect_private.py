#!/usr/bin/env python3
"""
fix_advect_private.py — Add missing private() clauses to !$acc parallel loop
directives in module_advect_em.f90.

With !$acc parallel loop, scalar variables computed inside the loop are SHARED
by default (unlike !$acc kernels which auto-privatizes). Loop-local scalars like
vel, cr, mu, mrdx, mrdy, scale must be private to each thread.

This script finds all !$acc parallel loop directives in the advection module
and adds private() clauses for common loop-local scalars.
"""
import os, sys, re

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

TARGET = os.path.join(WRF_DIR, "dyn_em", "module_advect_em.f90")

with open(TARGET) as f:
    text = f.read()

# Idempotency
if "fix_advect_private" in text:
    print(f"SKIP: private clause fix already applied to {TARGET}")
    sys.exit(0)

lines = text.split("\n")

# Common loop-local scalars in advection subroutines that need privatization
# These are computed inside DO loops and would race without private()
PRIVATE_VARS = [
    "vel", "cr", "mu", "mrdx", "mrdy", "scale",
    "dx", "dy", "dz", "dvm", "dvp",
    "cf1", "cf2", "cf3",
    "qip2", "qip1", "qi", "qim1", "qim2",
    "pw1", "pw2",
]

changes = 0
new_lines = []

for i, line in enumerate(lines):
    stripped = line.strip()

    # Match !$acc parallel loop directives (both enabled and disabled)
    if re.match(r'\s*!?\$acc\s+parallel\s+loop', stripped, re.IGNORECASE):
        # Check if private() already exists
        if 'private(' not in stripped.lower():
            # Find which vars are actually used in the loop body below
            # Look ahead ~50 lines for variable usage
            used_privates = []
            for var in PRIVATE_VARS:
                for j in range(i+1, min(i+80, len(lines))):
                    body_line = lines[j].strip()
                    # Stop at next ACC directive or ENDDO matching the parallel loop
                    if body_line.startswith('!$acc end') or body_line.startswith('!!$acc end'):
                        break
                    # Check if variable is used as an assignment target (left side of =)
                    if re.search(rf'\b{var}\b\s*=', body_line, re.IGNORECASE):
                        used_privates.append(var)
                        break

            if used_privates:
                # Add private clause to the directive
                private_str = ", ".join(used_privates)
                # Find the end of the directive (may span multiple lines with &)
                directive_end = i
                while directive_end < len(lines) - 1 and lines[directive_end].rstrip().endswith("&"):
                    directive_end += 1

                # Add private() before the last line of the directive
                last_line = lines[directive_end].rstrip()
                if last_line.endswith(")"):
                    # Append private clause
                    new_lines.append(line)
                    # Skip to directive_end, adding intermediate lines
                    for k in range(i+1, directive_end):
                        new_lines.append(lines[k])
                    if i != directive_end:
                        new_lines.append(lines[directive_end].rstrip() + f" private({private_str})")
                    else:
                        new_lines[-1] = new_lines[-1].rstrip() + f" private({private_str})"
                    changes += 1
                    continue
                else:
                    # Simple single-line directive
                    new_lines.append(line.rstrip() + f" private({private_str})")
                    changes += 1
                    continue

    new_lines.append(line)

# Add marker for idempotency
new_lines.insert(0, "! fix_advect_private applied")

with open(TARGET, "w") as f:
    f.write("\n".join(new_lines))

print(f"fix_advect_private.py: Added private() clauses to {changes} parallel loop directives in {TARGET}")
