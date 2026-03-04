#!/usr/bin/env python3
"""
fix_advect_pd_race.py — Fix the PD limiter write-write race condition in advection.

ROOT CAUSE: advect_scalar_pd and advect_scalar_wenopd have a positive-definite
flux limiter that modifies fqx(i+1), fqy(j+1), fqz(k+1) from cell (i,k,j).
Adjacent cells write to the same flux face → race condition on GPU.

FIX: Replace the racy single-pass limiter with a race-free two-pass algorithm
(already proven in advect_scalar_mono):
  Pass 1: Compute scale_out(i,k,j) per cell (fully parallel, each cell writes own index)
  Pass 2: Apply per-face: fqx(i,k,j) = min(1, scale_out(i-1,k,j)) * fqx(i,k,j)
           (fully parallel, each face written once)

Also adds scale_out array declaration and ACC create() for it.
"""
import os, sys, re

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

TARGET = os.path.join(WRF_DIR, "dyn_em", "module_advect_em.f90")

with open(TARGET) as f:
    text = f.read()

# Idempotency check
if "scale_out" in text and "fix_advect_pd_race" in text:
    print(f"SKIP: PD race fix already applied to {TARGET}")
    sys.exit(0)

lines = text.split("\n")
changes = 0

# ============================================================
# Pattern: Find the racy PD limiter block and replace it
# The racy block looks like:
#   DO j=j_start, j_end
#   DO k=kts, ktf
#   DO i=i_start, i_end
#     IF( flux_out(i,k,j) .gt. ph_low(i,k,j) ) THEN
#       scale = max(0.,ph_low(i,k,j)/(flux_out(i,k,j)+eps))
#       IF( fqx (i+1,k,j) .gt. 0.) fqx(i+1,k,j) = scale*fqx(i+1,k,j)
#       ...
#     END IF
#   ENDDO x3
#
# We need to find ALL instances (advect_scalar_pd and advect_scalar_wenopd)
# ============================================================

def find_racy_limiter(lines, start_from=0):
    """Find the start of a racy PD limiter block.
    Returns (loop_start, loop_end) or (-1, -1) if not found."""
    # Look for the signature pattern: flux_out(i,k,j) .gt. ph_low(i,k,j)
    # followed by fqx(i+1,k,j)
    for i in range(start_from, len(lines)):
        if 'flux_out(i,k,j) .gt. ph_low(i,k,j)' in lines[i]:
            # Check next few lines for the racy pattern (fqx(i+1,...))
            for j in range(i+1, min(i+10, len(lines))):
                if 'fqx(i+1,k,j)' in lines[j] or 'fqx (i+1,k,j)' in lines[j]:
                    # Found it. Now walk backwards to find the DO j loop start
                    loop_start = i
                    for k in range(i-1, max(i-20, 0), -1):
                        stripped = lines[k].strip()
                        if stripped.startswith('DO j=') or stripped.startswith('DO j ='):
                            loop_start = k
                            break
                        # Also check for ACC directive before the loop
                        if '!$acc' in stripped or '!!$acc' in stripped:
                            loop_start = k
                            break

                    # Walk forward to find the end (three nested ENDDO + optional acc end)
                    enddo_count = 0
                    loop_end = i
                    for k in range(i, min(i+30, len(lines))):
                        stripped = lines[k].strip().upper()
                        if stripped.startswith('ENDDO') or stripped.startswith('END DO'):
                            enddo_count += 1
                            if enddo_count == 3:
                                loop_end = k
                                # Check for ACC end directive after
                                for m in range(k+1, min(k+3, len(lines))):
                                    if '!$acc end' in lines[m] or '!!$acc end' in lines[m]:
                                        loop_end = m
                                        break
                                break

                    return (loop_start, loop_end)
    return (-1, -1)


def make_twopass_limiter(indent="   "):
    """Generate the race-free two-pass PD limiter code."""
    code = f"""
{indent}! --- Race-free two-pass PD limiter (fix_advect_pd_race.py) ---
{indent}! Pass 1: compute per-cell scale factor (fully parallel)
{indent}!$acc parallel loop collapse(3) gang vector present(flux_out, ph_low, scale_out)
{indent}DO j=j_start, j_end
{indent}DO k=kts, ktf
{indent}DO i=i_start, i_end
{indent}  scale_out(i,k,j) = 1.0
{indent}  IF( flux_out(i,k,j) .gt. ph_low(i,k,j) ) THEN
{indent}    scale_out(i,k,j) = max(0., ph_low(i,k,j)/(flux_out(i,k,j)+eps))
{indent}  END IF
{indent}ENDDO
{indent}ENDDO
{indent}ENDDO
{indent}!$acc end parallel loop
{indent}
{indent}! Pass 2a: apply scale to x-fluxes (each face written once)
{indent}!$acc parallel loop collapse(3) gang vector present(fqx, scale_out)
{indent}DO j=j_start, j_end
{indent}DO k=kts, ktf
{indent}DO i=i_start, i_end+1
{indent}  IF( fqx(i,k,j) .gt. 0. .and. i > i_start) THEN
{indent}    fqx(i,k,j) = min(1.0, scale_out(i-1,k,j)) * fqx(i,k,j)
{indent}  ELSE IF( fqx(i,k,j) .lt. 0. .and. i <= i_end) THEN
{indent}    fqx(i,k,j) = min(1.0, scale_out(i,k,j)) * fqx(i,k,j)
{indent}  END IF
{indent}ENDDO
{indent}ENDDO
{indent}ENDDO
{indent}!$acc end parallel loop
{indent}
{indent}! Pass 2b: apply scale to y-fluxes (each face written once)
{indent}!$acc parallel loop collapse(3) gang vector present(fqy, scale_out)
{indent}DO j=j_start, j_end+1
{indent}DO k=kts, ktf
{indent}DO i=i_start, i_end
{indent}  IF( fqy(i,k,j) .gt. 0. .and. j > j_start) THEN
{indent}    fqy(i,k,j) = min(1.0, scale_out(i,k,j-1)) * fqy(i,k,j)
{indent}  ELSE IF( fqy(i,k,j) .lt. 0. .and. j <= j_end) THEN
{indent}    fqy(i,k,j) = min(1.0, scale_out(i,k,j)) * fqy(i,k,j)
{indent}  END IF
{indent}ENDDO
{indent}ENDDO
{indent}ENDDO
{indent}!$acc end parallel loop
{indent}
{indent}! Pass 2c: apply scale to z-fluxes (each face written once)
{indent}! Note: z flux sign is opposite in mass coordinate
{indent}!$acc parallel loop collapse(3) gang vector present(fqz, scale_out)
{indent}DO j=j_start, j_end
{indent}DO k=kts, ktf+1
{indent}DO i=i_start, i_end
{indent}  IF( fqz(i,k,j) .lt. 0. .and. k > kts) THEN
{indent}    fqz(i,k,j) = min(1.0, scale_out(i,k-1,j)) * fqz(i,k,j)
{indent}  ELSE IF( fqz(i,k,j) .gt. 0. .and. k <= ktf) THEN
{indent}    fqz(i,k,j) = min(1.0, scale_out(i,k,j)) * fqz(i,k,j)
{indent}  END IF
{indent}ENDDO
{indent}ENDDO
{indent}ENDDO
{indent}!$acc end parallel loop
{indent}! --- End race-free PD limiter ---
"""
    return code.strip().split("\n")


# Find and replace ALL racy limiter instances
search_from = 0
instance = 0
while True:
    start, end = find_racy_limiter(lines, search_from)
    if start < 0:
        break

    instance += 1
    print(f"  Found racy PD limiter instance {instance} at lines {start+1}-{end+1}")

    # Get indentation from original code
    indent = "   "

    # Replace the racy block with the two-pass version
    replacement = make_twopass_limiter(indent)
    lines = lines[:start] + replacement + lines[end+1:]

    # Adjust search position
    search_from = start + len(replacement)
    changes += 1

print(f"  Replaced {changes} racy PD limiter instances")

# ============================================================
# Add scale_out array declaration to advect_scalar_pd and advect_scalar_wenopd
# It needs the same dimensions as flux_out: (its-1:ite+2, kts:kte, jts-1:jte+2)
# ============================================================

text = "\n".join(lines)

# Find flux_out declarations and add scale_out after them
decl_pattern = r'(REAL,DIMENSION\(\s*its-1:ite\+2,\s*kts:kte,\s*jts-1:jte\+2\s*\)\s*::\s*flux_out,\s*ph_low)'
matches = list(re.finditer(decl_pattern, text))
print(f"  Found {len(matches)} flux_out declarations to augment with scale_out")

# Replace from end to start to preserve positions
for match in reversed(matches):
    original = match.group(0)
    if 'scale_out' not in text[match.start():match.start()+200]:
        augmented = original + "\n   REAL,DIMENSION( its-1:ite+2, kts:kte, jts-1:jte+2  ) :: scale_out"
        text = text[:match.start()] + augmented + text[match.end():]
        changes += 1
        print(f"  Added scale_out declaration")

# Also add scale_out to any !$acc data create blocks that contain flux_out
# Search for acc directives that mention flux_out and add scale_out
text = text.replace("flux_out, ph_low)", "flux_out, ph_low, scale_out)")

with open(TARGET, "w") as f:
    f.write(text)

print(f"\nPATCHED: {TARGET}")
print(f"  Total changes: {changes}")
print(f"  Racy PD limiters replaced with race-free two-pass algorithm")
print(f"  Added scale_out array declarations")
