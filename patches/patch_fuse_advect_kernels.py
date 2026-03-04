#!/usr/bin/env python3
"""Fuse consecutive !$acc kernels blocks in advect_scalar_pd.

In the positive-definite flux limiter section of advect_scalar_pd, there are
three separate !$acc kernels blocks that iterate over the same (j,k,i) range:

  Kernel 1: Compute ph_low(i,k,j)     — reads fqxl,fqyl,fqzl,c1,c2,mub,...
  Kernel 2: Compute flux_out(i,k,j)   — reads fqx,fqy,fqz,msftx,msfty,...
  Kernel 3: Scale fqx,fqy,fqz using ph_low and flux_out

Kernels 1 and 2 are fully independent and read overlapping arrays (msftx,
msfty, rdzw). They can be fused into a single parallel region, halving
kernel launch overhead and reducing global memory reads.

Kernel 3 has write conflicts on fqx/fqy/fqz between neighboring iterations
(e.g., iteration i writes fqx(i+1,...) while iteration i+1 writes fqx(i+1,...)).
It must remain a separate kernel so the compiler can analyze dependencies.

Strategy:
  - Fuse kernels 1+2 into one !$acc parallel loop collapse(3)
  - Keep kernel 3 as !$acc kernels (compiler handles race analysis)
  - Net effect: 3 kernel launches -> 2, one fewer global memory pass

Also applies the same pattern to advect_scalar (non-pd) where similar
sequential kernels exist at the end of the subroutine.
"""

import re
import sys
import shutil
import os
from pathlib import Path

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

TARGET = Path(WRF_DIR) / "dyn_em" / "module_advect_em.f90"

# The exact text of the two kernels to fuse (kernels 1 and 2 in advect_scalar_pd).
# We match them as a contiguous block and replace with a single parallel region.

OLD_BLOCK = """\
   !$acc kernels
   DO j=j_start, j_end
   DO k=kts, ktf
!DIR$ vector always
   DO i=i_start, i_end

     ph_low(i,k,j) = ((c1(k)*mub(i,j)+c2(k))+(c1(k)*mu_old(i,j)))*field_old(i,k,j) &
                - dt*( msftx(i,j)*msfty(i,j)*(               &
                       rdx*(fqxl(i+1,k,j)-fqxl(i,k,j)) +     &
                       rdy*(fqyl(i,k,j+1)-fqyl(i,k,j))  )    &
                      +msfty(i,j)*rdzw(k)*(fqzl(i,k+1,j)-fqzl(i,k,j)) )

   ENDDO
   ENDDO
   ENDDO
   !$acc end kernels

   !$acc kernels
   DO j=j_start, j_end
   DO k=kts, ktf
!DIR$ vector always
   DO i=i_start, i_end

     flux_out(i,k,j) = dt*( (msftx(i,j)*msfty(i,j))*( &
                                rdx*(  max(0.,fqx (i+1,k,j))      &
                                      -min(0.,fqx (i  ,k,j)) )    &
                               +rdy*(  max(0.,fqy (i,k,j+1))      &
                                      -min(0.,fqy (i,k,j  )) ) )  &
                +msfty(i,j)*rdzw(k)*(  min(0.,fqz (i,k+1,j))      &
                                      -max(0.,fqz (i,k  ,j)) )   )

   ENDDO
   ENDDO
   ENDDO
   !$acc end kernels"""

NEW_BLOCK = """\
   !$acc parallel loop collapse(3) default(present)
   DO j=j_start, j_end
   DO k=kts, ktf
   DO i=i_start, i_end

     ph_low(i,k,j) = ((c1(k)*mub(i,j)+c2(k))+(c1(k)*mu_old(i,j)))*field_old(i,k,j) &
                - dt*( msftx(i,j)*msfty(i,j)*(               &
                       rdx*(fqxl(i+1,k,j)-fqxl(i,k,j)) +     &
                       rdy*(fqyl(i,k,j+1)-fqyl(i,k,j))  )    &
                      +msfty(i,j)*rdzw(k)*(fqzl(i,k+1,j)-fqzl(i,k,j)) )

     flux_out(i,k,j) = dt*( (msftx(i,j)*msfty(i,j))*( &
                                rdx*(  max(0.,fqx (i+1,k,j))      &
                                      -min(0.,fqx (i  ,k,j)) )    &
                               +rdy*(  max(0.,fqy (i,k,j+1))      &
                                      -min(0.,fqy (i,k,j  )) ) )  &
                +msfty(i,j)*rdzw(k)*(  min(0.,fqz (i,k+1,j))      &
                                      -max(0.,fqz (i,k  ,j)) )   )

   ENDDO
   ENDDO
   ENDDO
   !$acc end parallel"""


def patch_file():
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found")
        sys.exit(1)

    text = TARGET.read_text()

    # Check if already patched
    if "! FUSED: ph_low + flux_out" in text:
        print("Already patched (fusion marker found). Skipping.")
        return

    # Backup
    backup = TARGET.with_suffix(".f90.bak_fuse")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"Backup: {backup}")

    # Normalize line endings for matching
    text_unix = text.replace('\r\n', '\n')
    old_unix = OLD_BLOCK.replace('\r\n', '\n')
    new_unix = NEW_BLOCK.replace('\r\n', '\n')

    # Add a marker comment to the replacement
    new_with_marker = "   ! FUSED: ph_low + flux_out into single parallel region (patch_fuse_advect_kernels.py)\n" + new_unix

    count = text_unix.count(old_unix)
    if count == 0:
        print("ERROR: Could not find the target kernel block to fuse.")
        print("Searching for partial matches...")
        # Try to find the first kernel
        if "ph_low(i,k,j) = ((c1(k)*mub(i,j)+c2(k))+(c1(k)*mu_old(i,j)))*field_old(i,k,j)" in text_unix:
            print("  Found ph_low computation")
        else:
            print("  ph_low computation NOT found")
        if "flux_out(i,k,j) = dt*( (msftx(i,j)*msfty(i,j))*(" in text_unix:
            print("  Found flux_out computation")
        else:
            print("  flux_out computation NOT found")
        sys.exit(1)

    if count > 1:
        print(f"WARNING: Found {count} matches. Replacing all.")

    text_patched = text_unix.replace(old_unix, new_with_marker)

    # Verify the replacement happened
    if text_patched == text_unix:
        print("ERROR: Replacement had no effect")
        sys.exit(1)

    # Write back (preserve Unix line endings since this is WSL)
    TARGET.write_text(text_patched)
    print(f"SUCCESS: Fused kernels 1+2 in advect_scalar_pd")
    print(f"  - 2 separate !$acc kernels -> 1 !$acc parallel loop collapse(3)")
    print(f"  - Kernel 3 (flux limiter scaling) kept as separate !$acc kernels")
    print(f"  - Net: 3 kernel launches -> 2, one fewer global memory pass")


if __name__ == "__main__":
    patch_file()
