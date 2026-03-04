#!/usr/bin/env python3
"""
build_gpu_init_targeted.py — Create gpu_init_domain_data with ONLY the arrays
actually needed by our GPU dynamics + physics coupling patches.

Instead of scanning all 2588 grid fields and blindly copying 1577 arrays
(many of which are NULL pointers for unused physics options), this script
adds ONLY the arrays that appear in present() clauses of our patched modules.

Each copyin is guarded with ASSOCIATED() to handle optional/unallocated arrays.

Must run AFTER create_gpu_init.py (which creates the subroutine stub).

Usage:
    WRF_DIR=/path/to/WRF python build_gpu_init_targeted.py
"""

import os
import sys
import re
from pathlib import Path

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

DOMAIN_FILE = Path(WRF_DIR) / "frame" / "module_domain.f90"

# ── Arrays needed for GPU dynamics patches ──────────────────────────
# These are organized by which module needs them.
# All are grid% arrays that must be device-resident for present() clauses.

# Core dynamics state (used by small_step, big_step, module_em, solve_em)
DYNAMICS_CORE = [
    # Velocity and mass
    "u_1", "u_2", "v_1", "v_2", "w_1", "w_2",
    "t_1", "t_2", "ph_1", "ph_2", "phb",
    "mu_1", "mu_2", "mub", "muts", "mudf",
    # Geopotential / pressure
    "p", "pb", "al", "alb", "alt", "php",
    "p_hyd", "p_hyd_w",
    # Map scale factors (unstaggered and staggered)
    "msfux", "msfuy", "msfvx", "msfvy", "msftx", "msfty",
    "msfvx_inv",
    # Coriolis, topography
    "f", "e", "sina", "cosa", "ht",
    # Vertical coordinate coefficients
    "c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f",
    "dn", "dnw", "rdn", "rdnw", "fnm", "fnp",
    # Coupled velocity
    "ru_m", "rv_m", "ww_m",
    "ru_s", "rv_s", "ww_s",
    # Tendencies
    "ru_tend", "rv_tend",
    "h_diabatic", "mut",
    # Scalars
    "qv_diabatic", "qc_diabatic",
]

# Physics tendency arrays (used by calculate_phy_tend in module_em)
PHYSICS_TENDENCIES = [
    # PBL tendencies
    "rublten", "rvblten", "rthblten",
    "rqvblten", "rqcblten", "rqiblten",
    # Radiation tendencies
    "rthraten", "rthratenlw", "rthratensw",
    # Cumulus tendencies
    "rucuten", "rvcuten", "rthcuten",
    "rqvcuten", "rqccuten", "rqicuten", "rqrcuten", "rqscuten",
    # Shallow convection
    "rushten", "rvshten", "rthshten",
    "rqvshten", "rqcshten", "rqishten", "rqrshten", "rqsshten",
    # FDDA nudging
    "rundgdten", "rvndgdten", "rthndgdten", "rqvndgdten",
    # IAU tendencies
    "ruiaten", "rviaten", "rthiaten",
    # Other physics
    "rthften", "rqvften",
]

# Diffusion arrays
DIFFUSION = [
    "xkmh", "xkmv", "xkhh", "xkhv",
    "defor11", "defor22", "defor33",
    "defor12", "defor13", "defor23",
    "div", "tke_1", "tke_2",
    "bn2",
]

# Microphysics coupling
MICROPHYSICS = [
    "rainnc", "rainncv", "snownc", "snowncv",
    "graupelnc", "graupelncv", "sr",
    "re_cloud", "re_ice", "re_snow",
]

# Surface/boundary layer
SURFACE = [
    "tsk", "xland", "lakemask", "ivgtyp", "isltyp",
    "ust", "pblh", "hfx", "qfx", "lh",
    "znt", "mol", "rmol", "br", "regime",
    "flhc", "flqc", "exch_h", "exch_m",
    "t2", "q2", "psfc", "u10", "v10", "th2",
    "wspd", "qsfc",
]

# Moisture (4D array)
MOISTURE_4D = [
    "moist",
]

ALL_ARRAYS = (DYNAMICS_CORE + PHYSICS_TENDENCIES + DIFFUSION +
              MICROPHYSICS + SURFACE + MOISTURE_4D)


def main():
    if not DOMAIN_FILE.exists():
        print(f"ERROR: {DOMAIN_FILE} not found")
        sys.exit(1)

    text = DOMAIN_FILE.read_text()

    if "SUBROUTINE gpu_init_domain_data" not in text:
        print("ERROR: gpu_init_domain_data not found — run create_gpu_init.py first")
        sys.exit(1)

    # Check if already populated
    if "acc enter data copyin(grid%" in text.lower():
        # Count existing entries
        n = text.lower().count("acc enter data copyin(grid%")
        print(f"gpu_init_domain_data already has {n} copyin directives")
        if n > 50:
            print("  Looks like build_gpu_init_from_struct.py already ran.")
            print("  Replacing with targeted version...")
            # Remove existing copyin lines and rebuild
            lines = text.split("\n")
            new_lines = []
            in_gpu_init = False
            skip_copyin = False
            for line in lines:
                if "SUBROUTINE gpu_init_domain_data" in line and "END" not in line:
                    in_gpu_init = True
                    new_lines.append(line)
                    continue
                if in_gpu_init and "END SUBROUTINE gpu_init_domain_data" in line:
                    in_gpu_init = False
                    # Re-insert the body
                    new_lines.append("    ! Targeted GPU init — only needed arrays")
                    new_lines.append("    WRITE(*,*) '  GPU_OPENACC: gpu_init_domain_data called'")
                    new_lines.append(_build_copyin_block())
                    new_lines.append(line)
                    continue
                if in_gpu_init:
                    # Skip existing body
                    continue
                new_lines.append(line)
            text = "\n".join(new_lines)
            DOMAIN_FILE.write_text(text)
            print(f"Replaced gpu_init_domain_data with {len(ALL_ARRAYS)} targeted arrays")
            return

    # Insert copyin statements into the existing stub
    # Find the WRITE(*,*) line inside gpu_init_domain_data and insert after it
    pattern = r"(WRITE\(\*,\*\) '  GPU_OPENACC: gpu_init_domain_data called')"
    match = re.search(pattern, text)
    if not match:
        print("ERROR: Could not find WRITE statement in gpu_init_domain_data")
        sys.exit(1)

    insert_pos = match.end()
    copyin_block = "\n" + _build_copyin_block()

    new_text = text[:insert_pos] + copyin_block + text[insert_pos:]
    DOMAIN_FILE.write_text(new_text)
    print(f"Added {len(ALL_ARRAYS)} targeted copyin directives to gpu_init_domain_data")


def _build_copyin_block():
    """Build Fortran copyin statements with ASSOCIATED() guards."""
    lines = []
    lines.append("    ! --- Targeted GPU data init (dynamics + physics coupling) ---")

    # Group into chunks for readability
    for arr in ALL_ARRAYS:
        lines.append(f"    IF (ASSOCIATED(grid%{arr})) THEN")
        lines.append(f"      !$acc enter data copyin(grid%{arr})")
        lines.append(f"    END IF")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
