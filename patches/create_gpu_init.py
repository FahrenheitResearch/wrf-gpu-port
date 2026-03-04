#!/usr/bin/env python3
"""
Create stub gpu_init_domain_data and gpu_finalize_domain_data subroutines
in WRF's module_domain.f90.

Must run BEFORE build_gpu_init_from_struct.py, which augments these
subroutines but does not create them.

Usage:
    WRF_DIR=/path/to/WRF python create_gpu_init.py
    python create_gpu_init.py /path/to/WRF
"""
import os
import sys
import re

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)

DOMAIN_FILE = os.path.join(WRF_DIR, "frame", "module_domain.f90")

if not os.path.isfile(DOMAIN_FILE):
    print(f"ERROR: {DOMAIN_FILE} not found")
    print("Run './compile em_real' first to generate .f90 files from .F sources.")
    sys.exit(1)

with open(DOMAIN_FILE, "r") as f:
    text = f.read()

# Idempotency: if subroutines already exist, skip
if "SUBROUTINE gpu_init_domain_data" in text:
    print(f"SKIP: gpu_init_domain_data already exists in {DOMAIN_FILE}")
    sys.exit(0)

STUB = """\

SUBROUTINE gpu_init_domain_data(grid)
    USE module_domain_type
    IMPLICIT NONE
    TYPE(domain), INTENT(INOUT) :: grid
    ! OpenACC data initialization - populated by build_gpu_init_from_struct.py
    WRITE(*,*) '  GPU_OPENACC: gpu_init_domain_data called'
END SUBROUTINE gpu_init_domain_data

SUBROUTINE gpu_finalize_domain_data(grid)
    USE module_domain_type
    IMPLICIT NONE
    TYPE(domain), INTENT(INOUT) :: grid
    ! OpenACC data finalization
    WRITE(*,*) '  GPU_OPENACC: gpu_finalize_domain_data called'
END SUBROUTINE gpu_finalize_domain_data

"""

# Find the final END MODULE line (case-insensitive)
# Pattern: END MODULE at the end of the file (possibly with module name after it)
pattern = re.compile(
    r'^(\s*END\s+MODULE\b.*?)$',
    re.IGNORECASE | re.MULTILINE,
)

matches = list(pattern.finditer(text))
if not matches:
    print("ERROR: Could not find 'END MODULE' in module_domain.f90")
    sys.exit(1)

# Use the last END MODULE match (the one closing module_domain)
last_match = matches[-1]
insert_pos = last_match.start()

new_text = text[:insert_pos] + STUB + text[insert_pos:]

with open(DOMAIN_FILE, "w") as f:
    f.write(new_text)

print(f"CREATED: gpu_init_domain_data + gpu_finalize_domain_data in {DOMAIN_FILE}")
print(f"  Inserted before final END MODULE at character offset {insert_pos}")
