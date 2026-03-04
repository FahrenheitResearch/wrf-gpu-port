#!/usr/bin/env python3
"""
patch_bc_gpu.py — Add OpenACC directives to set_physical_bc2d and set_physical_bc3d
in share/module_bc.f90 so boundary condition operations execute on GPU.

These routines are called 42 times per RK step from solve_em. Without GPU directives,
each call forces a device→host transfer of the array, CPU boundary update, then
host→device transfer back. With !$acc data present(dat) and !$acc parallel loop
on each loop nest, the boundary operations run on GPU with zero data transfer.

Strategy:
  - Wrap each routine body in !$acc data present(dat)
  - Add !$acc parallel loop before each DO loop nest
  - 2D loops (j,i or i only): collapse where possible
  - 3D loops (j,k,i or k,i): collapse where possible
  - bdyzone=4 so inner i-loops have tiny trip counts; collapse helps
  - Branching on config_flags stays on host (evaluated before loop launch)
  - Corner-fill loops (doubly periodic) also annotated for completeness

No computational logic is modified — only OpenACC directives are added.
"""

import sys

F90_PATH = "/home/drew/WRF_BUILD_GPU/share/module_bc.f90"


def patch_once(text, old, new, label=""):
    """Replace exactly one occurrence of old with new in text."""
    count = text.count(old)
    if count == 0:
        print(f"  WARNING: pattern not found for [{label}]")
        return text
    if count > 1:
        print(f"  INFO: {count} occurrences for [{label}], replacing first only")
    idx = text.index(old)
    return text[:idx] + new + text[idx + len(old):]


def patch_all(text, old, new, label=""):
    """Replace ALL occurrences of old with new in text."""
    count = text.count(old)
    if count == 0:
        print(f"  WARNING: pattern not found for [{label}]")
        return text
    print(f"  Replacing {count} occurrences for [{label}]")
    return text.replace(old, new)


def apply_patches(text):
    """Apply all OpenACC patches to module_bc.f90."""

    # ==================================================================
    # PART 1: set_physical_bc2d — Add !$acc data present(dat)
    # ==================================================================
    print("=== Patching set_physical_bc2d ===")

    # Add !$acc data present(dat) after variable declarations, before first compute
    # The first compute line after declarations is: debug = .false.
    # We find the unique context in bc2d (after the jstag declaration line)
    text = patch_once(text,
        "      LOGICAL  :: debug, open_bc_copy\n"
        "\n"
        "\n"
        "\n"
        "      debug = .false.\n"
        "\n"
        "      open_bc_copy = .false.\n"
        "\n"
        "      variable = variable_in\n"
        "      IF ( variable_in .ge. 'A' .and. variable_in .le. 'Z' ) THEN\n"
        "        variable = CHAR( ICHAR(variable_in) - ICHAR('A') + ICHAR('a') )\n"
        "      ENDIF\n"
        "      IF ((variable == 'u') .or. (variable == 'v') .or.  &\n"
        "          (variable == 'w') .or. (variable == 't') .or.  &\n"
        "          (variable == 'x') .or. (variable == 'y') .or.  &\n"
        "          (variable == 'r') .or. (variable == 'p') ) open_bc_copy = .true.",

        "      LOGICAL  :: debug, open_bc_copy\n"
        "\n"
        "!$acc data present(dat)\n"
        "\n"
        "      debug = .false.\n"
        "\n"
        "      open_bc_copy = .false.\n"
        "\n"
        "      variable = variable_in\n"
        "      IF ( variable_in .ge. 'A' .and. variable_in .le. 'Z' ) THEN\n"
        "        variable = CHAR( ICHAR(variable_in) - ICHAR('A') + ICHAR('a') )\n"
        "      ENDIF\n"
        "      IF ((variable == 'u') .or. (variable == 'v') .or.  &\n"
        "          (variable == 'w') .or. (variable == 't') .or.  &\n"
        "          (variable == 'x') .or. (variable == 'y') .or.  &\n"
        "          (variable == 'r') .or. (variable == 'p') ) open_bc_copy = .true.",
        "bc2d: data present")

    # Add !$acc end data before END SUBROUTINE set_physical_bc2d
    text = patch_once(text,
        "   END SUBROUTINE set_physical_bc2d",
        "!$acc end data\n"
        "   END SUBROUTINE set_physical_bc2d",
        "bc2d: end data")

    # --- bc2d periodicity_x: west boundary (j,i loop) ---
    text = patch_once(text,
        "          IF ( its == ids ) THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO i = 0,-(bdyzone-1),-1\n"
        "              dat(ids+i-1,j) = dat(ide+i-1,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ENDIF\n"
        "\n"
        "          IF ( ite == ide ) THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "\n"
        "            DO i = -istag , bdyzone\n"
        "              dat(ide+i+istag,j) = dat(ids+i+istag,j)\n"
        "            ENDDO\n"
        "            ENDDO",

        "          IF ( its == ids ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO i = 0,-(bdyzone-1),-1\n"
        "              dat(ids+i-1,j) = dat(ide+i-1,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ENDIF\n"
        "\n"
        "          IF ( ite == ide ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO i = -istag , bdyzone\n"
        "              dat(ide+i+istag,j) = dat(ids+i+istag,j)\n"
        "            ENDDO\n"
        "            ENDDO",
        "bc2d: periodic_x west/east")

    # --- bc2d symmetry_xs: non-u/x case ---
    text = patch_once(text,
        "          IF ( (variable /= 'u') .and. (variable /= 'x') ) THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO i = 1, bdyzone\n"
        "              dat(ids-i,j) = dat(ids+i-1,j) \n"
        "            ENDDO                             \n"
        "            ENDDO",

        "          IF ( (variable /= 'u') .and. (variable /= 'x') ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO i = 1, bdyzone\n"
        "              dat(ids-i,j) = dat(ids+i-1,j) \n"
        "            ENDDO                             \n"
        "            ENDDO",
        "bc2d: symmetry_xs non-u")

    # --- bc2d symmetry_xs: u case ---
    text = patch_once(text,
        "            IF( variable == 'u' ) THEN\n"
        "\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO i = 0, bdyzone-1\n"
        "                dat(ids-i,j) = - dat(ids+i,j) \n"
        "              ENDDO                             \n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO i = 0, bdyzone-1\n"
        "                dat(ids-i,j) =   dat(ids+i,j) \n"
        "              ENDDO                             \n"
        "              ENDDO",

        "            IF( variable == 'u' ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO i = 0, bdyzone-1\n"
        "                dat(ids-i,j) = - dat(ids+i,j) \n"
        "              ENDDO                             \n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO i = 0, bdyzone-1\n"
        "                dat(ids-i,j) =   dat(ids+i,j) \n"
        "              ENDDO                             \n"
        "              ENDDO",
        "bc2d: symmetry_xs u/x")

    # --- bc2d symmetry_xe: non-u/x case ---
    text = patch_once(text,
        "          IF ( (variable /= 'u') .and. (variable /= 'x') ) THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO i = 1, bdyzone\n"
        "              dat(ide+i-1,j) = dat(ide-i,j)  \n"
        "            ENDDO\n"
        "            ENDDO",

        "          IF ( (variable /= 'u') .and. (variable /= 'x') ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO i = 1, bdyzone\n"
        "              dat(ide+i-1,j) = dat(ide-i,j)  \n"
        "            ENDDO\n"
        "            ENDDO",
        "bc2d: symmetry_xe non-u")

    # --- bc2d symmetry_xe: u case and else ---
    text = patch_once(text,
        "            IF (variable == 'u' ) THEN\n"
        "\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO i = 0, bdyzone-1\n"
        "                dat(ide+i,j) = - dat(ide-i,j)  \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "\n"
        "            ELSE\n"
        "\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO i = 0, bdyzone-1\n"
        "                dat(ide+i,j) = dat(ide-i,j)  \n"
        "              ENDDO\n"
        "              ENDDO",

        "            IF (variable == 'u' ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO i = 0, bdyzone-1\n"
        "                dat(ide+i,j) = - dat(ide-i,j)  \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "\n"
        "            ELSE\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO i = 0, bdyzone-1\n"
        "                dat(ide+i,j) = dat(ide-i,j)  \n"
        "              ENDDO\n"
        "              ENDDO",
        "bc2d: symmetry_xe u/else")

    # --- bc2d open_xs ---
    text = patch_once(text,
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              dat(ids-1,j) = dat(ids,j) \n"
        "              dat(ids-2,j) = dat(ids,j)\n"
        "              dat(ids-3,j) = dat(ids,j)\n"
        "            ENDDO\n"
        "\n"
        "        ENDIF open_xs",

        "!$acc parallel loop\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              dat(ids-1,j) = dat(ids,j) \n"
        "              dat(ids-2,j) = dat(ids,j)\n"
        "              dat(ids-3,j) = dat(ids,j)\n"
        "            ENDDO\n"
        "\n"
        "        ENDIF open_xs",
        "bc2d: open_xs")

    # --- bc2d open_xe: non-u case ---
    text = patch_once(text,
        "          IF ( variable /= 'u' .and. variable /= 'x') THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              dat(ide  ,j) = dat(ide-1,j)\n"
        "              dat(ide+1,j) = dat(ide-1,j)\n"
        "              dat(ide+2,j) = dat(ide-1,j)\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              dat(ide+1,j) = dat(ide,j)\n"
        "              dat(ide+2,j) = dat(ide,j)\n"
        "              dat(ide+3,j) = dat(ide,j)\n"
        "            ENDDO",

        "          IF ( variable /= 'u' .and. variable /= 'x') THEN\n"
        "\n"
        "!$acc parallel loop\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              dat(ide  ,j) = dat(ide-1,j)\n"
        "              dat(ide+1,j) = dat(ide-1,j)\n"
        "              dat(ide+2,j) = dat(ide-1,j)\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "!$acc parallel loop\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              dat(ide+1,j) = dat(ide,j)\n"
        "              dat(ide+2,j) = dat(ide,j)\n"
        "              dat(ide+3,j) = dat(ide,j)\n"
        "            ENDDO",
        "bc2d: open_xe")

    # --- bc2d periodicity_y: south/north ---
    text = patch_once(text,
        "          IF( jts == jds ) then\n"
        "\n"
        "            DO j = 0, -(bdyzone-1), -1\n"
        "              \n"
        "              DO i = istart, iend\n"
        "                dat(i,jds+j-1) = dat(i,jde+j-1)\n"
        "              ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          END IF\n"
        "\n"
        "          IF( jte == jde ) then\n"
        "\n"
        "            DO j = -jstag, bdyzone\n"
        "              \n"
        "              DO i = istart, iend\n"
        "                dat(i,jde+j+jstag) = dat(i,jds+j+jstag)\n"
        "              ENDDO\n"
        "            ENDDO",

        "          IF( jts == jds ) then\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = 0, -(bdyzone-1), -1\n"
        "              DO i = istart, iend\n"
        "                dat(i,jds+j-1) = dat(i,jde+j-1)\n"
        "              ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          END IF\n"
        "\n"
        "          IF( jte == jde ) then\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = -jstag, bdyzone\n"
        "              DO i = istart, iend\n"
        "                dat(i,jde+j+jstag) = dat(i,jds+j+jstag)\n"
        "              ENDDO\n"
        "            ENDDO",
        "bc2d: periodic_y south/north")

    # --- bc2d symmetry_ys: non-v/y, then v, then else ---
    text = patch_once(text,
        "          IF ( (variable /= 'v') .and. (variable /= 'y') ) THEN\n"
        "\n"
        "            DO j = 1, bdyzone\n"
        "              \n"
        "              DO i = istart, iend\n"
        "                dat(i,jds-j) = dat(i,jds+j-1)\n"
        "              ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF (variable == 'v') THEN\n"
        "\n"
        "              DO j = 1, bdyzone\n"
        "                \n"
        "                DO i = istart, iend\n"
        "                  dat(i,jds-j) = - dat(i,jds+j)\n"
        "                ENDDO              \n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "              DO j = 1, bdyzone\n"
        "                \n"
        "                DO i = istart, iend\n"
        "                  dat(i,jds-j) = dat(i,jds+j)\n"
        "                ENDDO              \n"
        "              ENDDO",

        "          IF ( (variable /= 'v') .and. (variable /= 'y') ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = 1, bdyzone\n"
        "              DO i = istart, iend\n"
        "                dat(i,jds-j) = dat(i,jds+j-1)\n"
        "              ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF (variable == 'v') THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "              DO j = 1, bdyzone\n"
        "                DO i = istart, iend\n"
        "                  dat(i,jds-j) = - dat(i,jds+j)\n"
        "                ENDDO              \n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "              DO j = 1, bdyzone\n"
        "                DO i = istart, iend\n"
        "                  dat(i,jds-j) = dat(i,jds+j)\n"
        "                ENDDO              \n"
        "              ENDDO",
        "bc2d: symmetry_ys")

    # --- bc2d symmetry_ye: non-v/y, then v, then else ---
    text = patch_once(text,
        "          IF ( (variable /= 'v') .and. (variable /= 'y') ) THEN\n"
        "\n"
        "            DO j = 1, bdyzone\n"
        "              \n"
        "              DO i = istart, iend\n"
        "                dat(i,jde+j-1) = dat(i,jde-j)\n"
        "              ENDDO                               \n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF (variable == 'v' ) THEN\n"
        "\n"
        "              DO j = 1, bdyzone\n"
        "                \n"
        "                DO i = istart, iend\n"
        "                  dat(i,jde+j) = - dat(i,jde-j)    \n"
        "                ENDDO                               \n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "              DO j = 1, bdyzone\n"
        "                \n"
        "                DO i = istart, iend\n"
        "                  dat(i,jde+j) = dat(i,jde-j)\n"
        "                ENDDO                               \n"
        "              ENDDO",

        "          IF ( (variable /= 'v') .and. (variable /= 'y') ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = 1, bdyzone\n"
        "              DO i = istart, iend\n"
        "                dat(i,jde+j-1) = dat(i,jde-j)\n"
        "              ENDDO                               \n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF (variable == 'v' ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "              DO j = 1, bdyzone\n"
        "                DO i = istart, iend\n"
        "                  dat(i,jde+j) = - dat(i,jde-j)    \n"
        "                ENDDO                               \n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "              DO j = 1, bdyzone\n"
        "                DO i = istart, iend\n"
        "                  dat(i,jde+j) = dat(i,jde-j)\n"
        "                ENDDO                               \n"
        "              ENDDO",
        "bc2d: symmetry_ye")

    # --- bc2d open_ys ---
    text = patch_once(text,
        "            \n"
        "            DO i = istart, iend\n"
        "              dat(i,jds-1) = dat(i,jds)\n"
        "              dat(i,jds-2) = dat(i,jds)\n"
        "              dat(i,jds-3) = dat(i,jds)\n"
        "            ENDDO\n"
        "\n"
        "        ENDIF open_ys",

        "!$acc parallel loop\n"
        "            DO i = istart, iend\n"
        "              dat(i,jds-1) = dat(i,jds)\n"
        "              dat(i,jds-2) = dat(i,jds)\n"
        "              dat(i,jds-3) = dat(i,jds)\n"
        "            ENDDO\n"
        "\n"
        "        ENDIF open_ys",
        "bc2d: open_ys")

    # --- bc2d open_ye: non-v case and v case ---
    text = patch_once(text,
        "          IF  (variable /= 'v' .and. variable /= 'y' ) THEN\n"
        "\n"
        "            \n"
        "            DO i = istart, iend\n"
        "              dat(i,jde  ) = dat(i,jde-1)\n"
        "              dat(i,jde+1) = dat(i,jde-1)\n"
        "              dat(i,jde+2) = dat(i,jde-1)\n"
        "            ENDDO                               \n"
        "\n"
        "          ELSE\n"
        "\n"
        "            \n"
        "            DO i = istart, iend\n"
        "              dat(i,jde+1) = dat(i,jde)\n"
        "              dat(i,jde+2) = dat(i,jde)\n"
        "              dat(i,jde+3) = dat(i,jde)\n"
        "            ENDDO",

        "          IF  (variable /= 'v' .and. variable /= 'y' ) THEN\n"
        "\n"
        "!$acc parallel loop\n"
        "            DO i = istart, iend\n"
        "              dat(i,jde  ) = dat(i,jde-1)\n"
        "              dat(i,jde+1) = dat(i,jde-1)\n"
        "              dat(i,jde+2) = dat(i,jde-1)\n"
        "            ENDDO                               \n"
        "\n"
        "          ELSE\n"
        "\n"
        "!$acc parallel loop\n"
        "            DO i = istart, iend\n"
        "              dat(i,jde+1) = dat(i,jde)\n"
        "              dat(i,jde+2) = dat(i,jde)\n"
        "              dat(i,jde+3) = dat(i,jde)\n"
        "            ENDDO",
        "bc2d: open_ye")

    # --- bc2d corner fills (doubly periodic) ---
    # Lower left
    text = patch_once(text,
        "         IF ( (its == ids) .and. (jts == jds) ) THEN  \n"
        "           DO j = 0, -(bdyzone-1), -1\n"
        "           DO i = 0, -(bdyzone-1), -1\n"
        "             dat(ids+i-1,jds+j-1) = dat(ide+i-1,jde+j-1)",

        "         IF ( (its == ids) .and. (jts == jds) ) THEN  \n"
        "!$acc parallel loop collapse(2)\n"
        "           DO j = 0, -(bdyzone-1), -1\n"
        "           DO i = 0, -(bdyzone-1), -1\n"
        "             dat(ids+i-1,jds+j-1) = dat(ide+i-1,jde+j-1)",
        "bc2d: corner LL")

    # Lower right
    text = patch_once(text,
        "         IF ( (ite == ide) .and. (jts == jds) ) THEN  \n"
        "           DO j = 0, -(bdyzone-1), -1\n"
        "           DO i = 1, bdyzone\n"
        "             dat(ide+i+istag,jds+j-1) = dat(ids+i+istag,jde+j-1)",

        "         IF ( (ite == ide) .and. (jts == jds) ) THEN  \n"
        "!$acc parallel loop collapse(2)\n"
        "           DO j = 0, -(bdyzone-1), -1\n"
        "           DO i = 1, bdyzone\n"
        "             dat(ide+i+istag,jds+j-1) = dat(ids+i+istag,jde+j-1)",
        "bc2d: corner LR")

    # Upper right
    text = patch_once(text,
        "         IF ( (ite == ide) .and. (jte == jde) ) THEN  \n"
        "           DO j = 1, bdyzone\n"
        "           DO i = 1, bdyzone\n"
        "             dat(ide+i+istag,jde+j+jstag) = dat(ids+i+istag,jds+j+jstag)",

        "         IF ( (ite == ide) .and. (jte == jde) ) THEN  \n"
        "!$acc parallel loop collapse(2)\n"
        "           DO j = 1, bdyzone\n"
        "           DO i = 1, bdyzone\n"
        "             dat(ide+i+istag,jde+j+jstag) = dat(ids+i+istag,jds+j+jstag)",
        "bc2d: corner UR")

    # Upper left
    text = patch_once(text,
        "         IF ( (its == ids) .and. (jte == jde) ) THEN  \n"
        "           DO j = 1, bdyzone\n"
        "           DO i = 0, -(bdyzone-1), -1\n"
        "             dat(ids+i-1,jde+j+jstag) = dat(ide+i-1,jds+j+jstag)",

        "         IF ( (its == ids) .and. (jte == jde) ) THEN  \n"
        "!$acc parallel loop collapse(2)\n"
        "           DO j = 1, bdyzone\n"
        "           DO i = 0, -(bdyzone-1), -1\n"
        "             dat(ids+i-1,jde+j+jstag) = dat(ide+i-1,jds+j+jstag)",
        "bc2d: corner UL")

    # ==================================================================
    # PART 2: set_physical_bc3d — Add !$acc data present(dat)
    # ==================================================================
    print("\n=== Patching set_physical_bc3d ===")

    # Add !$acc data present(dat) after variable declarations
    text = patch_once(text,
        "      LOGICAL  :: debug, open_bc_copy\n"
        "\n"
        "\n"
        "\n"
        "      debug = .false.\n"
        "\n"
        "      open_bc_copy = .false.\n"
        "\n"
        "      variable = variable_in\n"
        "      IF ( variable_in .ge. 'A' .and. variable_in .le. 'Z' ) THEN\n"
        "        variable = CHAR( ICHAR(variable_in) - ICHAR('A') + ICHAR('a') )\n"
        "      ENDIF\n"
        "\n"
        "      IF ((variable == 'u') .or. (variable == 'v') .or.     &\n"
        "          (variable == 'w') .or. (variable == 't') .or.     &\n"
        "          (variable == 'd') .or. (variable == 'e') .or. &\n"
        "          (variable == 'x') .or. (variable == 'y') .or. &\n"
        "          (variable == 'f') .or. (variable == 'r') .or. &\n"
        "          (variable == 'p')                        ) open_bc_copy = .true.",

        "      LOGICAL  :: debug, open_bc_copy\n"
        "\n"
        "!$acc data present(dat)\n"
        "\n"
        "      debug = .false.\n"
        "\n"
        "      open_bc_copy = .false.\n"
        "\n"
        "      variable = variable_in\n"
        "      IF ( variable_in .ge. 'A' .and. variable_in .le. 'Z' ) THEN\n"
        "        variable = CHAR( ICHAR(variable_in) - ICHAR('A') + ICHAR('a') )\n"
        "      ENDIF\n"
        "\n"
        "      IF ((variable == 'u') .or. (variable == 'v') .or.     &\n"
        "          (variable == 'w') .or. (variable == 't') .or.     &\n"
        "          (variable == 'd') .or. (variable == 'e') .or. &\n"
        "          (variable == 'x') .or. (variable == 'y') .or. &\n"
        "          (variable == 'f') .or. (variable == 'r') .or. &\n"
        "          (variable == 'p')                        ) open_bc_copy = .true.",
        "bc3d: data present")

    # Add !$acc end data before END SUBROUTINE set_physical_bc3d
    text = patch_once(text,
        "   END SUBROUTINE set_physical_bc3d",
        "!$acc end data\n"
        "   END SUBROUTINE set_physical_bc3d",
        "bc3d: end data")

    # --- bc3d periodicity_x: west boundary (j,k,i loop) ---
    text = patch_once(text,
        "          IF ( its == ids ) THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO k = kts, k_end\n"
        "            DO i = 0,-(bdyzone-1),-1\n"
        "              dat(ids+i-1,k,j) = dat(ide+i-1,k,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ENDIF\n"
        "\n"
        "\n"
        "          IF ( ite == ide ) THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO k = kts, k_end\n"
        "            DO i = -istag , bdyzone\n"
        "              dat(ide+i+istag,k,j) = dat(ids+i+istag,k,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO",

        "          IF ( its == ids ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO k = kts, k_end\n"
        "            DO i = 0,-(bdyzone-1),-1\n"
        "              dat(ids+i-1,k,j) = dat(ide+i-1,k,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ENDIF\n"
        "\n"
        "\n"
        "          IF ( ite == ide ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO k = kts, k_end\n"
        "            DO i = -istag , bdyzone\n"
        "              dat(ide+i+istag,k,j) = dat(ids+i+istag,k,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO",
        "bc3d: periodic_x west/east")

    # --- bc3d symmetry_xs: istag==-1 case ---
    text = patch_once(text,
        "          IF ( istag == -1 ) THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO k = kts, k_end\n"
        "            DO i = 1, bdyzone\n"
        "              dat(ids-i,k,j) = dat(ids+i-1,k,j) \n"
        "            ENDDO                                 \n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF ( variable == 'u' ) THEN\n"
        "\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO k = kts, k_end\n"
        "              DO i = 1, bdyzone\n"
        "                dat(ids-i,k,j) = - dat(ids+i,k,j) \n"
        "              ENDDO                                 \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO k = kts, k_end\n"
        "              DO i = 1, bdyzone\n"
        "                dat(ids-i,k,j) = dat(ids+i,k,j) \n"
        "              ENDDO                               \n"
        "              ENDDO\n"
        "              ENDDO",

        "          IF ( istag == -1 ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO k = kts, k_end\n"
        "            DO i = 1, bdyzone\n"
        "              dat(ids-i,k,j) = dat(ids+i-1,k,j) \n"
        "            ENDDO                                 \n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF ( variable == 'u' ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO k = kts, k_end\n"
        "              DO i = 1, bdyzone\n"
        "                dat(ids-i,k,j) = - dat(ids+i,k,j) \n"
        "              ENDDO                                 \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO k = kts, k_end\n"
        "              DO i = 1, bdyzone\n"
        "                dat(ids-i,k,j) = dat(ids+i,k,j) \n"
        "              ENDDO                               \n"
        "              ENDDO\n"
        "              ENDDO",
        "bc3d: symmetry_xs")

    # --- bc3d symmetry_xe: istag==-1 case ---
    text = patch_once(text,
        "          IF ( istag == -1 ) THEN\n"
        "\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO k = kts, k_end\n"
        "            DO i = 1, bdyzone\n"
        "              dat(ide+i-1,k,j) = dat(ide-i,k,j)  \n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF (variable == 'u') THEN\n"
        "\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO k = kts, k_end\n"
        "              DO i = 1, bdyzone\n"
        "                dat(ide+i,k,j) = - dat(ide-i,k,j)  \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO k = kts, k_end\n"
        "              DO i = 1, bdyzone\n"
        "                dat(ide+i,k,j) = dat(ide-i,k,j)  \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "              ENDDO",

        "          IF ( istag == -1 ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "            DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "            DO k = kts, k_end\n"
        "            DO i = 1, bdyzone\n"
        "              dat(ide+i-1,k,j) = dat(ide-i,k,j)  \n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF (variable == 'u') THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO k = kts, k_end\n"
        "              DO i = 1, bdyzone\n"
        "                dat(ide+i,k,j) = - dat(ide-i,k,j)  \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "              DO j = MAX(jds,jts-1), MIN(jte+1,jde+jstag)\n"
        "              DO k = kts, k_end\n"
        "              DO i = 1, bdyzone\n"
        "                dat(ide+i,k,j) = dat(ide-i,k,j)  \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "              ENDDO",
        "bc3d: symmetry_xe")

    # --- bc3d open_xs ---
    text = patch_once(text,
        "            DO j = jts-bdyzone, MIN(jte,jde+jstag)+bdyzone\n"
        "            DO k = kts, k_end\n"
        "              dat(ids-1,k,j) = dat(ids,k,j) \n"
        "              dat(ids-2,k,j) = dat(ids,k,j)\n"
        "              dat(ids-3,k,j) = dat(ids,k,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "        ENDIF open_xs",

        "!$acc parallel loop collapse(2)\n"
        "            DO j = jts-bdyzone, MIN(jte,jde+jstag)+bdyzone\n"
        "            DO k = kts, k_end\n"
        "              dat(ids-1,k,j) = dat(ids,k,j) \n"
        "              dat(ids-2,k,j) = dat(ids,k,j)\n"
        "              dat(ids-3,k,j) = dat(ids,k,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "        ENDIF open_xs",
        "bc3d: open_xs")

    # --- bc3d open_xe: non-u case ---
    text = patch_once(text,
        "          IF (variable /= 'u' .and. variable /= 'x' ) THEN\n"
        "\n"
        "            DO j = jts-bdyzone, MIN(jte,jde+jstag)+bdyzone\n"
        "            DO k = kts, k_end\n"
        "              dat(ide  ,k,j) = dat(ide-1,k,j)\n"
        "              dat(ide+1,k,j) = dat(ide-1,k,j)\n"
        "              dat(ide+2,k,j) = dat(ide-1,k,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "\n"
        "            DO j = MAX(jds,jts-1)-bdyzone, MIN(jte+1,jde+jstag)+bdyzone\n"
        "            DO k = kts, k_end\n"
        "              dat(ide+1,k,j) = dat(ide,k,j)\n"
        "              dat(ide+2,k,j) = dat(ide,k,j)\n"
        "              dat(ide+3,k,j) = dat(ide,k,j)\n"
        "            ENDDO\n"
        "            ENDDO",

        "          IF (variable /= 'u' .and. variable /= 'x' ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = jts-bdyzone, MIN(jte,jde+jstag)+bdyzone\n"
        "            DO k = kts, k_end\n"
        "              dat(ide  ,k,j) = dat(ide-1,k,j)\n"
        "              dat(ide+1,k,j) = dat(ide-1,k,j)\n"
        "              dat(ide+2,k,j) = dat(ide-1,k,j)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO j = MAX(jds,jts-1)-bdyzone, MIN(jte+1,jde+jstag)+bdyzone\n"
        "            DO k = kts, k_end\n"
        "              dat(ide+1,k,j) = dat(ide,k,j)\n"
        "              dat(ide+2,k,j) = dat(ide,k,j)\n"
        "              dat(ide+3,k,j) = dat(ide,k,j)\n"
        "            ENDDO\n"
        "            ENDDO",
        "bc3d: open_xe")

    # --- bc3d periodicity_y: south/north ---
    text = patch_once(text,
        "          IF( jts == jds ) then\n"
        "\n"
        "            DO j = 0, -(bdyzone-1), -1\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jds+j-1) = dat(i,k,jde+j-1)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          END IF\n"
        "\n"
        "          IF( jte == jde ) then\n"
        "\n"
        "            DO j = -jstag, bdyzone\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jde+j+jstag) = dat(i,k,jds+j+jstag)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO",

        "          IF( jts == jds ) then\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "            DO j = 0, -(bdyzone-1), -1\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jds+j-1) = dat(i,k,jde+j-1)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          END IF\n"
        "\n"
        "          IF( jte == jde ) then\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "            DO j = -jstag, bdyzone\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jde+j+jstag) = dat(i,k,jds+j+jstag)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "            ENDDO",
        "bc3d: periodic_y south/north")

    # --- bc3d symmetry_ys ---
    text = patch_once(text,
        "          IF ( jstag == -1 ) THEN\n"
        "\n"
        "            DO j = 1, bdyzone\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jds-j) = dat(i,k,jds+j-1)\n"
        "            ENDDO                               \n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF (variable == 'v') THEN\n"
        "\n"
        "              DO j = 1, bdyzone\n"
        "              DO k = kts, k_end\n"
        "              DO i = i_start, i_end\n"
        "                dat(i,k,jds-j) = - dat(i,k,jds+j)\n"
        "              ENDDO              \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "              DO j = 1, bdyzone\n"
        "              DO k = kts, k_end\n"
        "              DO i = i_start, i_end\n"
        "                dat(i,k,jds-j) = dat(i,k,jds+j)\n"
        "              ENDDO              \n"
        "              ENDDO\n"
        "              ENDDO",

        "          IF ( jstag == -1 ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "            DO j = 1, bdyzone\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jds-j) = dat(i,k,jds+j-1)\n"
        "            ENDDO                               \n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF (variable == 'v') THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "              DO j = 1, bdyzone\n"
        "              DO k = kts, k_end\n"
        "              DO i = i_start, i_end\n"
        "                dat(i,k,jds-j) = - dat(i,k,jds+j)\n"
        "              ENDDO              \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "              DO j = 1, bdyzone\n"
        "              DO k = kts, k_end\n"
        "              DO i = i_start, i_end\n"
        "                dat(i,k,jds-j) = dat(i,k,jds+j)\n"
        "              ENDDO              \n"
        "              ENDDO\n"
        "              ENDDO",
        "bc3d: symmetry_ys")

    # --- bc3d symmetry_ye ---
    text = patch_once(text,
        "          IF ( jstag == -1 ) THEN\n"
        "\n"
        "            DO j = 1, bdyzone\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jde+j-1) = dat(i,k,jde-j)\n"
        "            ENDDO                               \n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF ( variable == 'v' ) THEN\n"
        "\n"
        "              DO j = 1, bdyzone\n"
        "              DO k = kts, k_end\n"
        "              DO i = i_start, i_end\n"
        "                dat(i,k,jde+j) = - dat(i,k,jde-j)\n"
        "              ENDDO                               \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "              DO j = 1, bdyzone\n"
        "              DO k = kts, k_end\n"
        "              DO i = i_start, i_end\n"
        "                dat(i,k,jde+j) = dat(i,k,jde-j)\n"
        "              ENDDO                               \n"
        "              ENDDO\n"
        "              ENDDO",

        "          IF ( jstag == -1 ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "            DO j = 1, bdyzone\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jde+j-1) = dat(i,k,jde-j)\n"
        "            ENDDO                               \n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            IF ( variable == 'v' ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "              DO j = 1, bdyzone\n"
        "              DO k = kts, k_end\n"
        "              DO i = i_start, i_end\n"
        "                dat(i,k,jde+j) = - dat(i,k,jde-j)\n"
        "              ENDDO                               \n"
        "              ENDDO\n"
        "              ENDDO\n"
        "\n"
        "            ELSE\n"
        "\n"
        "!$acc parallel loop collapse(3)\n"
        "              DO j = 1, bdyzone\n"
        "              DO k = kts, k_end\n"
        "              DO i = i_start, i_end\n"
        "                dat(i,k,jde+j) = dat(i,k,jde-j)\n"
        "              ENDDO                               \n"
        "              ENDDO\n"
        "              ENDDO",
        "bc3d: symmetry_ye")

    # --- bc3d open_ys ---
    text = patch_once(text,
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jds-1) = dat(i,k,jds)\n"
        "              dat(i,k,jds-2) = dat(i,k,jds)\n"
        "              dat(i,k,jds-3) = dat(i,k,jds)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "        ENDIF open_ys",

        "!$acc parallel loop collapse(2)\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jds-1) = dat(i,k,jds)\n"
        "              dat(i,k,jds-2) = dat(i,k,jds)\n"
        "              dat(i,k,jds-3) = dat(i,k,jds)\n"
        "            ENDDO\n"
        "            ENDDO\n"
        "\n"
        "        ENDIF open_ys",
        "bc3d: open_ys")

    # --- bc3d open_ye ---
    text = patch_once(text,
        "          IF (variable /= 'v' .and. variable /= 'y' ) THEN\n"
        "\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jde  ) = dat(i,k,jde-1)\n"
        "              dat(i,k,jde+1) = dat(i,k,jde-1)\n"
        "              dat(i,k,jde+2) = dat(i,k,jde-1)\n"
        "            ENDDO                               \n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jde+1) = dat(i,k,jde)\n"
        "              dat(i,k,jde+2) = dat(i,k,jde)\n"
        "              dat(i,k,jde+3) = dat(i,k,jde)\n"
        "            ENDDO                               \n"
        "            ENDDO",

        "          IF (variable /= 'v' .and. variable /= 'y' ) THEN\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jde  ) = dat(i,k,jde-1)\n"
        "              dat(i,k,jde+1) = dat(i,k,jde-1)\n"
        "              dat(i,k,jde+2) = dat(i,k,jde-1)\n"
        "            ENDDO                               \n"
        "            ENDDO\n"
        "\n"
        "          ELSE\n"
        "\n"
        "!$acc parallel loop collapse(2)\n"
        "            DO k = kts, k_end\n"
        "            DO i = i_start, i_end\n"
        "              dat(i,k,jde+1) = dat(i,k,jde)\n"
        "              dat(i,k,jde+2) = dat(i,k,jde)\n"
        "              dat(i,k,jde+3) = dat(i,k,jde)\n"
        "            ENDDO                               \n"
        "            ENDDO",
        "bc3d: open_ye")

    return text


def main():
    print(f"Reading {F90_PATH}")
    with open(F90_PATH, "r") as f:
        text = f.read()

    # Guard: check if already patched
    if "!$acc data present(dat)" in text:
        print("Already patched (found '!$acc data present(dat)'). Skipping.")
        return

    original = text
    text = apply_patches(text)

    if text == original:
        print("\nNo changes made — already patched or patterns not found.")
        return

    with open(F90_PATH, "w") as f:
        f.write(text)

    # Count directives added
    n_data = text.count("!$acc data present(dat)")
    n_end_data = text.count("!$acc end data")
    n_parallel = text.count("!$acc parallel loop")
    print(f"\nDone. Added {n_data} data regions, {n_end_data} end data, "
          f"{n_parallel} parallel loop directives.")
    print(f"Written to {F90_PATH}")


if __name__ == "__main__":
    main()
