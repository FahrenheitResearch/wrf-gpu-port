#!/bin/bash
# Apply all safe GPU patches to freshly-built .f90 files.
# Usage: WRF_DIR=/path/to/WRF ./apply_all_patches.sh
#
# DOES NOT apply patch_wsm6_gpu.py (modified code logic → NaN)
# DOES NOT apply patch_ysu_gpu.py (caused GPU hang)
# Disables advect ACC by default (causes instability at ~5 min)
# Disables module_bc ACC by default (present() errors)
set -e

WRF="${1:-${WRF_DIR:-/home/$USER/WRF_BUILD_GPU}}"
export WRF_DIR="$WRF"
PATCHES="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "WRF_DIR: $WRF"
echo "PATCHES: $PATCHES"
echo ""

echo "============================================"
echo "PRE-STEP 0: Strip WRF native ACC directives"
echo "============================================"
echo "WRF 4.7.1 has built-in !\\$acc directives (via #ifdef GPU_OPENACC) that"
echo "reference local variables without proper data regions. Strip them so"
echo "our patches can add properly-scoped ACC with !\\$acc data create(...)."
echo ""

# Strip native ACC from sfclayrev (physics — present() errors on subroutine args)
SFCLAY_COUNT=0
for f in $(find "$WRF/phys" -name "*sfclayrev*" -type f 2>/dev/null); do
    n=$(grep -c '!\$acc' "$f" 2>/dev/null || true)
    if [ "$n" -gt 0 ]; then
        sed -i 's/^\([[:space:]]*\)!\$acc/\1!DISABLED_acc/' "$f"
        echo "  Stripped $n native ACC directives from $(basename $f)"
        SFCLAY_COUNT=$((SFCLAY_COUNT + n))
    fi
done

# Strip native ACC from solve_em.f90 (locals like ph_save referenced without data create)
SOLVE_COUNT=$(grep -c '!\$acc' "$WRF/dyn_em/solve_em.f90" 2>/dev/null || true)
if [ "$SOLVE_COUNT" -gt 0 ]; then
    sed -i 's/^\([[:space:]]*\)!\$acc/\1!DISABLED_acc/' "$WRF/dyn_em/solve_em.f90"
    echo "  Stripped $SOLVE_COUNT native ACC directives from solve_em.f90"
fi

echo "  Total stripped: $((SFCLAY_COUNT + SOLVE_COUNT)) native ACC directives"
echo "  Our patches will re-add properly-scoped ACC directives."
echo ""

echo "============================================"
echo "PRE-CHECK: Detect already-patched .f90 files"
echo "============================================"
# Check for our custom patches (not the native ones we just stripped)
ACC_COUNT=$(grep -rl '!\$acc' "$WRF"/dyn_em/module_small_step_em.f90 "$WRF"/dyn_em/module_big_step_utilities_em.f90 "$WRF"/dyn_em/module_diffusion_em.f90 "$WRF"/frame/module_domain.f90 2>/dev/null | wc -l || true)
if [ "$ACC_COUNT" -gt 0 ]; then
    echo "WARNING: $ACC_COUNT .f90 file(s) already contain !\$acc directives."
    echo "  Patches are designed to run on clean (unpatched) .f90 files."
    echo "  If you see errors, regenerate .f90 files with './compile em_real' first."
    echo ""
fi

echo "============================================"
echo "STEP 1: Verify .f90 files exist"
echo "============================================"
for f in $WRF/dyn_em/module_small_step_em.f90 \
         $WRF/dyn_em/module_advect_em.f90 \
         $WRF/dyn_em/module_big_step_utilities_em.f90 \
         $WRF/dyn_em/module_diffusion_em.f90 \
         $WRF/dyn_em/solve_em.f90 \
         $WRF/dyn_em/module_em.f90 \
         $WRF/dyn_em/start_em.f90 \
         $WRF/frame/module_domain.f90 \
         $WRF/share/module_bc.f90; do
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"
        echo "Run './compile em_real' first to generate .f90 files from .F sources."
        exit 1
    fi
done
echo "All .f90 files present."

echo ""
echo "============================================"
echo "STEP 2: Create + populate GPU init subroutines"
echo "============================================"
echo "Creating gpu_init_domain_data stub..."
$PYTHON $PATCHES/create_gpu_init.py "$WRF"
echo "Populating gpu_init with grid% copyin directives..."
$PYTHON $PATCHES/build_gpu_init_from_struct.py
$PYTHON $PATCHES/patch_gpu_init_les.py
echo "gpu_init patched"

echo ""
echo "============================================"
echo "STEP 3: Apply dynamics patches"
echo "============================================"

echo "Patching small_step_em..."
$PYTHON $PATCHES/patch_small_step_gpu.py
echo "  done"

echo "Patching advect_em..."
$PYTHON $PATCHES/patch_advect_gpu.py
$PYTHON $PATCHES/patch_advect_locals.py
$PYTHON $PATCHES/fix_advect_w_create.py
echo "  done"

echo "Patching big_step_utilities..."
$PYTHON $PATCHES/patch_big_step_gpu.py
$PYTHON $PATCHES/patch_remaining_bigstep_gpu.py
echo "  done"

echo "Patching diffusion (comprehensive)..."
$PYTHON $PATCHES/patch_diffusion_comprehensive.py
echo "  done"

echo ""
echo "============================================"
echo "STEP 3b: Infrastructure patches"
echo "============================================"

echo "Patching module_bc..."
$PYTHON $PATCHES/patch_bc_gpu.py
echo "  done"

echo "Patching module_em..."
$PYTHON $PATCHES/patch_module_em_gpu.py
echo "  done"

echo "Patching first_rk + physics_addtendc..."
$PYTHON $PATCHES/patch_first_rk_gpu.py || echo "  (skipped - file may not exist)"
$PYTHON $PATCHES/fix_addtendc.py || echo "  (skipped - file may not exist)"
echo "  done"

echo "Patching solve_em data regions..."
$PYTHON $PATCHES/patch_solve_em_gpu.py
echo "  done"

echo "Patching start_em (CALL gpu_init_domain_data)..."
$PYTHON $PATCHES/patch_start_em_gpu.py
echo "  done"

echo ""
echo "============================================"
echo "STEP 4: Apply physics patches (SAFE ONLY)"
echo "============================================"

echo "SKIPPING patch_sfclay_gpu.py (present() errors — physics callers don't set up data regions)"
echo "SKIPPING patch_wsm6_gpu.py (caused NaN — modified code logic)"
echo "SKIPPING patch_ysu_gpu.py (caused GPU hang)"

echo ""
echo "============================================"
echo "STEP 4b: Kernel fusion (must be after dynamics patches)"
echo "============================================"
$PYTHON $PATCHES/fuse_kernels.py || echo "  (skipped - kernel fusion optional)"
$PYTHON $PATCHES/patch_fuse_advect_kernels.py || echo "  (skipped - advect fusion optional)"
echo "  done"

echo ""
echo "============================================"
echo "STEP 5: Apply timing fix for GPU init"
echo "============================================"
$PYTHON $PATCHES/fix_gpu_init_timing.py || echo "  (skipped - may target .F file)"
$PYTHON $PATCHES/fix_acoustic_sync_v2.py || echo "  (skipped - may target .F file)"
echo "  done"

echo ""
echo "============================================"
echo "STEP 6: Disable known-broken ACC modules"
echo "============================================"

echo "Disabling advect ACC (causes instability at ~5 min)..."
$PYTHON $PATCHES/disable_one_acc.py module_advect_em.f90
echo "  done"

echo "Disabling module_bc ACC (present() errors)..."
$PYTHON $PATCHES/disable_bc_acc.py
echo "  done"

echo ""
echo "============================================"
echo "ALL PATCHES APPLIED"
echo "============================================"
echo ""
echo "Now run compile_patched.sh to rebuild patched modules and relink wrf.exe"
