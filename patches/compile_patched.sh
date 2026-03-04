#!/bin/bash
# compile_patched.sh — Recompile all GPU-patched WRF .f90 files with nvfortran + OpenACC
#
# Usage:
#   ./compile_patched.sh [WRF_DIR] [GPU_CC]
#
# Arguments:
#   WRF_DIR   Path to WRF build root (default: /home/$USER/WRF_BUILD_GPU)
#   GPU_CC    CUDA compute capability string (default: cc120 for RTX 5090 Blackwell)
#
# Notes:
#   - module_advect_em.f90 and module_bc.f90 ACC directives are disabled (!!$acc)
#     They still compile with -acc but no GPU kernels are generated for them.
#   - Physics: sfclay compiles WITH -acc; YSU WITHOUT -acc (GPU version hangs).
#   - WSM6 is NOT recompiled here — use the stock-built object from the archive.
#   - Run from any directory; all paths are derived from WRF_DIR.

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
WRF_DIR="${1:-/home/${USER}/WRF_BUILD_GPU}"
GPU_CC="${2:-cc120}"

# ---------------------------------------------------------------------------
# Validate WRF_DIR
# ---------------------------------------------------------------------------
if [ ! -d "${WRF_DIR}" ]; then
    echo "ERROR: WRF_DIR does not exist: ${WRF_DIR}"
    exit 1
fi

echo "========================================================================"
echo "compile_patched.sh"
echo "  WRF_DIR : ${WRF_DIR}"
echo "  GPU_CC  : ${GPU_CC}"
echo "  Started : $(date)"
echo "========================================================================"

# ---------------------------------------------------------------------------
# NVHPC 26.1 environment
# ---------------------------------------------------------------------------
NVHPC_ROOT="/opt/nvidia/hpc_sdk/Linux_x86_64/26.1"
export PATH="${NVHPC_ROOT}/compilers/bin:${NVHPC_ROOT}/comm_libs/mpi/bin:${PATH}"
export LD_LIBRARY_PATH="${NVHPC_ROOT}/compilers/lib:${NVHPC_ROOT}/comm_libs/mpi/lib:${NVHPC_ROOT}/cuda/13.1/lib64:/home/${USER}/libs/lib:${NETCDF}/lib:${HDF5}/lib:${LD_LIBRARY_PATH:-}"

# Confirm nvfortran is reachable
if ! command -v nvfortran &>/dev/null; then
    echo "ERROR: nvfortran not found in PATH after setting NVHPC_ROOT."
    echo "       Check NVHPC_ROOT=${NVHPC_ROOT}"
    exit 1
fi
echo "nvfortran: $(which nvfortran)  version: $(nvfortran --version 2>&1 | head -1)"
echo ""

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
# Primary flags — used for all GPU-enabled files.
# !!$acc markers mean ACC directives are commented out, but -acc is still passed
# (the compiler generates no GPU kernels where all directives are disabled).
FFLAGS="-O3 -fast -acc -gpu=${GPU_CC},fastmath -Mfree -Mrecursive -byteswapio -r4 -i4 -mp"

# CPU-only flags — for files where GPU kernels caused hangs (YSU).
FFLAGS_CPU="-O3 -fast -Mfree -Mrecursive -byteswapio -r4 -i4 -mp"

# Include paths — mirrors the paths used during the original WRF configure build.
# -DGPU_OPENACC guards conditional GPU code in WRF headers.
INCS=(
    "-I${WRF_DIR}/frame"
    "-I${WRF_DIR}/inc"
    "-I${WRF_DIR}/share"
    "-I${WRF_DIR}/phys"
    "-I${WRF_DIR}/main"
    "-I${WRF_DIR}/chem"
    "-I${WRF_DIR}/external/io_netcdf"
    "-I${WRF_DIR}/external/io_int"
    "-I${WRF_DIR}/external/esmf_time_f90"
    "-I${WRF_DIR}/dyn_em"
    "-I${WRF_DIR}/phys/physics_mmm"
    "-DGPU_OPENACC"
)
INCS_STR="${INCS[*]}"

# ---------------------------------------------------------------------------
# Helper: compile one file, with timing and error capture
# ---------------------------------------------------------------------------
compile_file() {
    local label="$1"       # human-readable description
    local dir="$2"         # directory containing the source file
    local src="$3"         # source filename (relative to dir)
    local out="$4"         # object filename (relative to dir)
    local flags="$5"       # compiler flags string
    local extra_incs="${6:-}"  # optional extra -I flags

    echo "--------------------------------------------------------------------"
    echo "Compiling: ${label}"
    echo "  Source : ${dir}/${src}"
    local t0
    t0=$(date +%s%N)

    local log_tmp
    log_tmp=$(mktemp /tmp/nvf_compile_XXXXXX.log)

    if nvfortran -o "${dir}/${out}" -c ${flags} ${INCS_STR} ${extra_incs} "${dir}/${src}" \
            >"${log_tmp}" 2>&1; then
        local t1
        t1=$(date +%s%N)
        local elapsed_ms=$(( (t1 - t0) / 1000000 ))
        echo "  Status : OK  (${elapsed_ms} ms)"
    else
        local exit_code=$?
        local t1
        t1=$(date +%s%N)
        local elapsed_ms=$(( (t1 - t0) / 1000000 ))
        echo "  Status : FAILED (exit ${exit_code}, ${elapsed_ms} ms)"
        echo "  --- compiler output (errors/severe only) ---"
        grep -iE "severe|error" "${log_tmp}" | head -20 || true
        echo "  --- full log: ${log_tmp} ---"
        # Print full log so CI/tty captures everything
        cat "${log_tmp}"
        rm -f "${log_tmp}"
        exit "${exit_code}"
    fi
    rm -f "${log_tmp}"
}

# ---------------------------------------------------------------------------
# Verify source files exist before starting
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "STEP 1: Verify patched .f90 source files"
echo "========================================================================"

REQUIRED_SOURCES=(
    "${WRF_DIR}/frame/module_domain.f90"
    "${WRF_DIR}/dyn_em/module_small_step_em.f90"
    "${WRF_DIR}/dyn_em/module_advect_em.f90"
    "${WRF_DIR}/dyn_em/module_big_step_utilities_em.f90"
    "${WRF_DIR}/dyn_em/module_diffusion_em.f90"
    "${WRF_DIR}/dyn_em/module_em.f90"
    "${WRF_DIR}/dyn_em/solve_em.f90"
    "${WRF_DIR}/share/module_bc.f90"
    "${WRF_DIR}/phys/module_sf_sfclayrev.F"
    "${WRF_DIR}/phys/physics_mmm/sf_sfclayrev.F90"
    "${WRF_DIR}/phys/module_bl_ysu.F"
)

all_present=1
for f in "${REQUIRED_SOURCES[@]}"; do
    if [ -f "${f}" ]; then
        echo "  OK  ${f}"
    else
        echo "  MISSING  ${f}"
        all_present=0
    fi
done

if [ "${all_present}" -eq 0 ]; then
    echo ""
    echo "ERROR: One or more source files are missing. Run apply_all_patches.sh first."
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Compile each patched file
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "STEP 2: Compile patched files"
echo "========================================================================"
echo ""

# frame/module_domain.f90
# GPU init: copyin of ~1404 grid% arrays into GPU memory.  ACC ENABLED.
compile_file \
    "frame/module_domain.f90  [ACC: enabled — gpu_init_domain_data]" \
    "${WRF_DIR}/frame" \
    "module_domain.f90" \
    "module_domain.o" \
    "${FFLAGS}"

# dyn_em/module_small_step_em.f90
# Acoustic timestep (pressure gradient, divergence).  ACC ENABLED.
compile_file \
    "dyn_em/module_small_step_em.f90  [ACC: enabled]" \
    "${WRF_DIR}/dyn_em" \
    "module_small_step_em.f90" \
    "module_small_step_em.o" \
    "${FFLAGS}"

# dyn_em/module_advect_em.f90
# Scalar/momentum advection.  ACC DIRECTIVES DISABLED (!!$acc) — dynamics
# reverted after mismatching kernel issues; compiles with -acc but no GPU kernels.
compile_file \
    'dyn_em/module_advect_em.f90  [ACC: directives disabled via !!$acc]' \
    "${WRF_DIR}/dyn_em" \
    "module_advect_em.f90" \
    "module_advect_em.o" \
    "${FFLAGS}"

# dyn_em/module_big_step_utilities_em.f90
# Large timestep utilities (mu, theta, etc.).  ACC ENABLED.
compile_file \
    "dyn_em/module_big_step_utilities_em.f90  [ACC: enabled]" \
    "${WRF_DIR}/dyn_em" \
    "module_big_step_utilities_em.f90" \
    "module_big_step_utilities_em.o" \
    "${FFLAGS}"

# dyn_em/module_diffusion_em.f90
# Horizontal/vertical diffusion, deformation.  ACC ENABLED (comprehensive patch).
compile_file \
    "dyn_em/module_diffusion_em.f90  [ACC: enabled]" \
    "${WRF_DIR}/dyn_em" \
    "module_diffusion_em.f90" \
    "module_diffusion_em.o" \
    "${FFLAGS}"

# dyn_em/module_em.f90
# Infrastructure: first_rk_step, physics_addtendc, etc.  ACC ENABLED.
compile_file \
    "dyn_em/module_em.f90  [ACC: enabled]" \
    "${WRF_DIR}/dyn_em" \
    "module_em.f90" \
    "module_em.o" \
    "${FFLAGS}"

# dyn_em/solve_em.f90
# Outer RK loop, data regions, sync points.  ACC ENABLED.
compile_file \
    "dyn_em/solve_em.f90  [ACC: enabled — data regions + sync]" \
    "${WRF_DIR}/dyn_em" \
    "solve_em.f90" \
    "solve_em.o" \
    "${FFLAGS}"

# share/module_bc.f90
# Boundary conditions.  ACC DIRECTIVES DISABLED (!!$acc) — CPU only by design.
compile_file \
    'share/module_bc.f90  [ACC: directives disabled via !!$acc]' \
    "${WRF_DIR}/share" \
    "module_bc.f90" \
    "module_bc.o" \
    "${FFLAGS}"

# phys/module_sf_sfclayrev.F  (preprocessed Fortran — .F extension)
# Revised surface-layer scheme driver.  ACC ENABLED.
compile_file \
    "phys/module_sf_sfclayrev.F  [ACC: enabled]" \
    "${WRF_DIR}/phys" \
    "module_sf_sfclayrev.F" \
    "module_sf_sfclayrev.o" \
    "${FFLAGS}"

# phys/physics_mmm/sf_sfclayrev.F90
# Core sfclay routines (physics_mmm sub-library).  ACC ENABLED.
# Needs an extra -I for the parent phys/ dir (used by physics_mmm includes).
compile_file \
    "phys/physics_mmm/sf_sfclayrev.F90  [ACC: enabled]" \
    "${WRF_DIR}/phys/physics_mmm" \
    "sf_sfclayrev.F90" \
    "sf_sfclayrev.o" \
    "${FFLAGS}" \
    "-I${WRF_DIR}/phys"

# phys/module_bl_ysu.F  — CPU ONLY
# YSU boundary-layer scheme.  ACC removed — GPU version caused hangs.
compile_file \
    "phys/module_bl_ysu.F  [ACC: disabled — CPU only, GPU version hangs]" \
    "${WRF_DIR}/phys" \
    "module_bl_ysu.F" \
    "module_bl_ysu.o" \
    "${FFLAGS_CPU}"

echo ""
echo "All files compiled successfully."

# ---------------------------------------------------------------------------
# Update archive
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "STEP 3: Update libwrflib.a"
echo "========================================================================"
t0=$(date +%s%N)

ar r "${WRF_DIR}/main/libwrflib.a" \
    "${WRF_DIR}/frame/module_domain.o" \
    "${WRF_DIR}/dyn_em/module_small_step_em.o" \
    "${WRF_DIR}/dyn_em/module_advect_em.o" \
    "${WRF_DIR}/dyn_em/module_big_step_utilities_em.o" \
    "${WRF_DIR}/dyn_em/module_diffusion_em.o" \
    "${WRF_DIR}/dyn_em/module_em.o" \
    "${WRF_DIR}/dyn_em/solve_em.o" \
    "${WRF_DIR}/share/module_bc.o" \
    "${WRF_DIR}/phys/module_sf_sfclayrev.o" \
    "${WRF_DIR}/phys/module_bl_ysu.o" \
    "${WRF_DIR}/phys/physics_mmm/sf_sfclayrev.o"

ranlib "${WRF_DIR}/main/libwrflib.a"

t1=$(date +%s%N)
elapsed_ms=$(( (t1 - t0) / 1000000 ))
echo "  Archive updated: ${WRF_DIR}/main/libwrflib.a  (${elapsed_ms} ms)"

# ---------------------------------------------------------------------------
# Relink wrf.exe
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "STEP 4: Link wrf.exe"
echo "========================================================================"
t0=$(date +%s%N)

# Confirm required link-time objects/libraries exist before calling nvfortran
REQUIRED_LINK=(
    "${WRF_DIR}/main/wrf.o"
    "${WRF_DIR}/main/module_wrf_top.o"
    "${WRF_DIR}/main/libwrflib.a"
    "${WRF_DIR}/external/fftpack/fftpack5/libfftpack.a"
    "${WRF_DIR}/external/io_grib1/libio_grib1.a"
    "${WRF_DIR}/external/io_grib_share/libio_grib_share.a"
    "${WRF_DIR}/external/io_int/libwrfio_int.a"
    "${WRF_DIR}/external/esmf_time_f90/libesmf_time.a"
    "${WRF_DIR}/external/RSL_LITE/librsl_lite.a"
    "${WRF_DIR}/frame/module_internal_header_util.o"
    "${WRF_DIR}/frame/pack_utils.o"
    "${WRF_DIR}/external/io_netcdf/libwrfio_nf.a"
)
for f in "${REQUIRED_LINK[@]}"; do
    if [ ! -f "${f}" ]; then
        echo "  WARNING: link-time object not found: ${f}"
    fi
done

mpif90 \
    -o "${WRF_DIR}/main/wrf.exe" \
    -mp -O3 -fast \
    -acc -gpu="${GPU_CC},fastmath" -cuda \
    -Mfree -Mrecursive -byteswapio -r4 -i4 \
    "${WRF_DIR}/main/wrf.o" \
    "${WRF_DIR}/main/module_wrf_top.o" \
    "${WRF_DIR}/main/libwrflib.a" \
    "${WRF_DIR}/external/fftpack/fftpack5/libfftpack.a" \
    "${WRF_DIR}/external/io_grib1/libio_grib1.a" \
    "${WRF_DIR}/external/io_grib_share/libio_grib_share.a" \
    "${WRF_DIR}/external/io_int/libwrfio_int.a" \
    -L"${WRF_DIR}/external/esmf_time_f90" -lesmf_time \
    "${WRF_DIR}/external/RSL_LITE/librsl_lite.a" \
    "${WRF_DIR}/frame/module_internal_header_util.o" \
    "${WRF_DIR}/frame/pack_utils.o" \
    -L"${WRF_DIR}/external/io_netcdf" -lwrfio_nf \
    -L"${NETCDF}/lib" -lnetcdff -lnetcdf \
    -L"${HDF5}/lib" \
        -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 \
    -lm -lz -ldl

t1=$(date +%s%N)
elapsed_ms=$(( (t1 - t0) / 1000000 ))
echo "  Link time: ${elapsed_ms} ms"

echo ""
ls -lh "${WRF_DIR}/main/wrf.exe"
echo ""
echo "========================================================================"
echo "BUILD COMPLETE: $(date)"
echo "========================================================================"
