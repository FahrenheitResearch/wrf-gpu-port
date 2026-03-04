#!/usr/bin/env bash
# build_libraries.sh — Build zlib + HDF5 + NetCDF-C + NetCDF-Fortran with NVHPC for WRF GPU builds
#
# Usage:
#   ./build_libraries.sh [--prefix /path/to/install] [--gpu cc120]
#
# Defaults:
#   --prefix  $HOME/wrf_gpu_libs
#   --gpu     cc120  (NVIDIA Blackwell / RTX 5090)

set -euo pipefail

# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
ZLIB_VERSION="1.3.1"
HDF5_VERSION="1.14.4"          # 1.14.x series
HDF5_VERSION_SHORT="1.14"      # for mirror path
NETCDF_C_VERSION="4.9.2"
NETCDF_F_VERSION="4.6.1"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
PREFIX="${HOME}/wrf_gpu_libs"
GPU_CC="cc120"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"; shift 2 ;;
        --gpu)
            GPU_CC="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--prefix DIR] [--gpu ccXXX]"
            echo "  --prefix DIR   Install root (default: \$HOME/wrf_gpu_libs)"
            echo "  --gpu ccXXX    GPU compute capability (default: cc120)"
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate NVHPC SDK
# ---------------------------------------------------------------------------
NVHPC_BASE="/opt/nvidia/hpc_sdk"
NVHPC_ARCH="Linux_x86_64"

find_nvhpc_version() {
    # Pick the highest installed version
    if [[ ! -d "${NVHPC_BASE}/${NVHPC_ARCH}" ]]; then
        echo ""
        return
    fi
    ls -1 "${NVHPC_BASE}/${NVHPC_ARCH}/" 2>/dev/null \
        | grep -E '^[0-9]+\.[0-9]+$' \
        | sort -t. -k1,1nr -k2,2nr \
        | head -1
}

NVHPC_VERSION=$(find_nvhpc_version)
if [[ -z "${NVHPC_VERSION}" ]]; then
    echo "ERROR: No NVHPC installation found under ${NVHPC_BASE}/${NVHPC_ARCH}/" >&2
    echo "       Install NVHPC SDK from https://developer.nvidia.com/hpc-sdk" >&2
    exit 1
fi

NVHPC_COMPILERS="${NVHPC_BASE}/${NVHPC_ARCH}/${NVHPC_VERSION}/compilers"
NVHPC_MATH="${NVHPC_BASE}/${NVHPC_ARCH}/${NVHPC_VERSION}/math_libs"
NVHPC_CUDA="${NVHPC_BASE}/${NVHPC_ARCH}/${NVHPC_VERSION}/cuda"

echo "======================================================================"
echo " NVHPC version   : ${NVHPC_VERSION}"
echo " Compilers       : ${NVHPC_COMPILERS}/bin"
echo " GPU target      : ${GPU_CC}"
echo " Install prefix  : ${PREFIX}"
echo "======================================================================"

# Source the NVHPC environment
NVHPC_ENV_SCRIPT="${NVHPC_BASE}/${NVHPC_ARCH}/${NVHPC_VERSION}/compilers/bin/nvhpc-env.sh"
if [[ -f "${NVHPC_ENV_SCRIPT}" ]]; then
    # shellcheck source=/dev/null
    source "${NVHPC_ENV_SCRIPT}"
else
    # Manually prepend compilers/bin; NVHPC sometimes skips the env script
    export PATH="${NVHPC_COMPILERS}/bin:${PATH}"
    export LD_LIBRARY_PATH="${NVHPC_COMPILERS}/lib:${NVHPC_MATH}/lib64:${NVHPC_CUDA}/lib64:${LD_LIBRARY_PATH:-}"
    export MANPATH="${NVHPC_COMPILERS}/man:${MANPATH:-}"
fi

# Verify compilers are accessible
for tool in nvc nvc++ nvfortran; do
    if ! command -v "${tool}" &>/dev/null; then
        echo "ERROR: ${tool} not found in PATH after sourcing NVHPC environment." >&2
        echo "       PATH = ${PATH}" >&2
        exit 1
    fi
done

echo "Compiler versions:"
nvc     --version 2>&1 | head -1
nvc++   --version 2>&1 | head -1
nvfortran --version 2>&1 | head -1
echo ""

# ---------------------------------------------------------------------------
# Environment for builds
# ---------------------------------------------------------------------------
export CC="nvc"
export CXX="nvc++"
export FC="nvfortran"
export F77="nvfortran"
export F90="nvfortran"

# NVHPC flags — fastmath + target GPU, keep strict-aliasing off for HDF5
COMMON_CFLAGS="-O2 -tp=native"
COMMON_FFLAGS="-O2 -tp=native -Mfreeform"
export CFLAGS="${COMMON_CFLAGS}"
export CXXFLAGS="${COMMON_CFLAGS}"
export FFLAGS="${COMMON_FFLAGS}"
export FCFLAGS="${COMMON_FFLAGS}"
export LDFLAGS="-L${PREFIX}/lib"
export CPPFLAGS="-I${PREFIX}/include"

# ---------------------------------------------------------------------------
# Build directories
# ---------------------------------------------------------------------------
BUILD_DIR="${PREFIX}/build"
mkdir -p "${BUILD_DIR}" "${PREFIX}/lib" "${PREFIX}/include" "${PREFIX}/bin"

cd "${BUILD_DIR}"

log_step() { echo; echo "----------------------------------------------------------------------"; echo " $*"; echo "----------------------------------------------------------------------"; }

# ---------------------------------------------------------------------------
# Helper: download if not already present
# ---------------------------------------------------------------------------
fetch() {
    local url="$1"
    local file="${url##*/}"
    if [[ ! -f "${file}" ]]; then
        echo "Downloading ${file} ..."
        curl -fsSL -o "${file}" "${url}"
    else
        echo "Already downloaded: ${file}"
    fi
}

# ---------------------------------------------------------------------------
# 1. zlib
# ---------------------------------------------------------------------------
log_step "Building zlib ${ZLIB_VERSION}"

ZLIB_DIR="${BUILD_DIR}/zlib-${ZLIB_VERSION}"
ZLIB_URL="https://zlib.net/zlib-${ZLIB_VERSION}.tar.gz"

fetch "${ZLIB_URL}"

if [[ ! -d "${ZLIB_DIR}" ]]; then
    tar -xzf "zlib-${ZLIB_VERSION}.tar.gz"
fi

(
    cd "${ZLIB_DIR}"
    # zlib uses its own configure, not autoconf — CC is enough
    CC="${CC}" CFLAGS="${COMMON_CFLAGS}" \
        ./configure --prefix="${PREFIX}" --static
    make -j"$(nproc)"
    make install
)
echo "zlib installed to ${PREFIX}"

# ---------------------------------------------------------------------------
# 2. HDF5
# ---------------------------------------------------------------------------
log_step "Building HDF5 ${HDF5_VERSION}"

HDF5_TARBALL="hdf5-${HDF5_VERSION}.tar.gz"
HDF5_DIR="${BUILD_DIR}/hdf5-${HDF5_VERSION}"
HDF5_URL="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION_SHORT}/hdf5-${HDF5_VERSION}/src/${HDF5_TARBALL}"
HDF5_URL_ALT="https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/${HDF5_TARBALL}"

if [[ ! -f "${HDF5_TARBALL}" ]]; then
    echo "Trying primary HDF5 mirror..."
    curl -fsSL -o "${HDF5_TARBALL}" "${HDF5_URL}" || {
        echo "Primary mirror failed, trying GitHub release..."
        curl -fsSL -o "${HDF5_TARBALL}" "${HDF5_URL_ALT}"
    }
else
    echo "Already downloaded: ${HDF5_TARBALL}"
fi

if [[ ! -d "${HDF5_DIR}" ]]; then
    tar -xzf "${HDF5_TARBALL}"
fi

(
    cd "${HDF5_DIR}"
    # Disable C++ and HL to keep it simple; WRF only needs the C + Fortran interfaces
    ./configure \
        --prefix="${PREFIX}" \
        --with-zlib="${PREFIX}" \
        --enable-fortran \
        --disable-cxx \
        --disable-hltools \
        --disable-shared \
        --enable-static \
        --disable-tests \
        CC="${CC}" FC="${FC}" CFLAGS="${COMMON_CFLAGS}" FCFLAGS="${COMMON_FFLAGS}" \
        LDFLAGS="${LDFLAGS}" CPPFLAGS="${CPPFLAGS}"
    make -j"$(nproc)"
    make install
)
echo "HDF5 installed to ${PREFIX}"

# Update flags for downstream libraries
export LDFLAGS="-L${PREFIX}/lib ${LDFLAGS}"
export CPPFLAGS="-I${PREFIX}/include ${CPPFLAGS}"

# ---------------------------------------------------------------------------
# 3. NetCDF-C
# ---------------------------------------------------------------------------
log_step "Building NetCDF-C ${NETCDF_C_VERSION}"

NETCDF_C_TARBALL="netcdf-c-${NETCDF_C_VERSION}.tar.gz"
NETCDF_C_DIR="${BUILD_DIR}/netcdf-c-${NETCDF_C_VERSION}"
NETCDF_C_URL="https://github.com/Unidata/netcdf-c/archive/refs/tags/v${NETCDF_C_VERSION}.tar.gz"

if [[ ! -f "${NETCDF_C_TARBALL}" ]]; then
    echo "Downloading NetCDF-C ${NETCDF_C_VERSION}..."
    curl -fsSL -L -o "${NETCDF_C_TARBALL}" "${NETCDF_C_URL}"
else
    echo "Already downloaded: ${NETCDF_C_TARBALL}"
fi

if [[ ! -d "${NETCDF_C_DIR}" ]]; then
    tar -xzf "${NETCDF_C_TARBALL}"
fi

(
    cd "${NETCDF_C_DIR}"
    ./configure \
        --prefix="${PREFIX}" \
        --disable-shared \
        --enable-static \
        --disable-dap \
        --disable-byterange \
        --disable-tests \
        CC="${CC}" CFLAGS="${COMMON_CFLAGS}" \
        LDFLAGS="${LDFLAGS}" CPPFLAGS="${CPPFLAGS}" \
        LIBS="-lhdf5_fortran -lhdf5 -lz -lm -ldl"
    make -j"$(nproc)"
    make install
)
echo "NetCDF-C installed to ${PREFIX}"

# nc-config needed by NetCDF-Fortran
export PATH="${PREFIX}/bin:${PATH}"

# ---------------------------------------------------------------------------
# 4. NetCDF-Fortran
# ---------------------------------------------------------------------------
log_step "Building NetCDF-Fortran ${NETCDF_F_VERSION}"

NETCDF_F_TARBALL="netcdf-fortran-${NETCDF_F_VERSION}.tar.gz"
NETCDF_F_DIR="${BUILD_DIR}/netcdf-fortran-${NETCDF_F_VERSION}"
NETCDF_F_URL="https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v${NETCDF_F_VERSION}.tar.gz"

if [[ ! -f "${NETCDF_F_TARBALL}" ]]; then
    echo "Downloading NetCDF-Fortran ${NETCDF_F_VERSION}..."
    curl -fsSL -L -o "${NETCDF_F_TARBALL}" "${NETCDF_F_URL}"
else
    echo "Already downloaded: ${NETCDF_F_TARBALL}"
fi

if [[ ! -d "${NETCDF_F_DIR}" ]]; then
    tar -xzf "${NETCDF_F_TARBALL}"
fi

(
    cd "${NETCDF_F_DIR}"
    # NetCDF-Fortran configure picks up nc-config from PATH for NETCDF_C_CFLAGS/LIBS
    NCDIR="${PREFIX}"
    NC_CONFIG="${PREFIX}/bin/nc-config"
    NETCDF_C_LIBS=$(${NC_CONFIG} --libs)

    ./configure \
        --prefix="${PREFIX}" \
        --disable-shared \
        --enable-static \
        --disable-tests \
        CC="${CC}" FC="${FC}" F77="${F77}" \
        CFLAGS="${COMMON_CFLAGS}" FFLAGS="${COMMON_FFLAGS}" FCFLAGS="${COMMON_FFLAGS}" \
        LDFLAGS="${LDFLAGS}" CPPFLAGS="${CPPFLAGS}" \
        LIBS="${NETCDF_C_LIBS} -lhdf5_fortran -lhdf5 -lz -lm -ldl"
    make -j"$(nproc)"
    make install
)
echo "NetCDF-Fortran installed to ${PREFIX}"

# ---------------------------------------------------------------------------
# Summary and environment export snippet
# ---------------------------------------------------------------------------
log_step "Build complete — summary"

echo "  zlib             ${PREFIX}  (static)"
echo "  HDF5             ${PREFIX}  (static, Fortran interface)"
echo "  NetCDF-C         ${PREFIX}  (static, no DAP)"
echo "  NetCDF-Fortran   ${PREFIX}  (static)"
echo ""
echo "Library inventory:"
ls -lh "${PREFIX}/lib/"libz*.a "${PREFIX}/lib/"libhdf5*.a "${PREFIX}/lib/"libnetcdf*.a 2>/dev/null || true
echo ""
echo "Fortran module check (netcdf.mod):"
ls "${PREFIX}/include/netcdf.mod" 2>/dev/null && echo "  OK" || echo "  WARNING: netcdf.mod not found"
echo ""

cat <<EOF
----------------------------------------------------------------------
 Environment variables for WRF configure
 Add these to your shell rc or source the snippet below:
----------------------------------------------------------------------
export NVHPC_ROOT="${NVHPC_BASE}/${NVHPC_ARCH}/${NVHPC_VERSION}"
export PATH="\${NVHPC_ROOT}/compilers/bin:${PREFIX}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${NVHPC_ROOT}/compilers/lib:\${NVHPC_ROOT}/cuda/lib64:${PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

# WRF library paths
export NETCDF="${PREFIX}"
export HDF5="${PREFIX}"
export ZLIB="${PREFIX}"
export LDFLAGS="-L${PREFIX}/lib"
export CPPFLAGS="-I${PREFIX}/include"
export LIBS="-lnetcdff -lnetcdf -lhdf5_fortran -lhdf5 -lz -lm -ldl"

# NVHPC compilers
export CC="nvc"
export CXX="nvc++"
export FC="nvfortran"
export F77="nvfortran"

# GPU target (used in WRF configure.wrf FCFLAGS)
export WRF_GPU_CC="${GPU_CC}"
----------------------------------------------------------------------
EOF

echo ""
echo "All libraries built successfully."
