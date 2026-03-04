#!/bin/bash
# Isolate which GPU module causes the 5-min crash.
# Strategy: start with all-on, remove one module at a time until stable.
set -e
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/13.1/lib64:/home/drew/WRF_BUILD_GPU/LIBRARIES/netcdf-install/lib:/home/drew/WRF_BUILD_GPU/LIBRARIES/hdf5-install/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export NVCOMPILER_ACC_CUDA_HEAPSIZE=4G
export NV_ACC_CUDA_STACKSIZE=65536

WRF=/home/drew/WRF_BUILD_GPU
PATCHES=/mnt/c/Users/drew/aifs-90d/wrf_gpu
FFLAGS="-O3 -fast -acc -gpu=cc120 -Mfree -Mrecursive -byteswapio -r4 -i4 -mp"
INCS="-I$WRF/frame -I$WRF/inc -I$WRF/share -I$WRF/phys -I$WRF/main -I$WRF/chem -I$WRF/external/io_netcdf -I$WRF/external/io_int -I$WRF/external/esmf_time_f90 -I$WRF/dyn_em -I$WRF/phys/physics_mmm -DGPU_OPENACC -D_ACCEL"

rebuild_and_test() {
    local label="$1"
    echo ""
    echo "============================================"
    echo "TEST: $label"
    echo "============================================"

    cd $WRF/dyn_em
    # Compile all 4
    for f in module_advect_em.f90 module_big_step_utilities_em.f90 module_diffusion_em.f90 module_em.f90; do
        echo -n "  Compiling $f... "
        nvfortran -o ${f%.f90}.o -c $FFLAGS $INCS $f 2>&1 | grep -iE "severe|fatal" | head -1 || echo "OK"
    done

    # Relink
    cd $WRF
    ar r main/libwrflib.a dyn_em/module_advect_em.o dyn_em/module_big_step_utilities_em.o dyn_em/module_diffusion_em.o dyn_em/module_em.o
    ranlib main/libwrflib.a
    cd main && rm -f wrf.exe
    nvfortran -o wrf.exe -mp -O3 -fast -acc -gpu=cc120 -Mfree -Mrecursive -byteswapio -r4 -i4 -acc -gpu=cc120 -cuda \
        wrf.o module_wrf_top.o libwrflib.a \
        $WRF/external/fftpack/fftpack5/libfftpack.a $WRF/external/io_grib1/libio_grib1.a $WRF/external/io_grib_share/libio_grib_share.a \
        $WRF/external/io_int/libwrfio_int.a -L$WRF/external/esmf_time_f90 -lesmf_time $WRF/external/RSL_LITE/librsl_lite.a \
        $WRF/frame/module_internal_header_util.o $WRF/frame/pack_utils.o \
        -L$WRF/external/io_netcdf -lwrfio_nf -L$WRF/LIBRARIES/netcdf-install/lib -lnetcdff -lnetcdf \
        -L$WRF/LIBRARIES/hdf5-install/lib -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lm -lz -ldl 2>&1 | tail -2
    ls -la wrf.exe || { echo "LINK FAILED"; return 1; }

    # Run for 7 min (should crash at ~5 min if buggy)
    cd /home/drew/wrf_gpu_clean_test
    rm -f wrfout_d01_* rsl.out.0000 rsl.error.0000
    ln -sf $WRF/main/wrf.exe .

    # Use timeout, capture exit code properly (no pipe!)
    timeout 480 mpirun -np 1 ./wrf.exe > /tmp/wrf_test_output.log 2>&1
    local rc=$?
    tail -10 /tmp/wrf_test_output.log

    local nfiles=$(ls wrfout_d01_* 2>/dev/null | wc -l)
    echo ""
    echo "Output files: $nfiles"

    # Check for success: either clean exit with multiple files, or timeout (still running)
    if grep -q "SUCCESS COMPLETE WRF" /tmp/wrf_test_output.log; then
        echo "RESULT: $label -> PASSED (completed, $nfiles files)"
        return 0
    elif [ $rc -eq 124 ] && [ $nfiles -ge 2 ]; then
        echo "RESULT: $label -> PASSED (timeout but $nfiles files produced)"
        return 0
    else
        echo "RESULT: $label -> CRASHED (exit $rc, $nfiles files)"
        return 1
    fi
}

# Step 1: Re-enable all ACC
echo "Re-enabling all ACC directives..."
python3 $PATCHES/reenable_acc.py

# Step 2: Confirm all-on crashes
echo ""
echo "========== BASELINE: ALL ON =========="
if rebuild_and_test "all_on"; then
    echo "ALL-ON PASSED?! No crash to isolate."
    exit 0
fi

# Step 3: Remove one at a time
# Try removing module_em first (init/setup routines)
echo ""
echo "========== REMOVING module_em =========="
python3 $PATCHES/reenable_acc.py
python3 $PATCHES/disable_one_acc.py module_em.f90
if rebuild_and_test "no_module_em"; then
    echo ">>> module_em is the culprit!"
    exit 0
fi

# Try removing diffusion
echo ""
echo "========== REMOVING diffusion =========="
python3 $PATCHES/reenable_acc.py
python3 $PATCHES/disable_one_acc.py module_diffusion_em.f90
if rebuild_and_test "no_diffusion"; then
    echo ">>> diffusion is the culprit!"
    exit 0
fi

# Try removing advect
echo ""
echo "========== REMOVING advect =========="
python3 $PATCHES/reenable_acc.py
python3 $PATCHES/disable_one_acc.py module_advect_em.f90
if rebuild_and_test "no_advect"; then
    echo ">>> advect is the culprit!"
    exit 0
fi

# Try removing big_step
echo ""
echo "========== REMOVING big_step =========="
python3 $PATCHES/reenable_acc.py
python3 $PATCHES/disable_one_acc.py module_big_step_utilities_em.f90
if rebuild_and_test "no_big_step"; then
    echo ">>> big_step is the culprit!"
    exit 0
fi

echo ""
echo "No single module removal fixed it — might be interaction between multiple modules."
