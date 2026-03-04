#!/bin/bash
set -e
cd /home/drew/WRF_BUILD_GPU/dyn_em

python3 /mnt/c/Users/drew/aifs-90d/wrf_gpu/disable_bc_acc.py 2>/dev/null || true

python3 -c "
for fname in ['module_advect_em.f90', 'module_big_step_utilities_em.f90', 'module_diffusion_em.f90', 'module_em.f90']:
    with open(fname) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('!\$acc') and not stripped.startswith('!!\$acc'):
            out.append(line.replace('!\$acc', '!!\$acc', 1))
        else:
            out.append(line)
    with open(fname, 'w') as f:
        f.writelines(out)
    print(f'Disabled: {fname}')
"

export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/mpi/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/13.1/lib64:/home/drew/WRF_BUILD_GPU/LIBRARIES/netcdf-install/lib:/home/drew/WRF_BUILD_GPU/LIBRARIES/hdf5-install/lib:$LD_LIBRARY_PATH
WRF=/home/drew/WRF_BUILD_GPU
FFLAGS="-O3 -fast -acc -gpu=cc120 -Mfree -Mrecursive -byteswapio -r4 -i4 -mp"
INCS="-I$WRF/frame -I$WRF/inc -I$WRF/share -I$WRF/phys -I$WRF/main -I$WRF/chem -I$WRF/external/io_netcdf -I$WRF/external/io_int -I$WRF/external/esmf_time_f90 -I$WRF/dyn_em -I$WRF/phys/physics_mmm -DGPU_OPENACC -D_ACCEL"

for f in module_advect_em.f90 module_big_step_utilities_em.f90 module_diffusion_em.f90 module_em.f90; do
    echo -n "  $f... "
    nvfortran -o ${f%.f90}.o -c $FFLAGS $INCS $f 2>&1 | grep -iE "severe|fatal" | head -1 || echo "OK"
done

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
ls -la wrf.exe && echo "BUILD OK"

export OMP_NUM_THREADS=1
export NVCOMPILER_ACC_CUDA_HEAPSIZE=4G
export NV_ACC_CUDA_STACKSIZE=65536
cd /home/drew/wrf_gpu_clean_test
rm -f wrfout_d01_*
ln -sf /home/drew/WRF_BUILD_GPU/main/wrf.exe .
echo "=== Small-step only, 30 min ==="
mpirun -np 1 ./wrf.exe 2>&1 | tail -20
echo ""
echo "Output files:"
ls wrfout_d01_* 2>/dev/null | wc -l
