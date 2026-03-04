# WRF 4.7.1 GPU Port (OpenACC)

GPU-accelerated Weather Research and Forecasting (WRF) model using NVIDIA NVHPC OpenACC, targeting consumer and datacenter NVIDIA GPUs.

## Overview

This project ports WRF 4.7.1's dynamical core to NVIDIA GPUs via OpenACC directives, applied as patches to the stock WRF source. The approach compiles WRF normally with NVHPC, then applies Python-based patch scripts that inject `!$acc kernels`, `!$acc data`, and `!$acc routine` directives into the generated `.f90` files. A final recompile produces GPU-enabled binaries with no changes to WRF's build system.

**What runs on GPU:**
- Dynamical core: `solve_em` (small_step, big_step, diffusion, `module_em`)
- Grid data initialization and management (`module_domain`)

**What runs on CPU:**
- Physics parameterizations (surface layer, microphysics, PBL)
- Advection (`module_advect_em` -- GPU version exists but disabled due to instability)
- Boundary conditions (`module_bc`)
- I/O and WPS preprocessing

## Performance

| Configuration | Grid | Resolution | CPU (gfortran) | GPU (RTX 5090) | Speedup |
|---------------|------|------------|-----------------|-----------------|---------|
| 3km test case | 200x200x80 | 3 km | ~14.4 s/step | ~2.4 s/step | **6x** |
| 250m real-data (Marshall Fire) | 401x401x50 | 250 m | -- | 4.1 s/step | -- |

- Tested on NVIDIA RTX 5090 (32 GB VRAM, Blackwell GB202, sm_120)
- 250m Marshall Fire simulation ran successfully for 30+ minutes of simulated time
- 3km test case completed a full 5-minute simulation
- GPU memory footprint scales with grid size; ~32 GB VRAM supports up to ~400x400 horizontal grid points at 250m

## Supported GPUs

| Architecture | Compute Capability | Example GPUs |
|--------------|-------------------|--------------|
| Blackwell | cc120 | RTX 5090, B200 |
| Hopper | cc90 | H100, H200 |
| Ada Lovelace | cc89 | RTX 4090, L40 |
| Ampere | cc80 | A100, RTX 3090 |

Adjust the `-gpu=ccXX` flag in `configure.wrf` for your target architecture.

## Requirements

- **NVHPC SDK 26.1+** (free download from [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk))
- **Linux** (tested on Ubuntu 22.04 under WSL2; native Linux also works)
- **NetCDF4 + HDF5** compiled with `nvfortran` (see Build Instructions)
- **WRF 4.7.1** source code
- **Python 3.x** (for applying patch scripts)
- **WPS** (Weather Preprocessing System, for generating initial/boundary conditions)

### System Prerequisites (Ubuntu/Debian)

```bash
apt-get install -y curl build-essential m4 csh file libxml2-dev
```

## Quick Start

```bash
# 1. Install NVHPC SDK (see https://developer.nvidia.com/hpc-sdk)
#    Ensure nvfortran, nvc, nvc++ are on PATH

# 2. Build NetCDF/HDF5 with nvfortran
./build_libraries.sh

# 3. Download and configure WRF 4.7.1
cd /path/to/WRF-4.7.1
export NETCDF_classic=1
./configure  # Grep for "PGI" in the menu, pick the dmpar variant
# Edit configure.wrf: add -acc -gpu=cc120,fastmath to FCFLAGS and LDFLAGS

# 4. Initial compile
./compile em_real 2>&1 | tee compile.log

# 5. Apply GPU patches
cd /path/to/wrf-gpu-port
./patches/apply_all_patches.sh /path/to/WRF-4.7.1

# 6. Recompile patched modules and relink
cd /path/to/WRF-4.7.1
./compile em_real 2>&1 | tee compile_gpu.log

# 7. Run
export OMP_NUM_THREADS=1
cd /path/to/run_directory
mpirun -np 1 ./wrf.exe
```

## Build Instructions

### 1. Install NVHPC SDK

Download NVHPC 26.1 or later from NVIDIA. After installation, set up the environment:

```bash
export NVHPC=/opt/nvidia/hpc_sdk
export PATH=$NVHPC/Linux_x86_64/26.1/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC/Linux_x86_64/26.1/compilers/lib:$LD_LIBRARY_PATH
```

### 2. Build NetCDF and HDF5 with nvfortran

WRF's I/O requires NetCDF4 and HDF5. These must be compiled with the same compiler used for WRF:

```bash
# Use the provided helper script
./build_libraries.sh

# Or build manually:
# - HDF5 1.14.x: FC=nvfortran CC=nvc ./configure --enable-fortran --prefix=...
# - NetCDF-C 4.9.x: CC=nvc ./configure --prefix=...
# - NetCDF-Fortran 4.6.x: FC=nvfortran ./configure --prefix=...
```

Set the environment variables before configuring WRF:

```bash
export NETCDF=/path/to/netcdf
export HDF5=/path/to/hdf5
```

### 3. Configure WRF

```bash
cd WRF-4.7.1
export NETCDF_classic=1
./configure
```

When the configuration menu appears, look for the **PGI/NVHPC** entries (grep for "PGI" in the list) and select the **dmpar** variant.

Edit `configure.wrf` to add GPU flags:

```makefile
# Add to FCOPTIM or FCFLAGS:
-acc -gpu=cc120,fastmath

# Add to LDFLAGS:
-acc -gpu=cc120,fastmath -cuda

# Set compilers:
DM_FC = nvfortran
SFC   = nvfortran

# Ensure FCBASEOPTS includes $(FORMAT_FREE) (critical for ESMF .f files):
FCBASEOPTS = -w $(FCDEBUG) $(FORMAT_FREE) $(BYTESWAPIO) -Mrecursive $(OMP)
```

Add **only** `-DGPU_OPENACC` to `ARCH_LOCAL`:

```makefile
ARCH_LOCAL = -DGPU_OPENACC
```

> **WARNING:** Do **not** add `-D_ACCEL` to `ARCH_LOCAL`. The `_ACCEL` flag enables broken legacy OpenACC 1.0 directives baked into WRF's WSM3/WSM5 microphysics code, which will cause compile or runtime failures. Use only `-DGPU_OPENACC`.

Replace `cc120` with your GPU's compute capability (e.g., `cc89` for RTX 4090, `cc80` for A100).

### 4. Initial Compile

```bash
./compile em_real 2>&1 | tee compile.log
```

This produces a CPU-only WRF binary compiled with NVHPC. The initial compile generates the `.f90` files that the patches target.

### 5. Apply GPU Patches

```bash
cd /path/to/wrf-gpu-port
./patches/apply_all_patches.sh /path/to/WRF-4.7.1
```

The patch scripts inject OpenACC directives into the following modules:

| Module | File | What it does |
|--------|------|--------------|
| `module_domain` | `frame/module_domain.F` | GPU data init (`!$acc copyin` for ~50 grid arrays) |
| `module_small_step_em` | `dyn_em/module_small_step_em.f90` | Acoustic/gravity wave integration |
| `module_big_step_utilities_em` | `dyn_em/module_big_step_utilities_em.f90` | `calc_alt`, `calc_php`, `restore_ph_mu` |
| `module_diffusion_em` | `dyn_em/module_diffusion_em.f90` | Horizontal and vertical diffusion |
| `module_em` | `dyn_em/module_em.f90` | Top-level dynamics driver |
| `solve_em` | `dyn_em/solve_em.f90` | Main time-step solver |

### 6. Recompile

```bash
cd /path/to/WRF-4.7.1
./compile em_real 2>&1 | tee compile_gpu.log
```

**Build gotcha:** Modifying `frame/module_domain.F` can trigger a recompile that fails on `test_adjust_io_timestr` due to `__FILE__` macro expansion. Workaround: edit the generated `.f90` file directly, then `touch -r module_domain.o module_domain.F module_domain.f90`.

## Running

### Environment Setup

```bash
# CRITICAL: You MUST set this. OpenMP threading conflicts with OpenACC GPU
# execution and will cause silent hangs or incorrect results if not disabled.
export OMP_NUM_THREADS=1
```

### Single-GPU Execution

```bash
cd /path/to/run_directory
# Ensure wrfinput_d01, wrfbdy_d01, and namelist.input are present
mpirun -np 1 ./wrf.exe
```

### Namelist Considerations

Standard WRF namelists work without modification. For 250m runs, recommended settings:

```fortran
&domains
 dx = 250,
 dy = 250,
 e_we = 401,
 e_sn = 401,
 e_vert = 50,
/

&dynamics
 diff_opt = 1,
 km_opt = 4,
/
```

### Verifying GPU Execution

Set `NV_ACC_NOTIFY=1` or `NVCOMPILER_ACC_NOTIFY=1` to see GPU kernel launches in stderr:

```bash
NV_ACC_NOTIFY=1 mpirun -np 1 ./wrf.exe 2>&1 | head -50
```

You should see messages like:
```
launch CUDA kernel  file=/path/to/module_small_step_em.f90 ...
```

## Architecture

```
Host (CPU)                          Device (GPU)
-----------                         ------------
WPS preprocessing
I/O (NetCDF read/write)
Physics:                            Dynamics:
  - sf_sfclay_physics                 - small_step (acoustic)
  - mp_physics (WSM6)                 - big_step (calc_alt, calc_php)
  - bl_pbl_physics (YSU)             - diffusion (horiz + vert)
  - radiation                         - module_em utilities
                                      - solve_em driver
        |                                    ^
        |  CPU arrays ----copyin/out----> GPU arrays
        v                                    |
  Physics tendencies ----update----> Dynamics state
```

Data transfer between CPU physics and GPU dynamics happens via OpenACC `!$acc update` directives at the boundaries of the dynamical core in `solve_em`. Grid arrays (~50 fields including u, v, w, theta, pressure, moisture) are resident on GPU for the duration of the time step.

## Known Issues

1. **GPU acceleration scope** -- GPU kernels currently cover specific dynamics modules: `small_step`, `big_step`, `diffusion`, and `module_em`. Advection and all physics parameterizations remain on CPU. The observed 3-4x speedup over gfortran comes primarily from NVHPC compiler optimizations (vectorization, instruction scheduling) rather than GPU offload. Full GPU speedup will require porting advection and physics.

2. **`module_advect_em` GPU causes instability** -- The advection module has 143 ACC directives applied but produces numerical instability after approximately 5 minutes of simulated time. This module is disabled by default (runs on CPU). Debugging is ongoing; the issue is likely related to array indexing or race conditions in the Runge-Kutta advection loop.

3. **`OMP_NUM_THREADS` must be 1** -- OpenMP threading conflicts with OpenACC GPU execution. Always set `OMP_NUM_THREADS=1` before running.

4. **`module_bc` ACC disabled** -- Boundary condition module patches produce `present()` data clause errors at runtime. Boundary conditions run on CPU.

5. **VRAM limits grid size** -- On a 32 GB GPU, the practical limit is approximately 400x400 horizontal grid points at 250m resolution with 50 vertical levels. Larger grids require more VRAM or domain decomposition across multiple GPUs (not yet tested).

6. **Physics on CPU** -- All physics parameterizations remain on CPU. This means data must be transferred between GPU and CPU each time step, which limits the achievable speedup. Porting physics (especially WSM6 microphysics) is a future goal.

7. **Single-GPU only** -- Multi-GPU domain decomposition via MPI has not been tested. The port currently targets single-GPU execution with `mpirun -np 1`.

8. **Build sensitivity** -- Touching certain framework files (especially `module_domain.F`) can trigger cascading recompiles that fail. Use the `.f90`-level editing workaround described in Build Instructions.

9. **Do not use `-D_ACCEL`** -- WRF's WSM3/WSM5 contain ancient OpenACC 1.0 directives guarded by `_ACCEL` that are incompatible with modern NVHPC. Use only `-DGPU_OPENACC` in `ARCH_LOCAL`.

## Repository Structure

```
wrf-gpu-port/
  README.md              This file
  patches/               Python patch scripts and apply_all_patches.sh
  docs/                  Additional documentation
  test_cases/            Sample namelists and test configurations
  utils/                 Helper scripts (build_libraries.sh, etc.)
```

## Contributing

Contributions are welcome, particularly in these areas:

- **Advection stability** -- Debugging `module_advect_em` GPU execution
- **Physics porting** -- Moving WSM6, YSU, or radiation to GPU
- **Multi-GPU** -- MPI domain decomposition with GPU-resident data
- **Testing** -- Validation against CPU results on different grid configurations
- **Additional GPU architectures** -- Testing on Hopper, Ada, or Ampere hardware

To contribute:

1. Fork this repository
2. Create a feature branch
3. Test against the 3km test case (5-minute simulation, verify output matches CPU within roundoff)
4. Submit a pull request with a description of changes and test results

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

Note: WRF itself is distributed under its own [public domain license](https://github.com/wrf-model/WRF/blob/master/LICENSE.txt). This repository contains only patch scripts and build tools, not WRF source code. WRF is developed and maintained by the [National Center for Atmospheric Research (NCAR)](https://www.mmm.ucar.edu/models/wrf). This GPU port is an independent community contribution and is not affiliated with or endorsed by NCAR.

## Acknowledgments

- [WRF Model](https://github.com/wrf-model/WRF) by NCAR/UCAR
- [NVIDIA NVHPC SDK](https://developer.nvidia.com/hpc-sdk) for OpenACC compiler support
