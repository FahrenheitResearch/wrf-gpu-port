# WRF GPU Port — Progress Log
## RTX 5090 (Blackwell sm_120) via NVHPC 26.1 OpenACC

---

## Performance Timeline

| Build | Date/Time | Step Time (3km) | Speedup vs CPU | What Changed |
|-------|-----------|-----------------|----------------|--------------|
| Baseline (CPU) | pre-03/03 | 2.4s | 1.0× | nvfortran CPU + 3 GPU routines (calc_alt, calc_php, restore_ph_mu) |
| Build 1 (auto-copy) | 03/03 02:37 | 25s | 0.1× | All patches applied but no data management → 876K auto-copies/step |
| Build 2 (advect fix) | 03/03 03:25 | **1.34s** | **1.8×** | `!$acc data create()` for advect_scalar_pd local arrays eliminated 874K uploads |
| Build 3 (all advect) | 03/03 03:49 | **1.30s** | **1.85×** | Fixed advect_w + advect_scalar vflux + config_flags copyin |
| Build 4 (diffusion) | 03/03 04:09 | **1.30s** | **1.85×** | Diffusion compiled CPU-only (clean .o), zero compile errors |
| Build 5 (kernel fusion) | 03/03 ~10:00 | **1.13s** | **2.12×** | 55 `!$acc kernels`→`parallel loop gang` (advect+big_step) |
| Build 6 (GPU diffusion) | 03/03 ~12:00 | **0.42s** | **5.7×** | Comprehensive diffusion data create (18 subs, 106 arrays) + OMP=4 |
| Build 7 (YSU GPU) | 03/03 | 110s | 0.02× | YSU routine seq — hangs/250× slower. **REVERTED** |
| **Build 8a** | **03/03** | **0.585s** | **4.1×** | WSM6 present(), YSU CPU-only. **HAD NaN PAST FRAME 0** |
| NaN debug | 03/03 PM | — | — | Discovered ALL builds 2-8a had NaN in output after timestep 0. Rolled back everything. |
| CPU baseline | 03/03 PM | 2.4s | 1.0× | Stripped all patches, pure nvfortran CPU. Verified ZERO NaN. |
| Small-step GPU | 03/03 PM | **1.75s** | **1.37×** | 8 small-step kernels only. Root cause of NaN found & fixed. **VERIFIED CORRECT** |
| **Build 9 (all dynamics)** | **03/03 EVE** | **1.6s** | **1.5×** | All dynamics on GPU (small_step+advect+big_step+diffusion+module_em). OMP=1 required. **VERIFIED CORRECT** |

### CRITICAL: OMP_NUM_THREADS must be 1 with GPU
OMP=4 corrupts data — GPU computes full domain but OMP tiles cause race conditions in CPU↔GPU data transfers.
75% of PSFC points zeroed with OMP=4, perfect output with OMP=1.
Build 6's 0.42s was with OMP=4 — that speed included corrupt data.

---

## NaN Root Cause (FOUND & FIXED 03/03)

**Problem**: ALL builds 2-8a produced NaN in output after the initial frame. We didn't notice because we were only checking timing, not data quality.

**Root cause**: `gpu_init_domain_data` copies grid arrays to GPU at startup, but derived quantities (`mut`, `muts`, `muu`, `muv`, `alt`, `php`, etc.) are computed AFTER gpu_init runs. GPU had zeros for these arrays. Then the GPU→CPU sync at line 878 in solve_em.F (before physics) copied GPU's zeros back to host, destroying correct host values.

**Smoking gun**: Diagnostic prints inside advance_w showed `mut min/max = 0.0, 0.0` on GPU. Should be 56000-85000.

**Fix**: Added `!$acc update device(grid%mut, grid%muts, ...)` at start of solve_em, BEFORE the GPU→CPU sync. This pushes all derived quantities from host to GPU after they're computed. Applied by `fix_gpu_init_timing.py`.

**Additional root cause for builds 2-8a**: `patch_wsm6_gpu.py` modified actual Fortran code, not just ACC directives — zeroed snow/graupel accumulators and removed write-back to grid arrays. This was the ONLY patch that changed code logic (all dynamics patches were ACC-only).

## Current State (Build 10 — 03/03 night)

### ISOLATION TEST RESULTS — advect_em is the crash culprit

All-on (5 modules) crashes at ~5 min sim time (W blows up to 158 m/s in upper levels).
Systematic removal identified the bug:

| Test Config | Result | Step Time | Notes |
|-------------|--------|-----------|-------|
| All on (small_step+advect+big_step+diffusion+module_em) | **CRASHED** at 17:04:24 | ~2.0s | Segfault, W=158 m/s at (76,90,60) |
| Remove module_em | **CRASHED** at 17:05:12 | ~2.0s | Same W blowup |
| Remove diffusion | **CRASHED** at 17:04:24 | ~2.3s | Same segfault |
| **Remove advect** | **PASSED 30 min** | **1.6s** | 7 output files, all correct |
| Small-step only | **PASSED 30 min** | 2.0s | 7 output files, all correct |

**Conclusion**: `module_advect_em.f90` (143 ACC directives) is the sole source of instability.

### What's Working — Build 10 (no-advect GPU)
- **ON GPU**: small_step (8 kernels) + big_step (110 directives) + diffusion (40 directives) + module_em (42 directives)
- **ON CPU**: advect (143 ACC directives disabled), module_bc (disabled), all physics
- **1.6 s/step** on 3km 200×200×80 with OMP=1 (vs 9.7s gfortran = **6× speedup**)
- **30-minute simulation verified**: 7 output files, zero NaN, physically correct T2/U10/V10/PSFC/HFX/W
- Plots: `wrf_gpu/gpu_30min_no_advect.png`, `wrf_gpu/gpu_30min_W_evolution.png`
- **MUST USE OMP_NUM_THREADS=1** — OMP>1 corrupts data

### What's NOT on GPU (running on CPU)
- **module_advect_em**: DISABLED — causes W blowup at ~5 min. Need to debug 143 ACC directives
- module_bc: ACC disabled (`present(dat)` errors)
- Physics: WSM6, YSU, sfclay all on CPU

### Next Steps
1. **Debug advect ACC patches** — the 143 directives in advect_em cause instability. Could be missing data create for locals, race condition in parallel loops, or incorrect collapse/gang directives
2. **250m idealized LES test** — em_les case with current build to prove GPU at 250m resolution
3. Physics GPU: sfclay (safe), WSM6 (use built-in ACC only)
4. Profile with nsys to find remaining bottlenecks

### Multi-Domain Scaling (Build 8a — current working build)

| Domain | Grid | GPU Step (OMP=4) | VRAM Peak | Status |
|--------|------|------------------|-----------|--------|
| 3km 200×200×80 | 3.2M | **0.585s** | 31,585 MB (96%) | SUCCESS |
| 1km 400×400×80 | 12.8M | **2.20s** | 31,690 MB (97%) | SUCCESS |
| 250m 401×401×50 | 8.0M | **1.38s** | 31,758 MB (97%) | SUCCESS |

**NOTE**: Build 6 (0.42s) exe now hangs — cannot reproduce. Build 8a is the only working exe.
**CRITICAL**: VRAM at 96-97% for ALL domains = gpu_init copies ~55 GB regardless of domain size.
~24 GB spilling to shared RAM (system RAM via PCIe, ~56× bandwidth penalty).

### Multi-Domain Scaling (Build 6 — historical, exe broken)

| Domain | Grid Points | GPU Step (OMP=4) | CPU Step (gfortran) | Speedup | VRAM |
|--------|------------|------------------|---------------------|---------|------|
| 200×200×80 | 3.2M | **0.42s** | 9.7s | **23.1×** | ~19 GB |
| 400×400×80 | 12.8M | **2.07s** | 13.4s | **6.5×** | 31.2 GB (96%) |

Key insights:
- OMP_NUM_THREADS=4 gives 2.3-2.7× speedup on CPU-side code (diffusion, etc)
- GPU diffusion (Build 6) adds another 12-16% on top of OMP
- Sub-linear GPU scaling: 4× grid (200→400) = 4.9× time
- VRAM limit: 400×400×80 = 31.2 GB (96%). Max feasible: ~500×500×50
- 1000×1000×50 needs ~58 GB — won't fit on single RTX 5090

### Nsight Systems Profile (Build 4, 200×200×80)

| Rank | Kernel | % GPU Time | Time (ms) | Calls |
|------|--------|-----------|-----------|-------|
| 1 | advect_scalar_pd line 7738 | 68.2% | 525 | 30 |
| 2 | sfclayrev line 157 | 8.7% | 67 | 5 |
| 3 | advance_w line 1380 | 3.5% | 27 | 35 |
| 4-5 | advect_scalar lines 4249/4266 | 4.1% | 31 | 29,850 |
| 6-7 | advance_uv lines 860/933 | 3.1% | 24 | 70 |

CUDA API overhead: cuLaunchKernel 410ms (79,659 launches), cuStreamSync 1148ms.
After Build 5 fusion: launches reduced ~75% (55 regions fused).

### Key Insight: Local Array Data Management
The massive slowdown (25s→1.34s) was caused by OpenACC auto-copying large local arrays for every kernel invocation. Fix: `!$acc data create()` regions around subroutine bodies allocate once on GPU, zero PCIe transfers.

### Transfer Budget (Build 4, per 5-step run)

| Source | Uploads | Downloads | Total MB | Notes |
|--------|---------|-----------|----------|-------|
| gpu_init (one-time) | 1,538 | 0 | 835 | Domain setup, not per-step |
| set_physical_bc3d | 27 | 27 | 760 | `present_or_copy(dat)` round-trip |
| sfclayrev | 250 | 240 | ~180 | u3d/v3d + 2D fields + scalars |
| advect_scalar vflux | 75 | 75 | 10 | Small 64KB local array |
| config_flags | 5 | 0 | 0.2 | copyin at solve_em entry |
| **Total per-step** | ~80 | ~70 | **~190** | ~3ms at 64 GB/s = negligible |

Transfers are no longer the bottleneck — GPU compute and kernel launch overhead dominate.

---

## Current State (Build 6 — 03/03 ~12:00)

### What's On GPU
- **Dynamics**: small_step (46 `!$acc parallel loop`), advect (39 fused `parallel loop gang` + remaining), big_step (16 fused + 16 original), module_em (42)
- **Diffusion**: Compiled WITH `-acc`, 178 GPU kernels across 21 subroutines, comprehensive data create for 106 local arrays in 18 subs
- **Kernel fusion**: 55 `!$acc kernels` → `!$acc parallel loop gang` (fuse_kernels.py)
- **BC**: module_bc with `present_or_copy(dat)` — graceful CPU/GPU handling
- **Physics**: WSM6 + sfclay (GPU), YSU (CPU-only)
- **GPU Init**: 1404 grid% array fields copyin'd at domain setup
- **solve_em**: `!$acc data create()` for 56 local arrays + `copyin(config_flags)`
- **advect locals**: `!$acc data create()` in all 5 subroutines
- **advect_scalar_pd**: ph_low+flux_out fused into single `parallel loop collapse(3)`

### What's CPU-Only
- **YSU PBL**: CPU-only (column physics, 40K columns)
- **Noah LSM, RRTMG radiation**: not ported yet
- **OpenMP**: OMP_NUM_THREADS=4 for CPU-side parallelism (2.3× speedup on CPU code)

### Performance Analysis (Build 6)
- 200×200 OMP=4: **0.42s/step** (23.1× vs gfortran CPU)
- 400×400 OMP=4: **2.07s/step** (6.5× vs gfortran CPU)
- Diffusion on GPU: 12-16% speedup vs CPU-only diffusion
- VRAM: 31.2 GB at 400×400×80 (96% full). Max feasible ~500×500×50

---

## Patch Inventory (16 original + 6 new)

### Core Patches (Wave 1)
| Patch Script | Target | Directives | Status |
|-------------|--------|-----------|--------|
| `patch_small_step_gpu.py` | module_small_step_em.f90 | 46 `!$acc parallel loop` | APPLIED |
| `patch_advect_gpu.py` | module_advect_em.f90 | 82 kernel regions | APPLIED |
| `patch_big_step_gpu.py` | module_big_step_utilities_em.f90 | 32 regions | APPLIED |
| `patch_diffusion_gpu.py` | module_diffusion_em.f90 | 361 directives | APPLIED (CPU-only compile) |
| `patch_wsm6_gpu.py` | WSM6 microphysics | Driver data region | APPLIED |
| `patch_sfclay_gpu.py` | sf_sfclayrev.F90 | private() fix | APPLIED |

### Infrastructure Patches (Wave 2)
| Patch Script | Target | Directives | Status |
|-------------|--------|-----------|--------|
| `patch_bc_gpu.py` | module_bc.f90 | 48 directives + present_or_copy | APPLIED |
| `patch_module_em_gpu.py` | module_em.f90 | 42 directives | APPLIED |
| `patch_rhs_ph_gpu.py` | big_step_utilities | 20 directives | APPLIED |
| `patch_remaining_bigstep_gpu.py` | big_step_utilities | 21 regions | APPLIED |
| `patch_first_rk_gpu.py` | first_rk + physics_addtendc | 26 patches | APPLIED |

### LES Patches (Wave 3) — directives in source but diffusion CPU-only
| Patch Script | Target | Directives | Status |
|-------------|--------|-----------|--------|
| `patch_gpu_init_les.py` | module_domain.f90 | 21 LES fields | APPLIED |
| `patch_les_deform_gpu.py` | module_diffusion_em.f90 | 18 directives | IN SOURCE |
| `patch_vert_diff_gpu.py` | module_diffusion_em.f90 | 39 patches | IN SOURCE |
| `patch_tke_gpu.py` | module_diffusion_em.f90 | 18 patches | IN SOURCE |

### Data Management Patches (New)
| Patch Script | Target | Purpose | Status |
|-------------|--------|---------|--------|
| `patch_solve_em_gpu.py` | solve_em.f90 | `!$acc data create()` for 56 local arrays + copyin(config_flags) | APPLIED |
| `build_gpu_init_from_struct.py` | module_domain.f90 | Comprehensive gpu_init (1404 arrays) | APPLIED |
| `patch_advect_locals.py` | module_advect_em.f90 | `!$acc data create()` for advect local arrays | APPLIED (all 5 subs) |
| `fix_advect_w_create.py` | module_advect_em.f90 | advect_w create(vflux, fqx, fqy) | APPLIED |
| `fix_addtendc.py` | module_physics_addtendc.f90 | Remove duplicate directives | APPLIED |
| `fix_all_present_clauses.py` | Multiple files | present() → default(present) | ABANDONED (breaks local arrays) |
| `fuse_kernels.py` | advect+big_step | 55 `!$acc kernels`→`parallel loop gang` | APPLIED (Build 5) |
| `patch_fuse_advect_kernels.py` | advect_scalar_pd | Fuse ph_low+flux_out kernels | APPLIED (Build 5) |

---

## Lessons Learned

### default(present) DOES NOT WORK for subroutine local arrays
- `default(present)` assumes ALL variables are on device
- Local stack-allocated arrays (e.g., `vflux`, `fqx`) are NOT on device
- Compiler generates `[if not already present]` auto-copy for bare `!$acc kernels` — this is the correct approach
- Only use explicit `present()` for arrays known to be on device (grid% fields in gpu_init)

### Diffusion GPU requires careful data management
- Enabling GPU code in diffusion without local array `!$acc data create()` causes 10-15× slowdown
- Each diffusion subroutine has multiple large local work arrays
- Must add `!$acc data create()` for locals in EACH subroutine before enabling `-acc`
- For now: compile diffusion CPU-only (FFLAGS_CPU without `-acc`)

### NVHPC compile "warnings" vs actual failures
- NVHPC `-S` (severe) errors in OpenACC may or may not produce .o files
- `-F` (fatal) errors always abort compilation
- Build scripts should use `rm -f *.o` before compile + check existence after
- The old rebuild script was masking failures by checking stale .o files

---

## Optimization Roadmap

### Next Steps (High Priority)
1. ~~Reduce kernel launch count~~ **DONE** — 55 fusions via fuse_kernels.py (Build 5)
2. ~~Test larger domain~~ **DONE** — 400×400 tested, 6.5× faster vs gfortran, VRAM 96%
3. ~~OpenMP for CPU code~~ **DONE** — OMP=4 gives 2.3-2.7× speedup (Build 5+OMP)
4. ~~Comprehensive diffusion GPU port~~ **DONE** — 18 subs, 106 arrays, 178 GPU kernels (Build 6)
5. **Port YSU PBL to GPU** — currently CPU-only, column-parallel physics
6. **Profile Build 6 with nsys** — identify remaining bottlenecks

### Medium Priority
6. Port Noah LSM to GPU (column physics)
7. Port RRTMG radiation to GPU (called infrequently)
8. Reduce VRAM usage — selective gpu_init to fit larger domains

### Low Priority
9. CUDA Fortran hotspots for advance_uv, advect_scalar
10. Multi-GPU support (domain decomposition for >500×500)

---

## Build System

### Compile Environment
```bash
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin:$PATH
FFLAGS="-O3 -fast -acc -gpu=cc120,fastmath -Mfree -Mrecursive -byteswapio -r4 -i4 -mp"
FFLAGS_CPU="-O3 -fast -Mfree -Mrecursive -byteswapio -r4 -i4 -mp"  # for diffusion
INCS="-I$WRF/frame -I$WRF/inc ... -DGPU_OPENACC -D_ACCEL"
```

### CRITICAL: Diffusion must be compiled CPU-only
```bash
# GPU modules:
nvfortran -o module_advect_em.o -c $FFLAGS $INCS module_advect_em.f90
# CPU-only modules:
nvfortran -o module_diffusion_em.o -c $FFLAGS_CPU $INCS module_diffusion_em.f90
```

### Key Files
- GPU build: `/home/drew/WRF_BUILD_GPU/`
- Patch scripts: `C:\Users\drew\aifs-90d\wrf_gpu/`
- Test case: `/home/drew/wrf_gpu_test/` (200×200×80, dx=3km, 1-min run)
- Runtime: `NVCOMPILER_ACC_CUDA_HEAPSIZE=4G NV_ACC_CUDA_STACKSIZE=65536`
