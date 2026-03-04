#!/usr/bin/env python3
"""
Patch Revised MM5 Surface Layer (sf_sfclayrev) for OpenACC GPU execution.

Modifies two files:
1. phys/physics_mmm/sf_sfclayrev.F90 (kernel) - verify/add !$acc directives
2. phys/module_sf_sfclayrev.F (wrapper) - add full private() clause + data region

ROOT CAUSE OF HEAP THRASHING:
  The wrapper's !$acc parallel loop gang private() clause was missing 30+ local
  arrays. Non-private automatic arrays in a gang-parallel region get allocated on
  GPU global heap, causing lock contention and thrashing.

  Additionally, sf_sfclayrev_run contains 26 automatic arrays dimensioned (its:ite).
  With !$acc routine seq, these go on the per-thread stack. NVHPC's default GPU
  stack is 1024 bytes, but 26 arrays x 200 elements x 8 bytes = ~41 KB.
  Must set NV_ACC_CUDA_STACKSIZE=65536 at runtime.

ANALYSIS:
  SAVE variables (module-scope, read-only after init):
    psim_stab(0:1000), psim_unstab(0:1000), psih_stab(0:1000), psih_unstab(0:1000)
    -> Already handled: !$acc declare copyin + !$acc update device in init

  Helper functions called from sf_sfclayrev_run (all already have !$acc routine seq):
    zolri, zolri2, psim_stable, psih_stable, psim_unstable, psih_unstable,
    psim_stable_full, psih_stable_full, psim_unstable_full, psih_unstable_full,
    depth_dependent_z0

  WRITE/PRINT statements:
    Line ~845: write(*,1001) -- ALREADY COMMENTED OUT
    Format 1001 is harmless (never executed)

  Automatic arrays in sf_sfclayrev_run (26 total, all dimensioned its:ite):
    za, thvx, zqkl, zqklp1, thx, qx, psih2, psim2, psih10, psim10,
    denomq, denomq2, denomt2, wspdi, gz2oz0, gz10oz0, rhox, govrth,
    tgdsa, scr3, scr4, thgb, psfc, pq, pq2, pq10
"""
import re
import os
import sys

WRF_DIR = os.environ.get("WRF_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
if not WRF_DIR:
    print("ERROR: Set WRF_DIR environment variable or pass WRF directory as argument")
    sys.exit(1)


def patch_kernel():
    """Verify and fix sf_sfclayrev.F90 kernel directives."""
    path = f'{WRF_DIR}/phys/physics_mmm/sf_sfclayrev.F90'
    with open(path, 'r') as f:
        content = f.read()

    original = content
    changes = []

    # 1. Check !$acc declare copyin for SAVE lookup tables
    if '!$acc declare copyin(psim_stab' not in content:
        # Add after the save line
        save_line = 'real(kind=kind_phys),dimension(0:1000 ),save:: psim_stab,psim_unstab,psih_stab,psih_unstab'
        if save_line in content:
            content = content.replace(
                save_line,
                save_line + '\n\n !$acc declare copyin(psim_stab, psim_unstab, psih_stab, psih_unstab)',
                1
            )
            changes.append('Added !$acc declare copyin for SAVE lookup tables')

    # 2. Check !$acc update device in init
    if '!$acc update device(psim_stab' not in content:
        end_init = ' end subroutine sf_sfclayrev_init'
        if end_init in content:
            content = content.replace(
                end_init,
                '\n !$acc update device(psim_stab, psim_unstab, psih_stab, psih_unstab)\n\n' + end_init,
                1
            )
            changes.append('Added !$acc update device in sf_sfclayrev_init')

    # 3. Check !$acc routine seq on sf_sfclayrev_run
    # The declaration spans ~14 lines; the !$acc routine seq should be right after closing )
    run_close = "                            )\n!$acc routine seq"
    if '!$acc routine seq' not in content.split('subroutine sf_sfclayrev_run')[1].split('end subroutine sf_sfclayrev_run')[0][:200]:
        # Try to add it
        run_paren = "                            )\n!================================================="
        if run_paren in content:
            content = content.replace(
                run_paren,
                "                            )\n!$acc routine seq\n!=================================================",
                1
            )
            changes.append('Added !$acc routine seq to sf_sfclayrev_run')

    # 4. Check all helper functions for !$acc routine seq
    helper_functions = [
        'zolri', 'zolri2',
        'psim_stable_full', 'psih_stable_full',
        'psim_unstable_full', 'psih_unstable_full',
        'psim_stable', 'psih_stable',
        'psim_unstable', 'psih_unstable',
        'depth_dependent_z0',
    ]

    lines = content.split('\n')
    insertions = []  # (line_index, text_to_insert)

    for fname in helper_functions:
        for idx, line in enumerate(lines):
            stripped = line.strip().lower()
            # Match the function declaration (not end function)
            if (f'function {fname}(' in stripped or f'function {fname} (' in stripped) \
               and not stripped.startswith('end') and not stripped.startswith('!'):
                # Find end of declaration (closing paren without continuation)
                end_idx = idx
                for scan in range(idx, min(len(lines), idx + 10)):
                    if ')' in lines[scan]:
                        # Check if this line has continuation
                        if '&' not in lines[scan].rstrip():
                            end_idx = scan
                            break
                # Check if !$acc routine seq is nearby
                has_acc = False
                for check in range(max(0, idx - 1), min(len(lines), end_idx + 4)):
                    if '!$acc routine seq' in lines[check].lower():
                        has_acc = True
                        break
                if not has_acc:
                    insertions.append((end_idx + 1, f' !$acc routine seq'))
                    changes.append(f'Added !$acc routine seq to {fname}')
                break

    # Apply insertions in reverse order to preserve indices
    for ins_idx, ins_text in sorted(insertions, reverse=True):
        lines.insert(ins_idx, ins_text)

    if insertions:
        content = '\n'.join(lines)

    # Write if changed
    if content != original:
        with open(path, 'w') as f:
            f.write(content)
        print(f'  Patched {path}:')
        for c in changes:
            print(f'    - {c}')
    else:
        print(f'  {path}: all directives already present (no changes)')

    # Report status
    # Count !$acc routine seq occurrences
    acc_count = content.lower().count('!$acc routine seq')
    print(f'    Total !$acc routine seq directives: {acc_count}')
    print(f'    Expected: 12 (1 main + 11 helpers)')

    return content


def patch_wrapper():
    """Add/fix OpenACC directives in module_sf_sfclayrev.F wrapper."""
    path = f'{WRF_DIR}/phys/module_sf_sfclayrev.F'
    with open(path, 'r') as f:
        content = f.read()

    original = content
    changes = []

    # 1. Add use statement import + routine declarations at module level
    old_use = ' use sf_sfclayrev,only: sf_sfclayrev_run'
    if '!$acc routine(sf_sfclayrev_run) seq' not in content:
        if old_use in content:
            new_use = (
                old_use + '\n'
                ' !$acc routine(sf_sfclayrev_run) seq\n'
                ' !$acc routine(sf_sfclayrev_pre_run) seq'
            )
            content = content.replace(old_use, new_use, 1)
            changes.append('Added !$acc routine declarations at module level')

    # 2. Add !$acc routine seq inside sf_sfclayrev_pre_run body
    pre_run_header = (
        ' subroutine sf_sfclayrev_pre_run(dz2d,u2d,v2d,qv2d,p2d,t2d,dz1d,u1d,v1d,qv1d,p1d,t1d, &\n'
        '                                 its,ite,kts,kte,errmsg,errflg)\n'
        '!================================================='
    )
    if pre_run_header in content:
        # Check if !$acc routine seq follows
        after_header = content.split(pre_run_header)[1][:200]
        if '!$acc routine seq' not in after_header:
            content = content.replace(
                pre_run_header,
                pre_run_header.rstrip('=') + '========================================\n !$acc routine seq',
                1
            )
            # Fix: that doubled the separator. Let me do it differently
            content = original  # reset
            changes = []

    # Simpler approach for pre_run: find the exact separator after the declaration
    pre_run_decl_end = "its,ite,kts,kte,errmsg,errflg)\n!="
    if pre_run_decl_end in content and '!$acc routine seq' not in \
       content.split('subroutine sf_sfclayrev_pre_run')[1][:300]:
        content = content.replace(
            "its,ite,kts,kte,errmsg,errflg)\n!=",
            "its,ite,kts,kte,errmsg,errflg)\n !$acc routine seq\n!=",
            1
        )
        changes.append('Added !$acc routine seq inside sf_sfclayrev_pre_run')

    # Re-add module-level declarations if needed (after the simpler approach)
    if '!$acc routine(sf_sfclayrev_run) seq' not in content:
        if old_use in content:
            new_use = (
                old_use + '\n'
                ' !$acc routine(sf_sfclayrev_run) seq\n'
                ' !$acc routine(sf_sfclayrev_pre_run) seq'
            )
            content = content.replace(old_use, new_use, 1)
            changes.append('Added !$acc routine declarations at module level')

    # 3. Add !$acc data present() region and parallel loop with full private()
    # Find the j-loop and add directives before it
    j_loop_marker = ' do j = jts,jte\n'

    # Check if !$acc parallel loop already exists
    if '!$acc parallel loop' in content:
        # Already has directives -- check if private() is complete
        # Extract current private list
        priv_match = re.search(r'private\(([^)]+)\)', content, re.IGNORECASE | re.DOTALL)
        if priv_match:
            current_private = priv_match.group(1)
            current_vars = set(v.strip() for v in
                              current_private.replace('&\n', ' ').replace('\n', ' ').replace('!$acc', '').split(',')
                              if v.strip())
            needed_vars = _get_all_private_vars()
            missing = [v for v in needed_vars if v not in current_vars]
            if missing:
                # Replace entire parallel loop + private block
                # Find from !$acc parallel to the do j line
                old_block = re.search(
                    r'(!\$acc\s+parallel\s+loop[^\n]*\n(?:!\$acc&[^\n]*\n)*)(\s*do j)',
                    content, re.IGNORECASE
                )
                if old_block:
                    new_parallel = _build_parallel_directive()
                    content = content[:old_block.start()] + new_parallel + old_block.group(2) + content[old_block.end():]
                    changes.append(f'Updated private() clause: added {len(missing)} missing arrays')
    else:
        # No !$acc directives yet -- add them fresh
        if j_loop_marker in content:
            data_block = _build_data_directive()
            parallel_block = _build_parallel_directive()
            content = content.replace(
                j_loop_marker,
                data_block + parallel_block + j_loop_marker,
                1
            )
            changes.append('Added !$acc data present() region')
            changes.append('Added !$acc parallel loop gang with full private() clause')

            # Add end directives
            # Find the enddo + end subroutine sfclayrev
            end_pattern = ' enddo\n\n end subroutine sfclayrev'
            if end_pattern not in content:
                end_pattern = ' enddo\n end subroutine sfclayrev'
            if end_pattern in content:
                content = content.replace(
                    end_pattern,
                    ' enddo\n!$acc end parallel loop\n!$acc end data\n\n end subroutine sfclayrev',
                    1
                )
                changes.append('Added !$acc end parallel loop + end data')

    # Write if changed
    if content != original:
        with open(path, 'w') as f:
            f.write(content)
        print(f'  Patched {path}:')
        for c in changes:
            print(f'    - {c}')
    else:
        print(f'  {path}: all directives already present (no changes)')

    return content


def _get_all_private_vars():
    """Return list of all local arrays that must be private per gang."""
    return [
        # 2D (i,k) work arrays
        'dz_hv', 'u_hv', 'v_hv', 'qv_hv', 'p_hv', 't_hv',
        # 1D work arrays (output of pre_run)
        'u1d', 'v1d', 't1d', 'qv1d', 'p1d', 'dz1d',
        # 1D input _hv
        'dx_hv', 'mavail_hv', 'pblh_hv', 'psfc_hv', 'tsk_hv',
        'xland_hv', 'water_depth_hv', 'lakemask_hv',
        # 1D output _hv
        'lh_hv', 'u10_hv', 'v10_hv', 'th2_hv', 't2_hv', 'q2_hv',
        # 1D optional output _hv
        'ck_hv', 'cka_hv', 'cd_hv', 'cda_hv',
        # 1D inout _hv
        'regime_hv', 'hfx_hv', 'qfx_hv', 'qsfc_hv', 'mol_hv', 'rmol_hv',
        'gz1oz0_hv', 'wspd_hv', 'br_hv', 'psim_hv', 'psih_hv', 'fm_hv',
        'fh_hv', 'znt_hv', 'zol_hv', 'ust_hv', 'cpm_hv', 'chs2_hv',
        'cqs2_hv', 'chs_hv', 'flhc_hv', 'flqc_hv', 'qgh_hv',
        # 1D optional inout _hv
        'ustm_hv',
    ]


def _build_data_directive():
    """Build the !$acc data present() block for the wrapper."""
    return (
        '!$acc data present(t3d, qv3d, p3d, dz8w, psfc,                      &\n'
        '!$acc&  znt, ust, pblh, mavail, zol, mol, regime, psim, psih, fm, fh, &\n'
        '!$acc&  xland, hfx, qfx, lh, tsk, flhc, flqc, qsfc, rmol,            &\n'
        '!$acc&  u10, v10, th2, t2, q2, wspd, br, dx,                          &\n'
        '!$acc&  lakemask, water_depth)                                         &\n'
        '!$acc& copyin(u3d, v3d)                                                &\n'
        '!$acc& copy(qgh, gz1oz0, chs, chs2, cqs2, cpm)                       &\n'
        '!$acc& create(ustm, ck, cka, cd, cda)\n'
    )


def _build_parallel_directive():
    """Build the !$acc parallel loop gang private() block."""
    all_vars = _get_all_private_vars()
    # Format: 5 vars per continuation line
    lines = []
    for i in range(0, len(all_vars), 5):
        chunk = all_vars[i:i+5]
        lines.append(', '.join(chunk))

    result = '!$acc parallel loop gang                                                &\n'
    for idx, line in enumerate(lines):
        if idx == 0:
            result += f'!$acc&  private({line}'
        else:
            result += f',  &\n!$acc&          {line}'
    result += ')\n'
    return result


def main():
    print('=' * 72)
    print('  Patching Surface Layer (sf_sfclayrev) for OpenACC GPU')
    print('=' * 72)
    print()

    # Step 1: Patch kernel
    print('--- Step 1: Verify/patch sf_sfclayrev.F90 kernel ---')
    patch_kernel()
    print()

    # Step 2: Patch wrapper
    print('--- Step 2: Patch module_sf_sfclayrev.F wrapper ---')
    patch_wrapper()
    print()

    # Summary
    print('=' * 72)
    print('  Summary & Next Steps')
    print('=' * 72)
    print()
    print('  SAVE variables (read-only lookup tables, already on GPU):')
    print('    psim_stab(0:1000), psim_unstab(0:1000)')
    print('    psih_stab(0:1000), psih_unstab(0:1000)')
    print()
    print('  CRITICAL: Set GPU stack size before running WRF:')
    print('    export NV_ACC_CUDA_STACKSIZE=65536')
    print('    (26 auto arrays in sf_sfclayrev_run need ~41 KB per thread)')
    print()
    print('  Recompile:')
    print(f'    cd {WRF_DIR}')
    print('    touch phys/physics_mmm/sf_sfclayrev.F90')
    print('    touch phys/module_sf_sfclayrev.F')
    print('    ./compile em_real 2>&1 | tail -20')


if __name__ == '__main__':
    main()
