"""
Mini Project 3.1 - Lab Tasks (1-6)
==================================
Extends the core mp3_1.py script with the six lab tasks:

  Task 1: Extend the layout generator to N = 9 and N = 12.
  Task 2: Add a third synthetic neighbour-cell series and pick the strongest
          of three at each decision point.
  Task 3: Sweep hysteresis margin and TTT jointly; pick the combination
          minimising handover count while keeping min serving RSRP > -100 dBm
          (Mobility Robustness Optimization flavour).
  Task 4: High-speed scenario (larger trend slope and random-walk variance).
  Task 5: Asymmetric hysteresis offset (load-balancing bias toward one cell).
  Task 6: Sector each cell into three 120-degree sectors and discuss the
          interference implications.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import PatchCollection
import os

try:
    fm.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from mp3_1 import (hex_centers_axial, hex_polygon, find_ij, freq_group,
                   build_layout, plot_layout, run_handover,
                   serving_rsrp, series_A_base, series_B_base)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(7)


# ============================================================================
# Task 1 - Extend the layout to N = 9 and N = 12
# ============================================================================
print("=" * 78)
print("LAB TASK 1 - Extend layout to N = 9 and N = 12")
print("=" * 78)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, N in zip(axes, [9, 12]):
    i, j = find_ij(N)
    Q = np.sqrt(3 * N)
    layout, _ = build_layout(N, R=1.0, max_ring=4)
    plot_layout(N, R=1.0, max_ring=4, ax=ax,
                title=f'N = {N}  (i={i}, j={j}, Q = {Q:.3f})')
    groups = sorted({c['group'] for c in layout})
    print(f"  N={N:>2}  i={i} j={j}  Q={Q:.4f}  distinct groups in plot = {len(groups)}")
fig.suptitle('Lab Task 1 - Extended reuse patterns N = 9 and N = 12', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT_DIR}/mp3_1_task1_N9_N12.png', dpi=130)

# Discussion table comparing all tested N
print("\nCapacity-vs-interference trade-off across N = 1..12:")
print(f"{'N':>3} {'Q':>8} {'Q gain vs N=1':>15} {'reuse per area':>17}")
print("-" * 50)
for N in [1, 3, 4, 7, 9, 12]:
    Q = np.sqrt(3 * N)
    print(f"{N:>3} {Q:>8.3f} {Q/np.sqrt(3):>15.3f} {1/N:>17.3f}")
print("\nAs N grows beyond 7, Q keeps rising (9 -> 5.196, 12 -> 6.000) so co-channel")
print("interference keeps falling, but each frequency is reused only 1/N times per")
print("cell, so spectral efficiency per area drops proportionally. N=9 and N=12 are")
print("typical of interference-limited dense deployments with tight QoS targets.")


# ============================================================================
# Task 2 - Third neighbour cell, pick the strongest of three
# ============================================================================
print("\n" + "=" * 78)
print("LAB TASK 2 - Three-cell strongest-neighbour handover")
print("=" * 78)

N_SAMPLES = 300
t = np.arange(N_SAMPLES)

# Three cells with staggered trends so the strongest cell changes over time:
# A falls, B rises first, then C (further along the route) rises even more.
series_A = series_A_base.copy()              # falling
series_B = series_B_base.copy()              # rising
# Cell C starts even lower and rises later/stronger
start_C = -110.0
trend_C = 0.13
rw_C = np.cumsum(np.random.normal(0, 0.45, N_SAMPLES))
noise_C = np.random.normal(0, 1.6, N_SAMPLES)
series_C = start_C + trend_C * t + rw_C + noise_C

series = np.vstack([series_A, series_B, series_C])  # shape (3, N)


def run_handover_3cell(series, mode='hyst_ttt', margin=3.0, ttt=5):
    """Strongest-of-three handover.

    Serving cell index in {0,1,2}; at each sample the strongest *other* cell
    is the candidate neighbour. A handover fires when that neighbour beats
    the serving cell by `margin` for `ttt` consecutive samples (or instantly
    in 'naive' mode).
    """
    n = series.shape[1]
    serving = np.zeros(n, dtype=int)
    handovers = []
    cur = 0
    streak = 0
    for i in range(n):
        serv_val = series[cur, i]
        # neighbour = the strongest cell other than cur
        others = [k for k in range(3) if k != cur]
        neigh_idx = others[np.argmax([series[k, i] for k in others])]
        neigh_val = series[neigh_idx, i]
        diff = neigh_val - serv_val
        cond = diff > margin if mode != 'naive' else diff > 0
        if mode == 'hyst_ttt':
            if cond:
                streak += 1
            else:
                streak = 0
            fire = streak >= ttt
        else:
            fire = cond
        if fire:
            cur = neigh_idx
            handovers.append((i, neigh_idx))
            streak = 0
        serving[i] = cur
    return handovers, serving


h3_naive, s3_naive = run_handover_3cell(series, 'naive')
h3_ttt, s3_ttt = run_handover_3cell(series, 'hyst_ttt', margin=3.0, ttt=5)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for ax, (name, h_list, serv) in zip(axes,
        [('Naive (strongest of 3)',         h3_naive, s3_naive),
         ('Hysteresis + TTT (strongest of 3)', h3_ttt,  s3_ttt)]):
    ax.plot(t, series_A, color='#1f77b4', label='Cell A', lw=1.0)
    ax.plot(t, series_B, color='#d62728', label='Cell B', lw=1.0)
    ax.plot(t, series_C, color='#2ca02c', label='Cell C', lw=1.0)
    serv_rsrp = series[serv, np.arange(N_SAMPLES)]
    ax.plot(t, serv_rsrp, color='black', lw=2.2, alpha=0.7, label='Serving')
    for h, target in h_list:
        ax.axvline(h, color='purple', ls='--', lw=0.8, alpha=0.7)
    ax.set_ylabel('RSRP (dBm)')
    ax.set_title(f'{name}  -  handover count = {len(h_list)}')
    ax.legend(loc='lower left', fontsize=8, ncol=4)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Sample index')
fig.suptitle('Lab Task 2 - Strongest-of-three neighbour handover', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{OUT_DIR}/mp3_1_task2_three_cell.png', dpi=130)

print(f"  Naive (strongest-of-3)         : {len(h3_naive)} handovers")
print(f"  Hysteresis+TTT (strongest-of-3): {len(h3_ttt)} handovers")
print("  Handover log (hyst+TTT):")
for h, target in h3_ttt:
    print(f"    sample {h:>3} -> cell {chr(ord('A')+target)}")


# ============================================================================
# Task 3 - Joint sweep of hysteresis margin and TTT (MRO flavour)
# ============================================================================
print("\n" + "=" * 78)
print("LAB TASK 3 - Joint margin x TTT sweep (Mobility Robustness Optimization)")
print("=" * 78)

# Reuse the original two-cell series from mp3_1
series_A_sweep = series_A_base
series_B_sweep = series_B_base
MIN_THRESH = -100.0

margins = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
ttts    = [1, 3, 5, 8, 12]
results = np.zeros((len(margins), len(ttts)))      # handover count
min_rsrp = np.full((len(margins), len(ttts)), np.nan)  # min serving RSRP

for mi, m in enumerate(margins):
    for ti, T in enumerate(ttts):
        h, s = run_handover(series_A_sweep, series_B_sweep, 'hyst_ttt',
                            margin=m, ttt=T)
        results[mi, ti] = len(h)
        serv_rsrp = serving_rsrp(series_A_sweep, series_B_sweep, s)
        min_rsrp[mi, ti] = np.min(serv_rsrp)

# Heatmap of handover count, marking configs that violate the RSRP floor
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(results, aspect='auto', cmap='viridis_r', origin='lower')
ax.set_xticks(range(len(ttts)));    ax.set_xticklabels(ttts)
ax.set_yticks(range(len(margins))); ax.set_yticklabels(margins)
ax.set_xlabel('Time-to-trigger (samples)')
ax.set_ylabel('Hysteresis margin (dB)')
ax.set_title('Lab Task 3 - Handover count over (margin, TTT) grid')
for mi in range(len(margins)):
    for ti in range(len(ttts)):
        ok = min_rsrp[mi, ti] >= MIN_THRESH
        color = 'white' if ok else 'red'
        mark = '' if ok else ' !'
        ax.text(ti, mi, f"{int(results[mi,ti])}{mark}",
                ha='center', va='center', color=color, fontsize=9)
plt.colorbar(im, ax=ax, label='Handover count')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mp3_1_task3_sweep.png', dpi=130)

# Identify the min-count config that still keeps serving RSRP above threshold
best = None
for mi in range(len(margins)):
    for ti in range(len(ttts)):
        if min_rsrp[mi, ti] >= MIN_THRESH:
            if best is None or results[mi, ti] < best[2] or \
               (results[mi, ti] == best[2] and margins[mi] + ttts[ti] <
                margins[best[0]] + ttts[best[1]]):
                best = (mi, ti, results[mi, ti])
print(f"  Best (margin, TTT) keeping min serving RSRP >= {MIN_THRESH} dBm:")
if best:
    mi, ti, cnt = best
    print(f"    margin = {margins[mi]} dB, TTT = {ttts[ti]} samples, "
          f"handover count = {int(cnt)}, min serving RSRP = {min_rsrp[mi,ti]:.2f} dBm")
print("  (cells marked '!' in red on the heatmap violate the RSRP floor ->")
print("   too-late-handover risk from over-tuned margin/TTT.)")


# ============================================================================
# Task 4 - High-speed scenario (larger trend slope + random-walk variance)
# ============================================================================
print("\n" + "=" * 78)
print("LAB TASK 4 - High-speed scenario (highway / high-speed rail)")
print("=" * 78)

trend_fast = -0.15        # high-speed: device crosses cell quickly
rw_sigma_fast = 0.9       # large random-walk jitter
noise_fast = 5.5          # large measurement noise that keeps resetting TTT
rw_Af = np.cumsum(np.random.normal(0, rw_sigma_fast, N_SAMPLES))
rw_Bf = np.cumsum(np.random.normal(0, rw_sigma_fast, N_SAMPLES))
series_A_fast = -88.0 + trend_fast * t + rw_Af + np.random.normal(0, noise_fast, N_SAMPLES)
series_B_fast = -90.0 + (-trend_fast) * t + rw_Bf + np.random.normal(0, noise_fast, N_SAMPLES)

h_naive_f, s_naive_f = run_handover(series_A_fast, series_B_fast, 'naive')
h_hyst_f,  s_hyst_f  = run_handover(series_A_fast, series_B_fast, 'hyst',  margin=3.0)
h_ttt_f,   s_ttt_f   = run_handover(series_A_fast, series_B_fast, 'hyst_ttt', margin=3.0, ttt=5)

# Count too-late-handover events: serving RSRP < -100 before fire
def count_late(series_A, series_B, h_list, s_arr, thresh=-100.0):
    serv_rsrp = serving_rsrp(series_A, series_B, s_arr)
    late = 0
    last = 0
    for h in h_list:
        seg = serv_rsrp[last:h+1]
        if len(seg) and np.min(seg) < thresh:
            late += 1
        last = h
    # also count end-of-run dips if no further handover
    tail = serv_rsrp[last:]
    if len(tail) and np.min(tail) < thresh and len(h_list) > 0:
        late += 1
    return late

print("  High-speed handover counts (same parameters as the slow scenario):")
print(f"    Naive          : {len(h_naive_f)} handovers, "
      f"{count_late(series_A_fast, series_B_fast, h_naive_f, s_naive_f)} too-late")
print(f"    Hysteresis only: {len(h_hyst_f)} handovers, "
      f"{count_late(series_A_fast, series_B_fast, h_hyst_f, s_hyst_f)} too-late")
print(f"    Hysteresis+TTT : {len(h_ttt_f)} handovers, "
      f"{count_late(series_A_fast, series_B_fast, h_ttt_f, s_ttt_f)} too-late")
print("  At high speed the serving RSRP collapses faster than the (3 dB, 5 sample)")
print("  timer can react: the condition may briefly break under noise, resetting")
print("  TTT, so the handover fires only after the serving RSRP is already below")
print("  -100 dBm -> classic too-late-handover / RLF risk. MRO must shrink TTT")
print("  and/or margin in this regime.")

# Save a high-speed figure
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
for ax, (name, h_list, serv) in zip(axes,
        [('Naive (high speed)',         h_naive_f, s_naive_f),
         ('Hysteresis only (high speed)', h_hyst_f, s_hyst_f),
         ('Hysteresis + TTT (high speed)', h_ttt_f, s_ttt_f)]):
    ax.plot(t, series_A_fast, color='#1f77b4', label='Cell A', lw=1.0)
    ax.plot(t, series_B_fast, color='#d62728', label='Cell B', lw=1.0)
    serv_rsrp = serving_rsrp(series_A_fast, series_B_fast, serv)
    ax.plot(t, serv_rsrp, color='black', lw=2.2, alpha=0.7, label='Serving')
    ax.axhline(-100, color='orange', ls=':', lw=1.0, label='-100 dBm floor')
    for h in h_list:
        ax.axvline(h, color='green', ls='--', lw=0.8, alpha=0.7)
    ax.set_ylabel('RSRP (dBm)')
    ax.set_title(f'{name}  -  handover count = {len(h_list)}')
    ax.legend(loc='lower left', fontsize=8, ncol=4)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Sample index')
fig.suptitle('Lab Task 4 - High-speed scenario (faster trend + larger noise)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f'{OUT_DIR}/mp3_1_task4_high_speed.png', dpi=130)


# ============================================================================
# Task 5 - Asymmetric hysteresis offset (load-balancing bias)
# ============================================================================
print("\n" + "=" * 78)
print("LAB TASK 5 - Asymmetric offset (load-balancing bias toward cell B)")
print("=" * 78)

def run_handover_asymmetric(series_A, series_B, margin_AB=3.0, margin_BA=8.0,
                            ttt=5):
    """Handover A->B uses margin_AB; B->A uses margin_BA (larger -> biased to
    keep the device on B once it gets there, e.g. to offload A)."""
    n = len(series_A)
    serving = np.zeros(n, dtype=int)
    handovers = []
    cur = 0
    streak = 0
    for i in range(n):
        serv_val = series_A[i] if cur == 0 else series_B[i]
        neigh_val = series_B[i] if cur == 0 else series_A[i]
        margin = margin_AB if cur == 0 else margin_BA
        cond = (neigh_val - serv_val) > margin
        if cond:
            streak += 1
        else:
            streak = 0
        if streak >= ttt:
            cur = 1 - cur
            handovers.append(i)
            streak = 0
        serving[i] = cur
    return handovers, serving

# Run on the high-noise baseline series with hysteresis-only (no TTT) so the
# asymmetric margin actually has back-and-forth handovers to act on.
h_sym, s_sym  = run_handover_asymmetric(series_A_base, series_B_base,
                                        margin_AB=3.0, margin_BA=3.0, ttt=1)
h_asym, s_asym = run_handover_asymmetric(series_A_base, series_B_base,
                                        margin_AB=3.0, margin_BA=8.0, ttt=1)

# Count direction of each handover
def handover_directions(h_list, s_arr):
    dirs = []
    for h in h_list:
        if h == 0:
            continue
        prev = s_arr[h-1] if h > 0 else 0
        cur = s_arr[h]
        if prev == 0 and cur == 1:
            dirs.append('A->B')
        elif prev == 1 and cur == 0:
            dirs.append('B->A')
    return dirs

d_sym = handover_directions(h_sym, s_sym)
d_asym = handover_directions(h_asym, s_asym)
ab_sym = d_sym.count('A->B'); ba_sym = d_sym.count('B->A')
ab_asym = d_asym.count('A->B'); ba_asym = d_asym.count('B->A')

print(f"  Symmetric  (A->B = 3 dB, B->A = 3 dB) : {len(h_sym)} handovers "
      f"({ab_sym} A->B, {ba_sym} B->A)")
print(f"  Asymmetric (A->B = 3 dB, B->A = 8 dB) : {len(h_asym)} handovers "
      f"({ab_asym} A->B, {ba_asym} B->A)")
print("  The larger B->A margin suppresses handovers back to A: once the device")
print("  is on B, only a >8 dB sustained advantage of A can pull it back, so the")
print("  B->A count drops sharply. This is exactly the load-balancing lever from")
print("  Section 3.1: bias the threshold to keep users on the less-loaded cell.")


# ============================================================================
# Task 6 - Sector each cell into three 120-degree sectors
# ============================================================================
print("\n" + "=" * 78)
print("LAB TASK 6 - Three-sector (120 degree) sites")
print("=" * 78)

def plot_sectored_layout(N, R=1.0, max_ring=3, ax=None, title=None):
    layout, (i, j) = build_layout(N, R=R, max_ring=max_ring)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap('tab20', N)
    sector_colors = ['#1f77b4', '#d62728', '#2ca02c']  # per-sector shading
    for c in layout:
        cx, cy, g = c['cx'], c['cy'], c['group']
        base_color = cmap(g - 1)
        # Three 120-degree wedges, slightly inset, with different alpha per sector
        for s, ang0 in enumerate([0, 120, 240]):
            w = Wedge((cx, cy), R * 0.95, ang0, ang0 + 120, width=R * 0.95,
                      facecolor=base_color, alpha=0.45 + 0.18 * s,
                      edgecolor='black', linewidth=0.5)
            ax.add_patch(w)
        ax.text(cx, cy, str(g), ha='center', va='center', fontsize=7)
    ax.set_aspect('equal')
    pad = R * (max_ring + 1.5)
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
    ax.set_title(title or f'N = {N}  -  three 120-degree sectors per cell')
    ax.set_xticks([]); ax.set_yticks([])
    return layout

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
plot_sectored_layout(7, R=1.0, max_ring=3, ax=axes[0],
                     title='N = 7 sectored (3 x 120 degrees per cell)')
# Plot the unsectored N=7 layout side-by-side for comparison
plot_layout(7, R=1.0, max_ring=3, ax=axes[1],
            title='N = 7 unsectored (omnidirectional)')
fig.suptitle('Lab Task 6 - Sectoring: 3 x 120-degree sites vs omnidirectional',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT_DIR}/mp3_1_task6_sectoring.png', dpi=130)

print("  Sectoring replaces each omni cell with 3 directional 120-degree beams.")
print("  On a given frequency group, only the sectors pointing at each other can")
print("  interfere, so the effective co-channel interferer count drops by ~2/3.")
print("  This lets the operator use a tighter reuse (smaller N) on each sector")
print("  without a proportional interference increase - the canonical")
print("  capacity-vs-interference lever alongside cell splitting.")

if __name__ == "__main__":
    plt.show()
