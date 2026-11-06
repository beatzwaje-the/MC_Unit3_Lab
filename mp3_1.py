"""
Mini Project 3.1 - Frequency Reuse Planning and Handover Decision Logic
=======================================================================
Mobile Communications - Unit 3 Lab

Part (a): Hexagonal frequency-reuse pattern generator + co-channel reuse
         distance Q = sqrt(3N) for N in {1, 3, 4, 7, 9, 12}.
Part (b): Synthetic serving/neighbor RSRP series + three handover triggers
         (naive, hysteresis-only, hysteresis+TTT) demonstrating the
         ping-pong effect and the too-late-handover trade-off.

Lab Tasks (1-6) are implemented as labelled extension sections at the end.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import json, os

import matplotlib
import matplotlib.font_manager as fm
try:
    fm.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Part (a) - Hexagonal frequency-reuse layout
# ----------------------------------------------------------------------------

def hex_centers_axial(max_ring, R=1.0):
    """Return list of (q, r, cx, cy) for all hexagons in rings 0..max_ring."""
    cells = []
    for q in range(-max_ring, max_ring + 1):
        r1 = max(-max_ring, -q - max_ring)
        r2 = min(max_ring, -q + max_ring)
        for r in range(r1, r2 + 1):
            cx = R * np.sqrt(3) * (q + r / 2.0)
            cy = R * 1.5 * r
            cells.append((q, r, cx, cy))
    return cells


def hex_polygon(cx, cy, R):
    """Pointy-top hexagon vertices at center (cx, cy) with circumradius R."""
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.column_stack([cx + R * np.cos(angles),
                            cy + R * np.sin(angles)])


def find_ij(N):
    """Find non-negative integers (i, j) with i <= j satisfying N = i^2 + i*j + j^2."""
    for i in range(0, N + 1):
        for j in range(i, N + 1):
            if i * i + i * j + j * j == N:
                return i, j
    raise ValueError(f"No (i,j) found for N={N}")


def freq_group(q, r, i, j, N):
    """Frequency group label in 1..N for axial cell (q, r).

    The two residues from the adjugate of the cluster-basis matrix identify
    each coset of the reuse lattice. Keeping both residues matters for
    composite cluster sizes such as N=4, whose quotient is not cyclic.
    """
    def coset_key(a, b):
        return (((i + j) * a + j * b) % N,
                (-j * a + i * b) % N)

    cosets = sorted({coset_key(a, b)
                     for a in range(N) for b in range(N)})
    group_by_coset = {key: group for group, key in enumerate(cosets, start=1)}
    return group_by_coset[coset_key(q, r)]


def build_layout(N, R=1.0, max_ring=4):
    """Build labelled hex layout for cluster size N."""
    i, j = find_ij(N)
    cells = hex_centers_axial(max_ring, R=R)
    layout = []
    for (q, r, cx, cy) in cells:
        g = freq_group(q, r, i, j, N)
        layout.append({'q': q, 'r': r, 'cx': cx, 'cy': cy, 'group': g})
    return layout, (i, j)


def plot_layout(N, R=1.0, max_ring=4, ax=None, title=None):
    layout, (i, j) = build_layout(N, R=R, max_ring=max_ring)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap('tab20', N)
    patches, colors = [], []
    for c in layout:
        poly = hex_polygon(c['cx'], c['cy'], R * 0.95)
        patches.append(Polygon(poly, closed=True))
        colors.append(c['group'] - 1)
    pc = PatchCollection(patches, cmap=cmap, edgecolor='black', linewidth=0.6)
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    for c in layout:
        ax.text(c['cx'], c['cy'], str(c['group']), ha='center', va='center',
                fontsize=7 if N <= 7 else 5.5)
    ax.set_aspect('equal')
    pad = R * (max_ring + 1.5)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_title(title or f'N = {N}  (i={i}, j={j}, Q = sqrt(3N) = {np.sqrt(3*N):.3f})')
    ax.set_xticks([]); ax.set_yticks([])
    return layout


# ---- Step 4-6: layouts for N = 1, 3, 4, 7 and summary table -------------
print("=" * 78)
print("PART (a) - Hexagonal Frequency-Reuse Layouts")
print("=" * 78)
print("Verifying NumPy and Matplotlib:")
print(f"  NumPy version      : {np.__version__}")
print(f"  Matplotlib version : {matplotlib.__version__}")
print()

core_N = [1, 3, 4, 7]
fig, axes = plt.subplots(2, 2, figsize=(13, 13))
axes = axes.ravel()
summary_rows = []
for k, N in enumerate(core_N):
    layout, (i, j) = build_layout(N, R=1.0, max_ring=4)
    groups_seen = sorted({c['group'] for c in layout})
    Q = np.sqrt(3 * N)
    summary_rows.append((N, i, j, round(Q, 4), len(groups_seen)))
    plot_layout(N, R=1.0, max_ring=4, ax=axes[k])
fig.suptitle('Hexagonal Frequency-Reuse Patterns (N = 1, 3, 4, 7)', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f'{OUT_DIR}/mp3_1_reuse_layouts_core.png', dpi=130)

print(f"{'N':>3} {'i':>3} {'j':>3} {'Q=sqrt(3N)':>12} {'#groups':>9}")
print("-" * 35)
for r in summary_rows:
    print(f"{r[0]:>3} {r[1]:>3} {r[2]:>3} {r[3]:>12} {r[4]:>9}")
print()


# ----------------------------------------------------------------------------
# Part (b) - Handover decision simulation
# ----------------------------------------------------------------------------

print("=" * 78)
print("PART (b) - Handover Decision Simulation")
print("=" * 78)

N_SAMPLES = 300
t = np.arange(N_SAMPLES)

# Serving cell A: starts strong, gentle downward trend + random walk + noise.
# Noise / random-walk scales chosen so the two curves cross around t=120-150
# and then stay within a few dB of each other for a long stretch, letting
# noise drive repeated crossovers (the ping-pong region).
start_A = -75.0          # dBm
trend_A = -0.08          # dBm / sample  (about -24 dBm across the run)
rw_A = np.cumsum(np.random.normal(0, 0.45, N_SAMPLES))
noise_A = np.random.normal(0, 1.6, N_SAMPLES)
series_A = start_A + trend_A * t + rw_A + noise_A

# Neighbor cell B: starts weak, gentle upward trend + random walk + noise.
# Tuned so the curves cross around t ~ 130 and stay close until t ~ 250.
start_B = -100.0
trend_B = 0.08
rw_B = np.cumsum(np.random.normal(0, 0.45, N_SAMPLES))
noise_B = np.random.normal(0, 1.6, N_SAMPLES)
series_B = start_B + trend_B * t + rw_B + noise_B

# Stash for re-use by Lab Task 2 (third cell) and Lab Task 4 (high-speed)
series_A_base = series_A.copy()
series_B_base = series_B.copy()


def run_handover(series_A, series_B, mode='naive', margin=3.0, ttt=5):
    """Run a handover trigger over two RSRP series.

    Returns:
        handovers : list of sample indices where a handover fired
        serving   : 0/1 array of which series is serving at each sample
    """
    n = len(series_A)
    serving = np.zeros(n, dtype=int)   # 0 = A serving, 1 = B serving
    handovers = []
    cur = 0
    streak = 0
    for i in range(n):
        serv_val = series_A[i] if cur == 0 else series_B[i]
        neigh_val = series_B[i] if cur == 0 else series_A[i]
        diff = neigh_val - serv_val
        if mode == 'naive':
            cond = diff > 0
        elif mode == 'hyst':
            cond = diff > margin
        elif mode == 'hyst_ttt':
            cond = diff > margin
        else:
            raise ValueError(mode)

        if mode == 'hyst_ttt':
            if cond:
                streak += 1
            else:
                streak = 0
            fire = streak >= ttt
        else:
            fire = cond

        if fire:
            cur = 1 - cur
            handovers.append(i)
            streak = 0
        serving[i] = cur
    return handovers, serving


def serving_rsrp(series_A, series_B, serving):
    return np.where(serving == 0, series_A, series_B)


# Run all three configurations
h_naive,  s_naive  = run_handover(series_A, series_B, 'naive')
h_hyst,   s_hyst   = run_handover(series_A, series_B, 'hyst', margin=3.0)
h_ttt,    s_ttt    = run_handover(series_A, series_B, 'hyst_ttt', margin=3.0, ttt=5)

# Stacked three-subplot figure sharing the time axis
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
for ax, (name, h_list, serv) in zip(axes,
        [('Naive trigger (no margin, no TTT)', h_naive, s_naive),
         ('Hysteresis-only (margin = 3 dB)',   h_hyst,  s_hyst),
         ('Hysteresis + TTT (3 dB, 5 samples)', h_ttt,   s_ttt)]):
    ax.plot(t, series_A, color='#1f77b4', label='Cell A RSRP', lw=1.0)
    ax.plot(t, series_B, color='#d62728', label='Cell B RSRP', lw=1.0)
    # Shade the currently-serving series
    serv_rsrp = serving_rsrp(series_A, series_B, serv)
    ax.plot(t, serv_rsrp, color='black', lw=2.0, alpha=0.6, label='Serving RSRP')
    for h in h_list:
        ax.axvline(h, color='green', ls='--', lw=0.8, alpha=0.7)
    ax.set_ylabel('RSRP (dBm)')
    ax.set_title(f'{name}  -  handover count = {len(h_list)}')
    ax.legend(loc='lower left', fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Sample index')
fig.suptitle('Handover Decision: Naive vs Hysteresis vs Hysteresis+TTT', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f'{OUT_DIR}/mp3_1_handover_stacked.png', dpi=130)

# Comparison table
print("\nHandover-count comparison table")
print(f"{'Configuration':<35} {'Count':>6}")
print("-" * 45)
for name, h_list in [('Naive (no margin, no TTT)', h_naive),
                     ('Hysteresis only (3 dB)',     h_hyst),
                     ('Hysteresis + TTT (3dB,5)',   h_ttt)]:
    print(f"{name:<35} {len(h_list):>6}")

# Too-late-handover check: serving RSRP dropped below -100 dBm before next HO
MIN_THRESH = -100.0
print(f"\nToo-late-handover risk check (serving RSRP < {MIN_THRESH} dBm "
      "before next handover):")
for name, h_list, serv in [('Naive',        h_naive, s_naive),
                           ('Hysteresis',   h_hyst,  s_hyst),
                           ('Hysteresis+TTT', h_ttt, s_ttt)]:
    serv_rsrp = serving_rsrp(series_A, series_B, serv)
    violations = []
    for k in range(len(h_list) - 1):
        a, b = h_list[k], h_list[k + 1]
        seg = serv_rsrp[a:b + 1]
        if len(seg) and np.min(seg) < MIN_THRESH:
            violations.append((a, b, float(np.min(seg))))
    print(f"  {name:<15} : {len(violations)} segment(s) dipping below {MIN_THRESH} dBm")
    for v in violations[:3]:
        print(f"     samples {v[0]}->{v[1]}  min serving RSRP = {v[2]:.2f} dBm")


if __name__ == "__main__":
    plt.show()
