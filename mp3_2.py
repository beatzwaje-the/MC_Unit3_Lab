"""
Mini Project 3.2 - Cellular Capacity and Frequency-Reuse Planning
=================================================================
Mobile Communications - Unit 3 Lab

Builds a planning tool that computes cellular system capacity under
different reuse factors N and uses it to analyze the coverage-vs-capacity
trade-off, including cell splitting and sectoring as alternative capacity
levers.

Inputs:
  total_spectrum      - total spectrum allocation (MHz)
  channel_bandwidth   - per-channel bandwidth (MHz)
  coverage_area       - total service area (km^2)
  cell_radius         - hexagonal cell radius R (km)
  candidate_N         - reuse factors to evaluate

For each N the script computes:
  total_channels      = total_spectrum / channel_bandwidth
  channels_per_cell   = total_channels / N  (floor)
  num_cells           = floor(coverage_area / hex_cell_area(R))
  system_capacity     = channels_per_cell * num_cells
  Q                   = sqrt(3*N)
  S/I (approx)        = (1/6) * (D/R)^n = (1/6) * Q^n  with n=4 (urban)

Then it runs a cell-splitting scenario (R -> R/2 -> R/4) and a sectoring
illustration, and writes summary data and charts.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json, os

try:
    fm.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Configuration (all inputs documented in one place)
# ----------------------------------------------------------------------------
CONFIG = {
    'total_spectrum_MHz'   : 20.0,    # total spectrum allocation
    'channel_bw_MHz'       : 0.2,     # per-channel bandwidth (200 kHz)
    'coverage_area_km2'    : 100.0,   # dense-urban service area
    'cell_radius_km'       : 1.0,     # baseline cell radius (urban macro)
    'candidate_N'          : [1, 3, 4, 7, 9, 12],
    'path_loss_exp'        : 4,       # urban path-loss exponent (S/I calc)
    'sectors_per_cell'     : 3,       # for sectoring illustration
    'split_factors'        : [2, 4],  # R -> R/2 and R/4 in cell-splitting
}
print("=" * 78)
print("MINI PROJECT 3.2 - Cellular Capacity and Frequency-Reuse Planning")
print("=" * 78)
print("\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k:<25} : {v}")


def hex_cell_area(R):
    """Area of a hexagonal cell with circumradius R."""
    return 1.5 * np.sqrt(3) * R * R


def total_channels(cfg):
    return int(cfg['total_spectrum_MHz'] / cfg['channel_bw_MHz'])


def channels_per_cell(cfg, N):
    return total_channels(cfg) // N


def num_cells(cfg, R):
    return int(np.floor(cfg['coverage_area_km2'] / hex_cell_area(R)))


def system_capacity(cfg, N, R=None):
    if R is None:
        R = cfg['cell_radius_km']
    return channels_per_cell(cfg, N) * num_cells(cfg, R)


def Q_of(N):
    return np.sqrt(3 * N)


def SIR_approx(cfg, N):
    """Approximate downlink S/I for omnidirectional cells with 6 co-channel
    interferers at distance D = Q*R, path-loss exponent n.

        S/I = (1/6) * (D/R)^n = (1/6) * Q^n
    """
    Q = Q_of(N)
    n = cfg['path_loss_exp']
    return (1.0 / 6.0) * Q**n


def SIR_approx_sectored(cfg, N, sectors=3):
    """With sectoring, only the co-channel cells whose sectors point at the
    serving sector contribute interference. For 3-sector sites with 120-deg
    beams, the average number of dominant interferers drops from 6 to ~2,
    so this simplified model scales S/I by the sector count.
    """
    return SIR_approx(cfg, N) * sectors


# ----------------------------------------------------------------------------
# Step 2-4: per-N table
# ----------------------------------------------------------------------------
print("\n" + "-" * 78)
print("Per-reuse-factor summary (baseline cell radius R = "
      f"{CONFIG['cell_radius_km']} km)")
print("-" * 78)
hdr = f"{'N':>3} {'Q':>7} {'S/I(dB)':>9} {'ch/cell':>8} {'#cells':>7} {'cap':>8}"
print(hdr)
print("-" * len(hdr))
rows = []
for N in CONFIG['candidate_N']:
    Q = Q_of(N)
    sir = SIR_approx(CONFIG, N)
    cpc = channels_per_cell(CONFIG, N)
    nc = num_cells(CONFIG, CONFIG['cell_radius_km'])
    cap = cpc * nc
    rows.append({'N': N, 'Q': Q, 'SIR_dB': 10*np.log10(sir),
                 'ch_per_cell': cpc, 'num_cells': nc, 'capacity': cap})
    print(f"{N:>3} {Q:>7.3f} {10*np.log10(sir):>9.2f} "
          f"{cpc:>8} {nc:>7} {cap:>8}")

# ----------------------------------------------------------------------------
# Step 5: cell-splitting scenario for N = 7 (and N = 4 for contrast)
# ----------------------------------------------------------------------------
print("\n" + "-" * 78)
print("Cell-splitting scenario (capacity vs cell radius for N = 7)")
print("-" * 78)
print(f"{'R (km)':>8} {'#cells':>8} {'cap':>8} {'cap gain':>10}")
print("-" * 38)
R0 = CONFIG['cell_radius_km']
base_cap = system_capacity(CONFIG, 7, R=R0)
split_rows = []
for f in [1, 2, 4, 8]:
    R = R0 / f
    cap = system_capacity(CONFIG, 7, R=R)
    split_rows.append({'R': R, 'factor': f, 'num_cells': num_cells(CONFIG, R),
                       'capacity': cap, 'gain': cap / base_cap})
    print(f"{R:>8.3f} {num_cells(CONFIG, R):>8} {cap:>8} {cap/base_cap:>10.2f}x")

# ----------------------------------------------------------------------------
# Step 6: sectoring illustration (numerical)
# ----------------------------------------------------------------------------
print("\n" + "-" * 78)
print("Sectoring illustration (3 sectors per cell, 120-degree beams)")
print("-" * 78)
print(f"{'N':>3} {'Q':>7} {'SIR omni (dB)':>15} {'SIR 3-sec (dB)':>16} "
      f"{'ch/sec':>7} {'cap/sec':>8}")
print("-" * 65)
sector_rows = []
for N in CONFIG['candidate_N']:
    Q = Q_of(N)
    sir_o = SIR_approx(CONFIG, N)
    # With 3 sectors, only ~2 of the 6 co-channel interferers point at the
    # serving sector, so interference falls by ~3x (4.77 dB).
    sir_s = SIR_approx_sectored(CONFIG, N, CONFIG['sectors_per_cell'])
    ch_per_sector = total_channels(CONFIG) // (N * CONFIG['sectors_per_cell'])
    cap_per_sector = ch_per_sector * num_cells(CONFIG, CONFIG['cell_radius_km']) \
                     * CONFIG['sectors_per_cell']
    sector_rows.append({'N': N, 'Q': Q,
                        'SIR_omni_dB': 10*np.log10(sir_o),
                        'SIR_sect_dB': 10*np.log10(sir_s),
                        'ch_per_sector': ch_per_sector,
                        'cap_per_sector_total': cap_per_sector})
    print(f"{N:>3} {Q:>7.3f} {10*np.log10(sir_o):>15.2f} "
          f"{10*np.log10(sir_s):>16.2f} {ch_per_sector:>7} {cap_per_sector:>8}")

# ----------------------------------------------------------------------------
# Step 7: charts
# ----------------------------------------------------------------------------
# Chart 1: capacity vs N (bar) and SIR vs N (line on twin axis)
fig, ax1 = plt.subplots(figsize=(9, 6))
Ns = [r['N'] for r in rows]
caps = [r['capacity'] for r in rows]
sirs = [r['SIR_dB'] for r in rows]
bars = ax1.bar(Ns, caps, color='#1f77b4', alpha=0.7, label='System capacity')
ax1.set_xlabel('Reuse factor N')
ax1.set_ylabel('Total system capacity (channels)', color='#1f77b4')
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_xticks(Ns)
for b, c in zip(bars, caps):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height(),
             f'{c}', ha='center', va='bottom', fontsize=9)
ax2 = ax1.twinx()
ax2.plot(Ns, sirs, 'o-', color='#d62728', lw=2, label='S/I (dB)')
ax2.set_ylabel('Approx. S/I (dB)', color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')
plt.title('Capacity vs Reuse Factor (with S/I trade-off)')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mp3_2_capacity_vs_N.png', dpi=130)

# Chart 2: capacity vs Q (scatter, bubble size = num_cells)
fig, ax = plt.subplots(figsize=(9, 6))
for r in rows:
    ax.scatter(r['Q'], r['capacity'], s=r['num_cells']*5,
               alpha=0.6, color='#2ca02c', edgecolor='black')
    ax.annotate(f"N={r['N']}\n({r['num_cells']} cells)",
                (r['Q'], r['capacity']),
                textcoords='offset points', xytext=(8, 6), fontsize=9)
ax.set_xlabel('Co-channel reuse ratio Q = sqrt(3N)')
ax.set_ylabel('Total system capacity (channels)')
ax.set_title('Capacity vs Reuse Ratio Q (bubble size = number of cells)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mp3_2_capacity_vs_Q.png', dpi=130)

# Chart 3: cell-splitting - capacity vs R for N=7
fig, ax = plt.subplots(figsize=(9, 6))
Rs = [r['R'] for r in split_rows]
cs  = [r['capacity'] for r in split_rows]
gs  = [r['gain'] for r in split_rows]
ax.loglog(Rs, cs, 'o-', color='#9467bd', lw=2, base=2)
for R, c, g in zip(Rs, cs, gs):
    ax.annotate(f"R={R:.3f}km\ncap={c}\n({g:.1f}x)", (R, c),
                textcoords='offset points', xytext=(8, -10), fontsize=9)
ax.set_xlabel('Cell radius R (km, log scale)')
ax.set_ylabel('Total system capacity (channels)')
ax.set_title('Cell-splitting: capacity grows as 1/R^2 (N = 7)')
ax.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mp3_2_cell_splitting.png', dpi=130)

# Chart 4: SIR omni vs sectored
fig, ax = plt.subplots(figsize=(9, 6))
Ns2 = [r['N'] for r in sector_rows]
sir_o = [r['SIR_omni_dB'] for r in sector_rows]
sir_s = [r['SIR_sect_dB'] for r in sector_rows]
x = np.arange(len(Ns2))
w = 0.35
ax.bar(x - w/2, sir_o, w, label='Omni (6 interferers)', color='#1f77b4')
ax.bar(x + w/2, sir_s, w, label='3-sector (~2 interferers)', color='#d62728')
ax.set_xticks(x); ax.set_xticklabels(Ns2)
ax.set_xlabel('Reuse factor N')
ax.set_ylabel('Approx. S/I (dB)')
ax.set_title('Sectoring reduces co-channel interference (~4.8 dB gain)')
ax.legend()
ax.grid(alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mp3_2_sectoring.png', dpi=130)

# Save the summary data for further analysis.
with open(f'{OUT_DIR}/mp3_2_summary.json', 'w') as f:
    json.dump({'config': CONFIG, 'rows': rows, 'split_rows': split_rows,
               'sector_rows': sector_rows}, f, indent=2, default=str)

print("\nAll charts saved:")
for f in ['mp3_2_capacity_vs_N.png', 'mp3_2_capacity_vs_Q.png',
          'mp3_2_cell_splitting.png', 'mp3_2_sectoring.png']:
    print(f"  {f}")

if __name__ == "__main__":
    plt.show()
