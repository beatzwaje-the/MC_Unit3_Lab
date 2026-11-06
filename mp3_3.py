"""
Mini Project 3.3 - Modeling Device Connection and Handover as a Finite-State
Machine
=============================================================================
Mobile Communications - Unit 3 Lab

States
  Idle, RA_Preamble, RA_RAR, RA_ConnReq, RA_Contention, Connected,
  Handover
  (RLF is a transition back to Idle, not a state.)

RACH (Section 3.7) is represented as the four logical steps:
  preamble -> Random Access Response (with timing advance) ->
  connection request -> contention resolution.

Two interchangeable handover policies (consistent with mp3_1):
  - naive:    fire when neighbour RSRP > serving RSRP
  - hyst_ttt: fire when neighbour RSRP > serving RSRP + margin for TTT
              consecutive samples

RLF: serving RSRP < RLF_THRESH for RLF_HOLD consecutive samples while in
Connected or Handover -> return to Idle (the too-late-handover failure mode).

Scenario: a device moves along a straight route, passing cells A, B, C in
order, with realistic overlapping coverage. Each policy is run on the same
scenario and the transition log is post-processed to count handovers, RLFs
and ping-pong sequences.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import json, os
from collections import deque

try:
    fm.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(11)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CFG = {
    'N_samples'     : 360,
    'tx_power_dBm'  : 23.0,        # per-cell Tx power
    'PL_exp'        : 3.5,         # path-loss exponent
    'PL_ref_dB'     : 38.0,        # reference loss at 1 km
    'noise_sigma'   : 1.5,         # RSRP measurement noise (dB)
    'RLF_thresh'    : -110.0,      # serving RSRP floor for RLF (dBm)
    'RLF_hold'      : 5,           # consecutive samples below floor
    'attach_thresh' : -100.0,      # any cell above this triggers attach
    'HO_complete'   : 3,           # samples to complete a handover
    'margin'        : 3.0,         # hysteresis margin (dB) for hyst_ttt
    'TTT'           : 5,           # time-to-trigger (samples) for hyst_ttt
    'pp_window'     : 20,          # ping-pong detection window (samples)
}
print("=" * 78)
print("MINI PROJECT 3.3 - Device Connection / Handover FSM")
print("=" * 78)
print("\nConfiguration:")
for k, v in CFG.items():
    print(f"  {k:<15} : {v}")


# ----------------------------------------------------------------------------
# Scenario: device moves along a line, passing cells A (x=0), B (x=5), C (x=10)
# ----------------------------------------------------------------------------
N = CFG['N_samples']
t = np.arange(N)
x_dev = 0.03 * t                       # device position (km), 0 -> 10.8 km
cell_pos = {'A': 0.0, 'B': 5.0, 'C': 10.0}

def rsrp(distance_km, sigma=CFG['noise_sigma']):
    """RSRP from path-loss model + Gaussian measurement noise."""
    d = max(distance_km, 0.05)
    pl = CFG['PL_ref_dB'] + 10 * CFG['PL_exp'] * np.log10(d)
    return CFG['tx_power_dBm'] - pl + np.random.normal(0, sigma)

# Pre-generate the three cell RSRP series
series = {}
for cell, pos in cell_pos.items():
    rsrps = np.array([rsrp(abs(x - pos)) for x in x_dev])
    series[cell] = rsrps

cells = ['A', 'B', 'C']

# Plot the scripted scenario
fig, ax = plt.subplots(figsize=(11, 5))
for c in cells:
    ax.plot(t, series[c], label=f'Cell {c}', lw=1.0)
ax.axhline(CFG['RLF_thresh'], color='red', ls=':', lw=1.2,
           label=f"RLF threshold ({CFG['RLF_thresh']} dBm)")
ax.axhline(CFG['attach_thresh'], color='green', ls=':', lw=1.2,
           label=f"Attach threshold ({CFG['attach_thresh']} dBm)")
ax.set_xlabel('Sample index')
ax.set_ylabel('RSRP (dBm)')
ax.set_title('Scripted three-cell mobility scenario (A -> B -> C)')
ax.legend(loc='lower center', ncol=5, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/mp3_3_scenario.png', dpi=130)


# ----------------------------------------------------------------------------
# State machine
# ----------------------------------------------------------------------------
# State names
S_IDLE   = 'Idle'
S_PREAM  = 'RA_Preamble'
S_RAR    = 'RA_RAR'
S_CONNRQ = 'RA_ConnReq'
S_CONT   = 'RA_Contention'
S_CONN   = 'Connected'
S_HO     = 'Handover'

# Allowed transitions (used for the diagram)
ALLOWED = {
    S_IDLE:   [S_PREAM],
    S_PREAM:  [S_RAR,    S_IDLE],          # RAR success / preamble failure
    S_RAR:    [S_CONNRQ, S_IDLE],          # conn req sent / RAR timeout
    S_CONNRQ: [S_CONT,   S_IDLE],          # contention phase / failure
    S_CONT:   [S_CONN,   S_IDLE],          # resolution OK / collision
    S_CONN:   [S_HO,     S_IDLE],          # handover triggered / RLF
    S_HO:     [S_CONN,   S_IDLE],          # HO complete / RLF mid-HO
}

# Pretty transition triggers (for the diagram / table)
TRANSITION_TRIGGERS = [
    (S_IDLE,      S_PREAM,  'Attach request (any cell RSRP > attach_thresh)'),
    (S_PREAM,     S_RAR,    'eNB detects preamble, sends RAR with timing advance'),
    (S_PARAM,     S_IDLE,   'Preamble failure (max retries)') if False else
    (S_PREAM,     S_IDLE,   'Preamble failure (max retries exceeded)'),
    (S_RAR,       S_CONNRQ, 'Timing advance applied, conn request sent'),
    (S_RAR,       S_IDLE,   'RAR timeout'),
    (S_CONNRQ,    S_CONT,   'Waiting for contention resolution'),
    (S_CONNRQ,    S_IDLE,   'Connection request failure'),
    (S_CONT,      S_CONN,   'Contention resolved, UE attached'),
    (S_CONT,      S_IDLE,   'Contention collision, backoff'),
    (S_CONN,      S_HO,     'Handover policy fires (naive or hyst+TTT)'),
    (S_CONN,      S_IDLE,   'RLF: serving RSRP < RLF_thresh for RLF_hold samples'),
    (S_HO,        S_CONN,   'Handover completes (target cell sync OK)'),
    (S_HO,        S_IDLE,   'RLF mid-handover'),
]


def strongest_other(series, cur_cell, i):
    """Return (cell_name, rsrp) of strongest cell other than cur_cell at i."""
    others = [c for c in cells if c != cur_cell]
    best = max(others, key=lambda c: series[c][i])
    return best, series[best][i]


def run_fsm(series, policy='naive', cfg=CFG, verbose=False):
    """Run the FSM over the scripted scenario under the given policy.

    Returns the transition log as a list of dicts:
        {t, from, to, cell, trigger}
    """
    log = []
    state = S_IDLE
    cur_cell = None
    ho_streak = 0                       # TTT counter for hyst_ttt
    rlf_streak = 0                      # consecutive samples below RLF floor
    ho_progress = 0                     # samples spent in Handover
    ho_target = None                    # cell being handed over to
    ho_origin = None                    # cell being handed over from
    recent_left = deque(maxlen=cfg['pp_window'])  # for ping-pong detection

    for i in range(N):
        # Determine serving RSRP at this sample (if any)
        if state in (S_CONN, S_HO) and cur_cell is not None:
            serv_rsrp = series[cur_cell][i]
        else:
            serv_rsrp = None

        # ---- RLF check ----------------------------------------------
        if state in (S_CONN, S_HO):
            if serv_rsrp < cfg['RLF_thresh']:
                rlf_streak += 1
            else:
                rlf_streak = 0
            if rlf_streak >= cfg['RLF_hold']:
                log.append({'t': i, 'from': state, 'to': S_IDLE,
                            'cell': cur_cell, 'trigger': 'RLF'})
                state = S_IDLE
                cur_cell = None
                rlf_streak = 0
                ho_streak = 0
                ho_progress = 0
                ho_target = None
                continue

        # ---- State-specific transitions -----------------------------
        if state == S_IDLE:
            # Attach if any cell above attach threshold
            best = max(cells, key=lambda c: series[c][i])
            if series[best][i] > cfg['attach_thresh']:
                cur_cell = best
                log.append({'t': i, 'from': S_IDLE, 'to': S_PREAM,
                            'cell': best, 'trigger': 'Attach request'})
                state = S_PREAM
        elif state == S_PREAM:
            log.append({'t': i, 'from': S_PREAM, 'to': S_RAR,
                        'cell': cur_cell, 'trigger': 'Preamble ACK, RAR rx (TA)'})
            state = S_RAR
        elif state == S_RAR:
            log.append({'t': i, 'from': S_RAR, 'to': S_CONNRQ,
                        'cell': cur_cell, 'trigger': 'TA applied, conn req sent'})
            state = S_CONNRQ
        elif state == S_CONNRQ:
            log.append({'t': i, 'from': S_CONNRQ, 'to': S_CONT,
                        'cell': cur_cell, 'trigger': 'Waiting for contention'})
            state = S_CONT
        elif state == S_CONT:
            log.append({'t': i, 'from': S_CONT, 'to': S_CONN,
                        'cell': cur_cell, 'trigger': 'Contention resolved, attached'})
            state = S_CONN
        elif state == S_CONN:
            # Apply handover policy
            neigh, neigh_rsrp = strongest_other(series, cur_cell, i)
            diff = neigh_rsrp - serv_rsrp
            if policy == 'naive':
                fire = diff > 0
            elif policy == 'hyst_ttt':
                if diff > cfg['margin']:
                    ho_streak += 1
                else:
                    ho_streak = 0
                fire = ho_streak >= cfg['TTT']
            else:
                raise ValueError(policy)
            if fire:
                ho_origin = cur_cell
                ho_target = neigh
                log.append({'t': i, 'from': S_CONN, 'to': S_HO,
                            'cell': f"{ho_origin}->{ho_target}",
                            'trigger': f"HO fired ({policy})"})
                state = S_HO
                ho_progress = 0
                ho_streak = 0
        elif state == S_HO:
            ho_progress += 1
            if ho_progress >= cfg['HO_complete']:
                # Handover completes
                recent_left.append((i, ho_origin))
                cur_cell = ho_target
                log.append({'t': i, 'from': S_HO, 'to': S_CONN,
                            'cell': cur_cell,
                            'trigger': 'HO complete (target sync OK)'})
                state = S_CONN
                ho_progress = 0
                ho_target = None
                ho_origin = None
    return log


def analyze_log(log, cfg=CFG):
    """Post-process the transition log -> counts of HO / RLF / ping-pong."""
    handovers = [e for e in log if e['from'] == S_HO and e['to'] == S_CONN]
    rlfs      = [e for e in log if e['to']   == S_IDLE and e['trigger'] == 'RLF']
    # Ping-pong: handover back to a cell we left within pp_window samples
    pps = []
    left_history = []   # list of (t_left, cell_left)
    for e in log:
        if e['from'] == S_CONN and e['to'] == S_HO:
            # parse "A->B"
            orig = e['cell'].split('->')[0]
            left_history.append((e['t'], orig))
        elif e['from'] == S_HO and e['to'] == S_CONN:
            new_cell = e['cell']
            for (t_left, cell_left) in left_history:
                if cell_left == new_cell and (e['t'] - t_left) <= cfg['pp_window']:
                    pps.append({'t': e['t'], 'back_to': new_cell,
                                'gap': e['t'] - t_left})
                    break
    return {'handovers': len(handovers), 'rlfs': len(rlfs),
            'pingpongs': len(pps), 'handover_log': handovers,
            'rlf_log': rlfs, 'pp_log': pps}


# Run both policies
print("\n" + "=" * 78)
print("State-transition table (allowed transitions and triggers)")
print("=" * 78)
for src, dst, trig in TRANSITION_TRIGGERS:
    print(f"  {src:<14} -> {dst:<14} : {trig}")

print("\n" + "=" * 78)
print("Running scenario under NAIVE policy")
print("=" * 78)
np.random.seed(11)
log_naive = run_fsm(series, policy='naive')
res_naive = analyze_log(log_naive)
print(f"  Handovers  : {res_naive['handovers']}")
print(f"  RLFs       : {res_naive['rlfs']}")
print(f"  Ping-pongs : {res_naive['pingpongs']}")
print("\n  Transition log (first 30 entries):")
for e in log_naive[:30]:
    print(f"    t={e['t']:>3}  {e['from']:<13} -> {e['to']:<13} "
          f"cell={e['cell']:<8} ({e['trigger']})")

print("\n" + "=" * 78)
print("Running scenario under HYSTERESIS + TTT policy "
      f"(margin={CFG['margin']} dB, TTT={CFG['TTT']})")
print("=" * 78)
np.random.seed(11)
log_ttt = run_fsm(series, policy='hyst_ttt')
res_ttt = analyze_log(log_ttt)
print(f"  Handovers  : {res_ttt['handovers']}")
print(f"  RLFs       : {res_ttt['rlfs']}")
print(f"  Ping-pongs : {res_ttt['pingpongs']}")
print("\n  Transition log (first 30 entries):")
for e in log_ttt[:30]:
    print(f"    t={e['t']:>3}  {e['from']:<13} -> {e['to']:<13} "
          f"cell={e['cell']:<8} ({e['trigger']})")

# Per-policy summary table
print("\n" + "=" * 78)
print("Per-policy summary")
print("=" * 78)
print(f"{'Policy':<25} {'HO':>4} {'RLF':>4} {'Ping-pong':>10}")
print("-" * 45)
for name, res in [('naive',                res_naive),
                  ('hysteresis + TTT',     res_ttt)]:
    print(f"{name:<25} {res['handovers']:>4} {res['rlfs']:>4} "
          f"{res['pingpongs']:>10}")

# Plot the serving-cell timeline for both policies
def serving_timeline(log, series, N):
    """Reconstruct (cell, rsrp) per sample from a transition log."""
    cell_at = [None] * N
    rsrp_at = [np.nan] * N
    cur = None
    for i in range(N):
        # find any transition that occurred at sample i (or earlier affecting i)
        # simpler: replay the log up to sample i
        pass
    # Replay approach
    state = S_IDLE
    cur_cell = None
    ho_origin = None
    ho_target = None
    ho_progress = 0
    for i in range(N):
        # process transitions occurring at this sample
        for e in log:
            if e['t'] == i:
                # apply transition (we trust the log order)
                if e['from'] == S_IDLE and e['to'] == S_PREAM:
                    cur_cell = e['cell']
                elif e['from'] == S_CONT and e['to'] == S_CONN:
                    pass
                elif e['from'] == S_CONN and e['to'] == S_HO:
                    ho_origin, ho_target = e['cell'].split('->')
                elif e['from'] == S_HO and e['to'] == S_CONN:
                    cur_cell = e['cell']
                state = e['to']
        # Record serving cell for this sample
        if state in (S_CONN, S_HO) and cur_cell is not None:
            cell_at[i] = cur_cell
            rsrp_at[i] = series[cur_cell][i]
        elif state == S_HO and ho_target is not None:
            cell_at[i] = ho_target + '*'    # mid-handover marker
            rsrp_at[i] = series[ho_target][i]
    return cell_at, np.array(rsrp_at)

cell_n, rsrp_n = serving_timeline(log_naive, series, N)
cell_t, rsrp_t = serving_timeline(log_ttt, series, N)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, (name, cell_at, rsrp_at, log) in zip(axes,
        [('Naive policy', cell_n, rsrp_n, log_naive),
         ('Hysteresis + TTT policy', cell_t, rsrp_t, log_ttt)]):
    for c in cells:
        ax.plot(t, series[c], color={'A':'#1f77b4','B':'#d62728','C':'#2ca02c'}[c],
                alpha=0.35, lw=1.0, label=f'Cell {c}')
    ax.plot(t, rsrp_at, color='black', lw=2.4, label='Serving RSRP')
    # mark handovers
    for e in log:
        if e['from'] == S_CONN and e['to'] == S_HO:
            ax.axvline(e['t'], color='green', ls='--', lw=0.9, alpha=0.8)
        if e['to'] == S_IDLE and e['trigger'] == 'RLF':
            ax.axvline(e['t'], color='red', ls=':', lw=2.0, alpha=0.9)
    ax.axhline(CFG['RLF_thresh'], color='red', ls=':', lw=1.0)
    ax.set_ylabel('RSRP (dBm)')
    ax.set_title(f"{name}  -  HO={sum(1 for e in log if e['from']==S_HO and e['to']==S_CONN)}, "
                 f"RLF={sum(1 for e in log if e['trigger']=='RLF')}")
    ax.legend(loc='lower center', ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Sample index')
fig.suptitle('Lab Task - FSM serving-cell timeline under two policies', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{OUT_DIR}/mp3_3_serving_timeline.png', dpi=130)

# Save logs and summary
with open(f'{OUT_DIR}/mp3_3_log_naive.txt', 'w') as f:
    for e in log_naive:
        f.write(f"t={e['t']:>3}  {e['from']:<13} -> {e['to']:<13} "
                f"cell={e['cell']:<10} ({e['trigger']})\n")
with open(f'{OUT_DIR}/mp3_3_log_hyst_ttt.txt', 'w') as f:
    for e in log_ttt:
        f.write(f"t={e['t']:>3}  {e['from']:<13} -> {e['to']:<13} "
                f"cell={e['cell']:<10} ({e['trigger']})\n")
with open(f'{OUT_DIR}/mp3_3_summary.json', 'w') as f:
    json.dump({'naive':    {'HO': res_naive['handovers'], 'RLF': res_naive['rlfs'],
                            'PP': res_naive['pingpongs']},
               'hyst_ttt': {'HO': res_ttt['handovers'],   'RLF': res_ttt['rlfs'],
                            'PP': res_ttt['pingpongs']},
               'cfg': CFG}, f, indent=2, default=str)

print("Logs saved: mp3_3_log_naive.txt, mp3_3_log_hyst_ttt.txt")
print("Summary saved: mp3_3_summary.json")

if __name__ == "__main__":
    plt.show()
