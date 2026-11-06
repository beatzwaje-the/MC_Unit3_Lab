# Mobile Communications — Unit 3 Lab (Mini Projects 3.1, 3.2, 3.3)

## Deliverables

| File | Description |
|------|-------------|
| `mp3_1.py` | Mini Project 3.1 — hexagonal frequency-reuse layouts (N = 1, 3, 4, 7) + handover decision simulation (naive / hysteresis / hysteresis+TTT). |
| `mp3_1_tasks.py` | Lab Tasks 1-6 for Mini Project 3.1 (N = 9, 12; three-cell strongest-neighbour; margin × TTT sweep; high-speed scenario; asymmetric offset; sectoring). |
| `mp3_2.py` | Mini Project 3.2 — cellular capacity planning tool (per-N capacity, Q, S/I, cell-splitting, sectoring illustration). |
| `mp3_3.py` | Mini Project 3.3 — FSM for device connection + handover (Idle / RACH × 4 / Connected / Handover) with RLF path, naive and hysteresis+TTT policies. |
| `figures/` | Generated figures (PNG), transition logs (TXT), and JSON summaries. |

This folder contains the simulation code and its generated outputs; assemble the
lab report separately using the numerical results and figures as needed.

## Running

```bash
python3 mp3_1.py          # core simulations for 3.1
python3 mp3_1_tasks.py    # lab tasks 1-6 for 3.1
python3 mp3_2.py          # capacity planning tool
python3 mp3_3.py          # FSM simulation
```

All scripts save figures and numerical/log outputs to `figures/` and display
their figures in Matplotlib windows when run directly. Close the plot windows
to let the script finish.
