# Event camera demo — Python + Matplotlib

A terminal-driven, original implementation for a 15-minute multimedia systems
Hacker presentation. No website, browser, server, Streamlit, or camera required.

## Quick start (Windows PowerShell)

Extract this folder, open its location in your editor, and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_experiments.py --threshold 0.3 --show --animate
```

Using the virtual environment's Python directly avoids PowerShell activation-policy
issues. Close the Matplotlib windows to return to the terminal. Animation loops.
The figure toolbar supports zoom/pan/save; there are no custom application controls.

## macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_experiments.py --threshold 0.3 --show --animate
```

A desktop GUI backend is needed for --show. If your Python lacks one, the saved
PNGs and GIF work in ordinary image viewers. No GUI is required for file export:

```bash
python run_experiments.py --threshold 0.3 --save-animation
```

These shorter `python` commands assume that Python has the dependencies installed;
otherwise substitute your virtual environment's Python path above.

## Experiments to run

```bash
python run_experiments.py --threshold 0.1 --output results_low --show --animate
python run_experiments.py --threshold 0.3 --scenario still --output results_still --show
python run_experiments.py --threshold 0.3 --scenario flicker --output results_flicker --show --animate
python run_experiments.py --threshold 0.3 --window-ms 200 --output results_long --show
python run_experiments.py --threshold 0.3 --packet-count 700 --packet-ms 20
python -m unittest -v
```

Every experiment also runs a threshold sweep on the SAME selected scene, so the
third plot always includes C=0.1, 0.2, 0.3 and 0.5 plus your selected threshold.
Each call saves four figures, events.csv and metrics.json. Animation is opt-in.
Use separate --output directories to keep comparisons; otherwise files overwrite.

## Source walkthrough order

| File | Read these functions | Why |
|---|---|---|
| simulator.py | create_scene | Define the toy problem |
| simulator.py | log_intensity, initialize_reference | Explain per-pixel baseline |
| simulator.py | generate_events | Core algorithm: thresholds and residuals |
| simulator.py | simulate | Connect frame samples to a time-ordered event stream |
| representations.py | event_image, time_surface | Explain what each representation discards |
| packetization.py | packetize, packet_statistics | Explain batching delay and pending tails |
| run_experiments.py | main | Show how small functions form the experiment |
| visualize.py | plot_* and create_animation | Display results without changing capture |

Every function has a detailed comment block BEFORE `def`, including purpose,
inputs, outputs, and relevant assumptions. The backend is separate from plotting.
The functions in test_demo.py have preceding comments describing their fixtures
and expected behavior too.

## Pre-generated results

The results/ directory includes the default C=0.3 moving-scene outputs:
- 01_representations.png: intensity, polarity image, time surface.
- 02_accumulation_windows.png: 5, 40 and 200 ms event windows.
- 03_threshold_comparison.png: total events and activity timeline.
- 04_packetization.png: sent-event delays and unsent tails.
- 05_demo.gif: looping four-second backup animation (20 fps playback).
- events.csv: selected event stream with x,y,t_seconds,polarity.
- metrics.json: settings, event counts and packet metrics.

The simulator captures 240 samples/second; the GIF displays 20 frames/second.
Those are different sampling stages. Plot generation is not capture latency.

## Known results and interpretation

Default moving scene: 38,400 events at C=0.1; 12,000 at C=0.3.
The faint square has log contrast about log(1.25)=0.223, so C=0.3 suppresses it.
The bright square still generates multiple threshold crossings per transition.
When motion stops at 2.5 s, ideal event generation stops; event images take one
window to empty, and the time surface fades according to tau=120 ms.

With C=0.3 and count=700, count-only batching sends 11,900 events and leaves
100 pending at the 4 s horizon. Their oldest event has waited 1,525 ms.
Time and hybrid policies transmit all 12,000 by the horizon at timeout=20 ms.
This is a deterministic grouping example, not a measured network experiment.

## Attribution and scope

Based on Gallego et al., *Event-based Vision: A Survey*, IEEE TPAMI (2022),
particularly Section 2.4 (capture), Section 3.1 (representations), and the
compression/reconstruction discussion in Section 4.6. The attached version has
an earlier manuscript date. Use your course's bibliographic citation.

All event-generation and batching code was written for this teaching demo, not
downloaded from an existing event-camera implementation. NumPy supplies arrays;
Matplotlib supplies plotting/animation; Pillow writes GIFs. This is a simplified
model described by a survey, not a reproduction of a proposed novel algorithm.
Packetization is a classroom extension motivated by variable-rate event output.


