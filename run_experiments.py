"""Run capture, representations, and batching experiments from the terminal."""
import argparse
import csv
import json
from pathlib import Path
from simulator import simulate, WIDTH, HEIGHT, SAMPLE_RATE, DURATION
from packetization import packetize, packet_statistics


# Purpose: Define the experiment controls and validate CLI numeric ranges.
# Inputs: Command-line arguments supplied by the user; --help lists all options.
# Output: argparse Namespace. Window values are milliseconds at the CLI boundary.
# Demo role: Reproducible commands replace a graphical control panel.
def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--scenario", choices=["move", "still", "flicker"], default="move")
    parser.add_argument("--window-ms", type=float, default=40)
    parser.add_argument("--packet-ms", type=float, default=20)
    parser.add_argument("--packet-count", type=int, default=700)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--show", action="store_true", help="Open native Matplotlib figure windows")
    parser.add_argument("--animate", action="store_true", help="Create animation; combine with --show for live playback")
    parser.add_argument("--save-animation", action="store_true", help="Export a 4-second GIF backup")
    args = parser.parse_args()
    import math
    for name in ["threshold", "window_ms", "packet_ms", "packet_count"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    return args


# Purpose: Export event tuples as a readable CSV for inspection or later analysis.
# Inputs: Event list and output path; timestamps are seconds, polarities +/-1.
# Output: CSV with a header; no return value.
# Important: CSV is human-readable, not the hypothetical 8-byte binary format.
def save_events(events, path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "t_seconds", "polarity"])
        writer.writerows(events)


# Purpose: Print concise classroom-readable results and all important assumptions.
# Inputs: Parsed settings, selected events, threshold sweep and batching statistics.
# Output: Text on stdout, including pending-event ages, not just successful waits.
# Demo role: Lets you explain results in the terminal before opening figures.
def print_summary(args, events, recordings, metrics):
    print(f"\nEVENT CAMERA EXPERIMENT | scene={args.scenario} | C={args.threshold:g}")
    print(f"Synthetic {WIDTH}x{HEIGHT} samples at {SAMPLE_RATE} Hz for {DURATION:g} s")
    print(f"Events: {len(events):,}; assumed binary payload: {len(events)*8:,} bytes")
    print(f"Raw 8-bit grayscale input: {WIDTH*HEIGHT*SAMPLE_RATE*int(DURATION):,} bytes")
    print("Payload comparison is NOT a compressed-video codec benchmark. CSV differs.")
    print("\nThreshold sweep (same scene):")
    for threshold, recording in recordings.items():
        print(f"  C={threshold:g}: {len(recording):,} events")
    print("\nPolicy          Sent    Pending   Mean wait(ms)  Max wait(ms)  Oldest pending(ms)")
    for policy, stats in metrics.items():
        mean = "N/A" if stats["mean_wait_ms"] is None else f'{stats["mean_wait_ms"]:.2f}'
        maximum = "N/A" if stats["max_wait_ms"] is None else f'{stats["max_wait_ms"]:.2f}'
        print(f'{policy:<14} {stats["sent_events"]:>7} {stats["pending_events"]:>9}'
              f' {mean:>15} {maximum:>13} {stats["oldest_pending_ms"]:>19.2f}')
    print("\nTiming is sample-limited (4.17 ms); model has no sensor noise or actual network.")
    print(f"Results: {args.output.resolve()}\n")


# Purpose: Orchestrate capture, analysis, file export and optional desktop display.
# Inputs: Command-line controls parsed above.
# Outputs: Events CSV, JSON metrics, four PNGs; optional native animation and GIF.
# Method: Reuse the selected simulation in the sweep, avoiding duplicate capture.
# Headless mode: Select Agg before importing plotting code if --show is absent.
# Demo role: A single entry point calls the small backend functions in clear order.
def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    events = simulate(args.threshold, args.scenario)
    recordings = {}
    for threshold in sorted({0.1, 0.2, 0.3, 0.5, args.threshold}):
        recordings[threshold] = events if threshold == args.threshold else simulate(threshold, args.scenario)
    window = args.packet_ms / 1000
    metrics = {policy: packet_statistics(packetize(events, policy, window, args.packet_count))
               for policy in ["time", "count", "hybrid"]}
    save_events(events, args.output/"events.csv")
    result = {"scenario": args.scenario, "threshold": args.threshold,
              "sample_rate_hz": SAMPLE_RATE, "duration_s": DURATION,
              "event_count": len(events), "assumed_bytes_per_event": 8,
              "event_window_ms": args.window_ms, "packet_window_ms": args.packet_ms,
              "packet_count": args.packet_count,
              "threshold_counts": {str(k): len(v) for k, v in recordings.items()},
              "packetization": metrics}
    (args.output/"metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize import (configure_style, plot_representations, plot_windows,
                           plot_thresholds, plot_packetization, create_animation)
    configure_style()
    plot_representations(events, args.scenario, args.threshold, args.output, window=args.window_ms/1000)
    plot_windows(events, args.output)
    plot_thresholds(recordings, args.output)
    plot_packetization(metrics, args.output, args.packet_count, window)
    animation = None
    if args.animate or args.save_animation:
        if args.show or args.save_animation:
            animation = create_animation(events, args.scenario, args.threshold,
                                         args.output, args.window_ms/1000, args.save_animation)
        else:
            print("Use --animate --show for live animation, or --save-animation for a GIF.")
    print_summary(args, events, recordings, metrics)
    if args.show:
        plt.show()  # Keep animation referenced until desktop windows are closed.
    else:
        plt.close("all")
    return animation


if __name__ == "__main__":
    main()
