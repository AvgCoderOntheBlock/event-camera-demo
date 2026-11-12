"""Scientific plots and optional native Matplotlib animation; no browser or server."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from simulator import create_scene, WIDTH, HEIGHT
from representations import event_image, time_surface, colorize_events, event_rate


# Purpose: Apply one consistent, readable style for classroom plots.
# Input/output: No input; updates Matplotlib style defaults for subsequent figures.
# Demo role: Labels, axes and units are part of explaining the experiment.
def configure_style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                         "figure.dpi": 110, "savefig.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})


# Purpose: Save a figure with predictable margins and a stable filename.
# Inputs: Matplotlib figure, destination directory and PNG filename.
# Output: Saved path; figure stays open so --show can display it later.
def save_figure(figure, output, filename):
    path = Path(output) / filename
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    return path


# Purpose: Compare source intensity, polarity activity and recency at one instant.
# Inputs: Captured events, scene, threshold, output directory, time and window.
# Output: Saved PNG plus an open figure. Axes retain sensor pixel coordinates.
# Demo role: The three panels describe the SAME scene/time with different data.
def plot_representations(events, scenario, threshold, output, time=1.5, window=0.04):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    counts = event_image(events, time, window)
    axes[0].imshow(create_scene(time, scenario), cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(colorize_events(counts))
    surface = axes[2].imshow(time_surface(events, time), cmap="viridis", vmin=0, vmax=1)
    titles = ["Input grayscale intensity", f"Events in last {window*1000:g} ms\nCyan: ON | Pink: OFF",
              "Time surface (tau = 120 ms)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    fig.colorbar(surface, ax=axes[2], label="Recency exp(-age/tau)", shrink=0.75)
    fig.suptitle(f"Sampled event-camera model | t={time:.2f} s | C={threshold:g} | {scenario}")
    return save_figure(fig, output, "01_representations.png")


# Purpose: Show the representation effect of short versus long accumulation.
# Inputs: Same event stream, output directory and fixed observation time.
# Method: Hold data and end time constant; vary ONLY the accumulation window.
# Output: Three-panel PNG using the same color/count saturation scale.
# Demo role: Longer windows gather more events and extend trails, losing timing.
def plot_windows(events, output, time=1.5):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, window in zip(axes, [0.005, 0.04, 0.2]):
        counts = event_image(events, time, window)
        ax.imshow(colorize_events(counts))
        ax.set_title(f"{window*1000:g} ms | {counts.sum():,} events")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    fig.suptitle(f"Accumulation windows ending at t={time:.2f} s (same event stream)")
    return save_figure(fig, output, "02_accumulation_windows.png")


# Purpose: Show how capture sensitivity changes this controlled scene's output.
# Input: Mapping from threshold values to event lists; output directory; duration.
# Method: Plot total counts and event-rate traces with fixed 50 ms time bins.
# Output: PNG containing a count comparison and complete activity timeline.
# Demo role: Relates contrast detail to event volume; vertical line marks motion stop.
def plot_thresholds(recordings, output, duration=4.0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = [f"{threshold:g}" for threshold in recordings]
    values = [len(events) for events in recordings.values()]
    axes[0].bar(labels, values, color="#267f90")
    for i, value in enumerate(values):
        axes[0].text(i, value, f"{value:,}", ha="center", va="bottom")
    axes[0].set_ylim(0, max(values + [1]) * 1.2)
    axes[0].set(xlabel="Contrast threshold C", ylabel="Events over 4 seconds",
                title="Smaller C reveals smaller changes")
    for threshold, events in recordings.items():
        centers, rates = event_rate(events, duration)
        axes[1].plot(centers, rates, label=f"C={threshold:g}")
    axes[1].axvline(2.5, color="black", linestyle="--", label="Motion stops")
    axes[1].set(xlabel="Time (seconds)", ylabel="Events / second (50 ms bins)",
                title="Output follows scene activity")
    axes[1].legend(fontsize=8)
    return save_figure(fig, output, "03_threshold_comparison.png")


# Purpose: Compare transmitted-event wait with explicit pending-event counts.
# Inputs: Policy-name to metrics mapping, output directory, selected count/timeout.
# Method: Plot mean/max wait for sent events and pending count on separate axes.
# Output: PNG; empty sent sets use zero-height bars marked N/A, not measured zeros.
# Demo role: Keeps batching latency distinct from network latency and tail starvation.
def plot_packetization(metrics, output, count, window):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    names = list(metrics)
    positions = np.arange(len(names))
    for shift, field, label in [(-0.18, "mean_wait_ms", "Mean sent-event wait"),
                                 (0.18, "max_wait_ms", "Maximum sent-event wait")]:
        values = [metrics[name][field] for name in names]
        axes[0].bar(positions+shift, [v if v is not None else 0 for v in values],
                    width=0.36, label=label)
        for x, value in zip(positions+shift, values):
            if value is None:
                axes[0].text(x, 0, "N/A", ha="center", fontsize=8)
    axes[0].set_xticks(positions, names)
    axes[0].set(ylabel="Batching delay (milliseconds)", title="Delay among transmitted events")
    axes[0].legend(fontsize=8)
    pending = [metrics[name]["pending_events"] for name in names]
    axes[1].bar(names, pending, color="#b25d3f")
    for i, value in enumerate(pending):
        axes[1].text(i, value, str(value), ha="center", va="bottom")
    axes[1].set_ylim(0, max(pending + [1]) * 1.25)
    axes[1].set(ylabel="Events not yet transmitted", title="Unfinished tails at t=4 seconds")
    fig.suptitle(f"Grouping only | count={count} | window/timeout={window*1000:g} ms")
    return save_figure(fig, output, "04_packetization.png")


# Purpose: Update all animation panels at the same simulated instant.
# Inputs: Frame index; fixed display times; events/scene; artists and text handle.
# Outputs: Updated artists, suitable for FuncAnimation; no capture state is changed.
# Demo role: Playback samples saved events, separating capture from visualization.
def update_animation(index, times, events, scenario, artists, title, threshold, window):
    time = times[index]
    artists[0].set_data(create_scene(time, scenario))
    counts = event_image(events, time, window)
    artists[1].set_data(colorize_events(counts))
    artists[2].set_data(time_surface(events, time))
    state = "stationary" if scenario == "still" else "moving" if time < 2.5 else "motion stopped"
    title.set_text(f"t={time:.2f} s | C={threshold:g} | {state} | {counts.sum():,} window events")
    return [*artists, title]


# Purpose: Prepare a native Matplotlib animation and optionally export a GIF.
# Inputs: Event stream, scene, threshold, output directory, window and save flag.
# Method: Display at 20 fps over 4 s; capture remains the separate 240 Hz model.
# Output: FuncAnimation object (caller must retain it); GIF if save=True.
# Demo role: Native desktop playback for class, with a file backup for reliability.
# Limit: Saving GIF needs Pillow; no browser, server, ffmpeg or internet is needed.
def create_animation(events, scenario, threshold, output, window=0.04, save=False):
    times = np.linspace(0, 4, 81)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    artists = [axes[0].imshow(create_scene(0, scenario), cmap="gray", vmin=0, vmax=1),
               axes[1].imshow(colorize_events(event_image(events, 0, window))),
               axes[2].imshow(time_surface(events, 0), cmap="viridis", vmin=0, vmax=1)]
    for ax, label in zip(axes, ["Intensity", "Events: cyan ON / pink OFF", "Time surface"]):
        ax.set_title(label)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    title = fig.suptitle("")
    animation = FuncAnimation(fig, update_animation, frames=len(times), interval=50,
                             fargs=(times, events, scenario, artists, title, threshold, window),
                             blit=False, repeat=True)
    if save:
        animation.save(Path(output)/"05_demo.gif", writer=PillowWriter(fps=20), dpi=85)
    return animation
