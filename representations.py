"""Represent the same events as polarity counts, recency, and activity rate."""
import numpy as np
from simulator import WIDTH, HEIGHT


# Purpose: Accumulate a recent event window into two separate count images.
# Inputs: Ordered event tuples; end time; window duration in seconds.
# Method: Include timestamps in (time-window, time]; channel 0 is ON, 1 is OFF.
# Output: Integer array with shape (2, HEIGHT, WIDTH).
# Demo role: Shows motion trails and avoids cancellation of opposite polarities.
# Information lost: Exact within-window timestamps and event ordering.
def event_image(events, time, window=0.04):
    if window <= 0:
        raise ValueError("Window must be positive")
    counts = np.zeros((2, HEIGHT, WIDTH), dtype=np.int32)
    for x, y, timestamp, polarity in events:
        if timestamp > time:
            break
        if timestamp > time - window:
            counts[0 if polarity > 0 else 1, y, x] += 1
    return counts


# Purpose: Summarize activity by the latest event timestamp at each pixel.
# Inputs: Ordered events, current time and positive decay constant tau (seconds).
# Method: Keep only each pixel's last timestamp, then compute exp(-age/tau).
# Output: HEIGHT-by-WIDTH float image: 1 means just active; 0 means never active.
# Demo role: Visualizes recent history independently of event polarity.
# Information lost: Every earlier event at a pixel is overwritten.
def time_surface(events, time, tau=0.12):
    if tau <= 0:
        raise ValueError("Decay constant must be positive")
    latest = np.full((HEIGHT, WIDTH), -np.inf)
    for x, y, timestamp, _ in events:
        if timestamp > time:
            break
        latest[y, x] = timestamp
    return np.exp(-(time - latest) / tau)


# Purpose: Convert separate polarity counts to an RGB image for Matplotlib.
# Input: Two-channel counts from event_image; saturation count defaults to 3.
# Method: Use cyan for ON and pink for OFF. Counts above saturation clip visually.
# Output: Float RGB image in [0,1]; the underlying count data is unchanged.
# Demo role: Color describes the SIGN of change, not the absolute scene color.
def colorize_events(counts, saturation=3):
    on = np.clip(counts[0] / saturation, 0, 1)
    off = np.clip(counts[1] / saturation, 0, 1)
    rgb = np.zeros((*on.shape, 3))
    rgb[..., 0] = 0.04 + 0.92 * off
    rgb[..., 1] = 0.06 + 0.80 * on
    rgb[..., 2] = 0.10 + 0.55 * on + 0.30 * off
    return np.clip(rgb, 0, 1)


# Purpose: Measure event activity in time bins, expressed as events per second.
# Inputs: Events, recording duration, requested bin width in seconds.
# Method: Histogram timestamps; divide by each bin's actual duration.
# Output: Bin centers and event rates; the last bin includes the horizon endpoint.
# Demo role: Makes burstiness and the stop at 2.5 s visible.
def event_rate(events, duration=4.0, bin_width=0.05):
    if duration <= 0 or bin_width <= 0:
        raise ValueError("Duration and bin width must be positive")
    edges = np.arange(0, duration, bin_width)
    edges = np.append(edges, duration)
    counts, _ = np.histogram([event[2] for event in events], bins=edges)
    return (edges[:-1] + edges[1:]) / 2, counts / np.diff(edges)
