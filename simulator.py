"""Ideal event-camera capture from deterministic synthetic grayscale samples.

Original teaching code based on Gallego et al., Event-based Vision: A Survey,
Section 2.4. Events are tuples (x, y, timestamp_seconds, polarity).
"""
import numpy as np

WIDTH, HEIGHT = 96, 64
SAMPLE_RATE, DURATION, EPSILON = 240, 4.0, 1e-6


# Purpose: Build one small grayscale image without loading any dataset or video.
# Inputs: time in seconds; scenario is 'move', 'still', or 'flicker'.
# Method: Draw a bright square and a faint square (25% above the background).
#         Move both at 20 pixels/second, then stop them at 2.5 seconds. In the
#         flicker scenario, multiply every pixel by a global sinusoidal light.
# Output: A HEIGHT-by-WIDTH float array with intensity values between 0 and 1.
# Demo role: Provides reproducible stimuli that isolate motion, contrast and light.
def create_scene(time, scenario="move"):
    if scenario not in {"move", "still", "flicker"}:
        raise ValueError("Unknown scene")
    image = np.full((HEIGHT, WIDTH), 0.12, dtype=np.float64)
    moving_time = 0 if scenario == "still" else min(time, 2.5)
    left = 8 + int(np.floor(20 * moving_time + 0.5))
    image[12:32, left:left + 18] = 0.75
    image[42:54, left:left + 12] = 0.15
    if scenario == "flicker":
        image *= 1 + 0.35 * np.sin(2 * np.pi * 3 * time)
    return image


# Purpose: Convert nonnegative linear intensities to natural-log intensities.
# Input: A finite nonnegative image array, usually with values in [0, 1].
# Method: Add a tiny epsilon before taking log, so black pixels do not give -inf.
# Output: A new floating-point array; the input image is not changed.
# Demo role: In log space, equal intensity ratios produce equal differences.
def log_intensity(image):
    image = np.asarray(image, dtype=np.float64)
    if not np.all(np.isfinite(image)) or np.any(image < 0):
        raise ValueError("Intensities must be finite and nonnegative")
    return np.log(image + EPSILON)


# Purpose: Establish a separate stored brightness baseline for every pixel.
# Input: The initial image of the sequence.
# Output: A new mutable reference array containing initial log intensities.
# Important: This step emits no events. The initial absolute image is not part
#            of the exported event stream; we only use it to start simulation.
def initialize_reference(initial_image):
    return log_intensity(initial_image)


# Purpose: Turn accumulated per-pixel log-brightness changes into signed events.
# Inputs: Current image; mutable reference array of the same shape; sample time;
#         finite positive contrast threshold C in natural-log intensity units.
# Method: Count how many full thresholds fit into each absolute difference.
#         Emit one event per crossing and advance the baseline by signed n*C.
# Outputs: List of (x,y,t,p) events; reference is updated IN PLACE.
# Important: A 0.75 change at C=0.2 emits 3 events and retains a 0.15 residual.
#            Crossings inside a sample share its timestamp: no sub-sample timing
#            is claimed. A 1e-12 tolerance handles floating-point boundaries.
def generate_events(image, reference, time, threshold):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Threshold must be finite and positive")
    current = log_intensity(image)
    if current.shape != reference.shape:
        raise ValueError("Image and reference shapes must match")
    difference = current - reference
    counts = np.floor((np.abs(difference) + 1e-12) / threshold).astype(int)
    polarities = np.where(difference >= 0, 1, -1)
    events = []
    for y, x in zip(*np.nonzero(counts)):
        for _ in range(counts[y, x]):
            events.append((int(x), int(y), float(time), int(polarities[y, x])))
    reference += polarities * counts * threshold
    return events


# Purpose: Run the capture model on a complete sampled synthetic sequence.
# Inputs: Threshold, scene choice, duration in seconds, input samples per second.
# Method: Initialize once; generate frames at k/sample_rate and update reference.
# Output: Chronologically ordered event tuples, including simultaneous events.
# Demo role: This is the main backend experiment.
# Limit: Input defaults to 240 Hz, not a physical sensor's microsecond timing.
def simulate(threshold=0.3, scenario="move", duration=DURATION,
             sample_rate=SAMPLE_RATE):
    if duration <= 0 or sample_rate <= 0:
        raise ValueError("Duration and sample rate must be positive")
    reference = initialize_reference(create_scene(0, scenario))
    events = []
    for step in range(1, int(np.floor(duration * sample_rate + 1e-9)) + 1):
        time = step / sample_rate
        events.extend(generate_events(create_scene(time, scenario), reference,
                                      time, threshold))
    return events
