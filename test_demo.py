"""Run backend tests with: python -m unittest -v. No GUI is required."""
import unittest
import numpy as np
from simulator import generate_events, simulate, EPSILON
from representations import event_image, time_surface, event_rate
from packetization import packetize, packet_statistics


class CaptureTests(unittest.TestCase):
    # Purpose: Verify multiple thresholds, reference mutation and residual retention.
    # Fixture: Log intensity jumps from 0 to 0.75, then falls to 0.10 at C=0.2.
    # Expected: Three positive events, then two negative events; baselines 0.6/0.2.
    def test_crossings_and_residual(self):
        reference = np.zeros((1, 1))
        positive = generate_events(np.array([[np.exp(0.75)-EPSILON]]), reference, 1, 0.2)
        self.assertEqual(len(positive), 3)
        self.assertTrue(all(event[3] == 1 for event in positive))
        self.assertAlmostEqual(reference[0, 0], 0.6)
        negative = generate_events(np.array([[np.exp(0.10)-EPSILON]]), reference, 2, 0.2)
        self.assertEqual(len(negative), 2)
        self.assertTrue(all(event[3] == -1 for event in negative))
        self.assertAlmostEqual(reference[0, 0], 0.2)

    # Purpose: Verify slow subthreshold changes accumulate against the saved baseline.
    # Expected: A first 0.1 change emits nothing; a later 0.25 total emits one event.
    def test_accumulation(self):
        reference = np.zeros((1, 1))
        self.assertFalse(generate_events(np.array([[np.exp(.1)-EPSILON]]), reference, 1, .2))
        self.assertAlmostEqual(reference[0, 0], 0)
        self.assertEqual(len(generate_events(np.array([[np.exp(.25)-EPSILON]]), reference, 2, .2)), 1)

    # Purpose: Check scene-level behavior and exact controlled-recording totals.
    # Expected: No stationary events; lower C reveals faint square; no post-stop events.
    def test_toy_scenes(self):
        self.assertEqual(simulate(scenario="still"), [])
        low, high = simulate(.1), simulate(.3)
        self.assertEqual((len(low), len(high)), (38400, 12000))
        self.assertTrue(any(event[1] >= 42 for event in low))
        self.assertFalse(any(event[1] >= 42 for event in high))
        self.assertTrue(all(event[2] <= 2.5 for event in high))
        flicker = simulate(.3, "flicker")
        self.assertTrue(any(event[2] > 2.5 for event in flicker))

    # Purpose: Reject invalid thresholds rather than produce undefined event counts.
    # Expected: Zero, negative and nonfinite C raise a clear ValueError.
    def test_invalid_threshold(self):
        for threshold in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                generate_events(np.ones((1, 1)), np.zeros((1, 1)), 0, threshold)


class RepresentationTests(unittest.TestCase):
    # Purpose: Verify polarity separation, window boundaries, future exclusion and recency.
    # Fixture: Opposite events share a pixel; a future event must not leak backward.
    # Expected: Both channels count independently and latest event has recency 1.
    def test_representations(self):
        events = [(1, 1, .01, 1), (1, 1, .02, -1), (2, 1, .03, 1)]
        image = event_image(events, .025, .02)
        self.assertEqual(tuple(image[:, 1, 1]), (1, 1))
        self.assertEqual(image[:, 1, 2].sum(), 0)
        surface = time_surface(events, .02)
        self.assertEqual(surface[1, 1], 1)
        self.assertEqual(surface[0, 0], 0)
        centers, rates = event_rate(events, .1, .05)
        self.assertAlmostEqual(sum(rates * .05), 3)


class PacketTests(unittest.TestCase):
    # Purpose: Check event conservation and each policy's intended waiting behavior.
    # Fixture: Three arrivals, count limit 2, timeout 20 ms, horizon 100 ms.
    # Expected: Count mode has an unfinished tail; time/hybrid waits are bounded.
    def test_conservation_and_bounds(self):
        events = [(0, 0, .01, 1), (0, 0, .02, 1), (0, 0, .03, 1)]
        for policy in ["time", "count", "hybrid"]:
            result = packetize(events, policy, .02, 2, .1)
            stats = packet_statistics(result, .1)
            self.assertEqual(stats["sent_events"] + stats["pending_events"], 3)
            if policy == "count":
                self.assertEqual(stats["pending_events"], 1)
                self.assertAlmostEqual(stats["oldest_pending_ms"], 70)
            else:
                self.assertLessEqual(stats["max_wait_ms"], 20 + 1e-8)

    # Purpose: Verify a timeout can fire without a subsequent event arrival.
    # Fixture: One event at 10 ms, observed at 100 ms and separately at 20 ms.
    # Expected: Hybrid releases at 30 ms only if that deadline has been reached.
    def test_idle_timeout_and_horizon(self):
        event = [(0, 0, .01, 1)]
        released, pending = packetize(event, "hybrid", .02, 700, .1)
        self.assertAlmostEqual(released[0][0], .03)
        self.assertFalse(pending)
        released, pending = packetize(event, "hybrid", .02, 700, .02)
        self.assertFalse(released)
        self.assertEqual(len(pending), 1)

    # Purpose: Check no-event records produce N/A sent-event delays rather than zero.
    # Expected: An empty stream has no packets, no pending events and undefined mean.
    def test_empty_stream(self):
        stats = packet_statistics(packetize([]))
        self.assertEqual(stats["packets"], 0)
        self.assertIsNone(stats["mean_wait_ms"])


if __name__ == "__main__":
    unittest.main()
