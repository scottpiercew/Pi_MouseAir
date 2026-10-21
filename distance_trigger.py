"""Filtered distance-change events across the ranger's usable range."""
from collections import deque
import math
import statistics


class DistanceTrigger:
    def __init__(self, minimum=3.0, maximum=350.0, change=8.0, samples=3):
        if not all(math.isfinite(v) for v in (minimum, maximum, change)) or not 0 < minimum < maximum or change <= 0 or samples < 1:
            raise ValueError('Invalid ultrasonic range/filter settings')
        self.minimum, self.maximum, self.change = minimum, maximum, change
        self.samples = samples
        self.window = deque(maxlen=samples)
        self.baseline = None
        self.misses = 0
        self.distance = None

    def observe(self, distance, now, not_before=0):
        if distance is None or not math.isfinite(distance) or not self.minimum <= distance <= self.maximum:
            self.window.clear()
            self.distance = None
            self.misses += 1
            if self.misses >= self.samples:
                self.baseline = None
            return False
        self.misses = 0
        self.window.append(distance)
        if len(self.window) < self.samples:
            return False
        self.distance = statistics.median(self.window)
        changed = self.baseline is None or abs(self.distance - self.baseline) >= self.change
        # Retain the prior baseline while rate-limited: a cat that arrives and
        # settles during the cooldown can still get a photo when it expires.
        if changed and now >= not_before:
            self.baseline = self.distance
            return True
        return False
