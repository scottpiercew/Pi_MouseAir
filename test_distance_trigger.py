import unittest
from distance_trigger import DistanceTrigger
from ultrasonic_ranger import UltrasonicRanger


class DistanceTests(unittest.TestCase):
    def feed(self, trigger, readings, start=0, not_before=0):
        return [trigger.observe(v, start + i * .25, not_before) for i, v in enumerate(readings)]

    def test_near_and_far_objects_and_both_directions(self):
        t = DistanceTrigger()
        self.assertEqual(self.feed(t, [30] * 3), [False, False, True])
        self.assertEqual(self.feed(t, [300] * 3, 1), [False, True, False])
        self.assertEqual(self.feed(t, [40] * 3, 2), [False, True, False])

    def test_stationary_object_does_not_retrigger(self):
        self.assertEqual(sum(self.feed(DistanceTrigger(), [50] * 30)), 1)

    def test_jitter_and_isolated_spike_do_not_trigger(self):
        t = DistanceTrigger()
        self.feed(t, [40] * 3)
        self.assertFalse(any(self.feed(t, [39, 42, 40, 200, 40, 41, 39], 1)))

    def test_timeouts_and_out_of_range_never_trigger(self):
        self.assertFalse(any(self.feed(DistanceTrigger(), [None, 525, -1, 0, 400, float('nan'), float('inf')] * 3)))

    def test_reentry_after_missing_echo_triggers_again(self):
        t = DistanceTrigger()
        self.feed(t, [40] * 3)
        self.assertFalse(any(self.feed(t, [None] * 3, 1)))
        self.assertEqual(self.feed(t, [40] * 3, 2), [False, False, True])

    def test_change_during_cooldown_remains_pending(self):
        t = DistanceTrigger()
        self.feed(t, [40] * 3)
        self.assertFalse(any(self.feed(t, [80] * 20, 1, not_before=10)))
        self.assertTrue(t.observe(80, 10, not_before=10))
        self.assertFalse(t.observe(80, 11, not_before=10))

    def test_approach_accumulates_small_steps(self):
        t = DistanceTrigger()
        self.feed(t, [100] * 3)
        self.assertTrue(any(self.feed(t, [98, 96, 94, 92, 90, 88], 1)))

    def test_invalid_settings(self):
        for kwargs in ({'minimum': 0}, {'maximum': 2}, {'change': 0}, {'change': float('nan')}):
            with self.assertRaises(ValueError):
                DistanceTrigger(**kwargs)


class FakeGPIO:
    BCM, OUT, IN = 11, 0, 1
    def __init__(self, mode):
        self.now, self.mode, self.cleaned = 0, mode, []
    def setmode(self, _): pass
    def setwarnings(self, _): pass
    def setup(self, *_): pass
    def output(self, *_): pass
    def cleanup(self, pin): self.cleaned.append(pin)
    def sleep(self, seconds): self.now += seconds
    def clock(self): return self.now
    def input(self, _):
        if self.mode == 'low': return 0
        if self.mode == 'high': return 1
        return int(.001 <= self.now < .003)


class RangerTests(unittest.TestCase):
    def make(self, mode):
        g = FakeGPIO(mode)
        return g, UltrasonicRanger(26, g, g.clock, g.sleep)

    def test_stuck_low_and_high_reads_are_bounded(self):
        for mode in ('low', 'high'):
            g, ranger = self.make(mode)
            self.assertIsNone(ranger.read_cm())
            self.assertLess(g.now, .041)

    def test_echo_pulse_converts_to_distance(self):
        _, ranger = self.make('echo')
        self.assertAlmostEqual(ranger.read_cm(), 34.3, delta=1)

    def test_cleanup_only_releases_ranger_pin(self):
        g, ranger = self.make('echo')
        ranger.close()
        self.assertEqual(g.cleaned, [26])


if __name__ == '__main__':
    unittest.main()
