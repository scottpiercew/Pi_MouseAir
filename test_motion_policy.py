import math
import unittest
from motion_policy import MotionPolicy, launch_sequence, cat_probability


class PolicyTests(unittest.TestCase):
    def test_imagenet_combines_domestic_cat_breeds(self):
        scores = [0.0] * 1000
        scores[281], scores[282], scores[285] = 0.2, 0.3, 0.25
        self.assertAlmostEqual(cat_probability(scores, [281, 282, 283, 284, 285]), 0.75)

    def test_non_cat_classes_do_not_count(self):
        scores = [0.0] * 1000
        scores[207] = 0.99
        scores[281] = 0.01
        self.assertAlmostEqual(cat_probability(scores, [281, 282, 283, 284, 285]), 0.01)

    def test_invalid_classifier_output_fails_closed(self):
        for scores, indices in [([float('nan'), 0.5], [0]), ([0.2, 0.8], [2]), ([0.2, 0.8], [0, 0])]:
            with self.assertRaises(ValueError):
                cat_probability(scores, indices)

    def test_sixty_percent_threshold(self):
        p = MotionPolicy(confidence=0.60)
        self.assertTrue(p.accepts_cat(0.60, 1))
        self.assertFalse(p.accepts_cat(0.599, 1))

    def test_three_moving_frames_required(self):
        p = MotionPolicy()
        self.assertFalse(p.observe(True, 1))
        self.assertFalse(p.observe(True, 2))
        self.assertTrue(p.observe(True, 3))

    def test_continuous_motion_has_only_one_capture(self):
        p = MotionPolicy()
        self.assertEqual(sum(p.observe(True, t) for t in range(50)), 1)

    def test_continuing_motion_retries_rejected_frame(self):
        p = MotionPolicy(retry_seconds=5)
        self.assertEqual([t for t in range(13) if p.observe(True, t)], [2, 7, 12])

    def test_retry_does_not_bypass_launch_cooldown(self):
        p = MotionPolicy(retry_seconds=5, cooldown=30)
        for t in range(3):
            p.observe(True, t)
        p.launched(2)
        self.assertFalse(any(p.observe(True, t) for t in range(3, 32)))
        self.assertTrue(p.observe(True, 32))

    def test_quiet_period_rearms_capture(self):
        p = MotionPolicy()
        for t in range(3):
            p.observe(True, t)
        for t in range(5):
            p.observe(False, t + 3)
        self.assertFalse(p.observe(True, 8))
        self.assertFalse(p.observe(True, 9))
        self.assertTrue(p.observe(True, 10))

    def test_low_or_invalid_confidence_cannot_launch(self):
        p = MotionPolicy()
        for score in [0, 0.5, 0.89999, float('nan'), float('inf'), 1.1, -1]:
            self.assertFalse(p.accepts_cat(score, 1))
        self.assertTrue(p.accepts_cat(0.9, 1))

    def test_launch_cooldown(self):
        p = MotionPolicy(cooldown=30)
        p.launched(10)
        self.assertFalse(p.accepts_cat(1, 39.99))
        self.assertTrue(p.accepts_cat(1, 40))
        self.assertFalse(p.observe(True, 20))
        self.assertFalse(p.observe(True, 21))
        self.assertFalse(p.observe(True, 22))

    def test_motor_stop_on_servo_failure(self):
        calls = []
        class Fake:
            def launchMotorsOn(self): calls.append('on')
            def launchServoStart(self): raise OSError('simulated I2C error')
            def launchMotorsOff(self): calls.append('off')
            def launchServoRetract(self): calls.append('retract')
        with self.assertRaises(OSError):
            launch_sequence(Fake())
        self.assertEqual(calls, ['on', 'off'])

    def test_motor_stop_on_partial_start_failure(self):
        calls = []
        class Fake:
            def launchMotorsOn(self): raise OSError('partial start')
            def launchMotorsOff(self): calls.append('off')
        with self.assertRaises(OSError):
            launch_sequence(Fake())
        self.assertEqual(calls, ['off'])

    def test_successful_sequence_preserves_order(self):
        calls = []
        class Fake:
            def launchMotorsOn(self): calls.append('on')
            def launchServoStart(self): calls.append('forward')
            def launchMotorsOff(self): calls.append('off')
            def launchServoRetract(self): calls.append('retract')
        launch_sequence(Fake())
        self.assertEqual(calls, ['on', 'forward', 'off', 'retract'])


if __name__ == '__main__':
    unittest.main()
