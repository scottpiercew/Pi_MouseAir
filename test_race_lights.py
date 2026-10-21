import unittest
from unittest.mock import patch
from race_lights import race_sequence
from MouseAirMotion import ToyLauncher


class RaceLightsTests(unittest.TestCase):
    def test_toy_status_uses_active_low_sensor_and_logs_changes(self):
        from unittest.mock import Mock
        launcher = ToyLauncher()
        launcher.detector = Mock()
        launcher.detector.CheckMousePresent.side_effect = [1, 1, 0, 0, 1]
        with self.assertLogs('MouseAirMotion', level='INFO') as logs:
            states = [launcher.toy_loaded() for _ in range(5)]
        self.assertEqual(states, [False, False, True, True, False])
        self.assertEqual([line.rsplit(': ', 1)[-1] for line in logs.output], ['NO', 'YES', 'NO'])

    def test_colors_and_timing(self):
        events = []
        race_sequence(lambda *rgb: events.append(rgb),
                      lambda seconds: events.append(seconds) or False)
        self.assertEqual(events, [(255, 0, 0), 1, (255, 255, 0), 1,
                                  (0, 255, 0), 3, (0, 0, 0)])

    def test_cancellation_clears_lights(self):
        colors = []
        race_sequence(lambda *rgb: colors.append(rgb), lambda _: True)
        self.assertEqual(colors, [(255, 0, 0), (0, 0, 0)])

    def test_driver_error_still_clears_lights(self):
        colors = []
        def fill(*rgb):
            colors.append(rgb)
            if rgb != (0, 0, 0):
                raise OSError('driver failure')
        with self.assertRaises(OSError):
            race_sequence(fill, lambda _: False)
        self.assertEqual(colors[-1], (0, 0, 0))

    def test_lights_start_before_launch_and_do_not_wait_for_completion(self):
        calls = []
        launcher = ToyLauncher()
        class Lights:
            def start(self): calls.append('lights')
        class Detector:
            def CheckMousePresent(self): calls.append('sensor'); return 0
        launcher.lights, launcher.detector, launcher.motors = Lights(), Detector(), object()
        with patch('MouseAirMotion.launch_sequence', lambda _: calls.append('motors')):
            self.assertTrue(launcher.launch())
        self.assertEqual(calls, ['lights', 'sensor', 'motors', 'sensor'])

    def test_lights_also_run_without_loaded_toy(self):
        from unittest.mock import Mock
        launcher = ToyLauncher()
        launcher.lights, launcher.detector = Mock(), Mock()
        launcher.detector.CheckMousePresent.return_value = 1
        self.assertFalse(launcher.launch())
        launcher.lights.start.assert_called_once_with()
        self.assertIsNone(launcher.motors)

    def test_motor_shutdown_failure_still_stops_lights(self):
        from unittest.mock import Mock
        launcher = ToyLauncher()
        launcher.lights, launcher.motors = Mock(), Mock()
        launcher.motors.launchMotorsOff.side_effect = OSError('driver failure')
        with self.assertRaises(OSError):
            launcher.stop()
        launcher.lights.stop.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
