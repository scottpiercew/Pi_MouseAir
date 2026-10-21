"""Exercise the actual controller loop without camera/GPIO/motor hardware."""
import signal
import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock, patch

import MouseAirMotion as controller


class ControllerTests(TestCase):
    def run_controller(self, distance, camera_module):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.tflite'
            model.write_bytes(b'test')
            classifier = Mock(default_confidence=.6, name='test', last_view='full')
            classifier.classify.return_value = 0.0
            ranger = Mock(pin=26)
            ranger.read_cm.return_value = distance
            cv2 = Mock()
            def save(filename, *_):
                Path(filename).write_bytes(b'photo')
                events.append('save')
                return True
            cv2.imwrite.side_effect = save
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
            import os
            cwd = os.getcwd()
            try:
                with patch.object(controller, 'ROOT', root), \
                     patch.object(controller, 'CatClassifier', return_value=classifier), \
                     patch('ultrasonic_ranger.UltrasonicRanger', return_value=ranger), \
                     patch.dict(sys.modules, {'cv2': cv2, 'numpy': Mock(), 'picamera': camera_module}), \
                     patch.object(sys, 'argv', ['controller', '--model', str(model), '--dry-run',
                                               '--seconds', '.5', '--distance-interval', '.1',
                                               '--camera-warmup', '0']):
                    controller.main()
            finally:
                os.chdir(cwd)
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
            ranger.close.assert_called_once_with()
            return events, classifier

    def test_no_echo_never_opens_camera_or_runs_inference(self):
        events, classifier = self.run_controller(None, None)
        self.assertEqual(events, [])
        classifier.classify.assert_not_called()

    def test_stable_presence_captures_one_still_then_closes_camera(self):
        hardware = []
        class Camera:
            def __enter__(self): hardware.append('open'); return self
            def __exit__(self, *_): hardware.append('close')
            def capture(self, *_, **kwargs):
                hardware.append('still')
                self.video_port = kwargs['use_video_port']
                if self.video_port:
                    raise AssertionError('Must use still capture')
        events, classifier = self.run_controller(40, Mock(PiCamera=Camera))
        self.assertEqual(hardware, ['open', 'still', 'close'])
        self.assertEqual(events, ['save'])
        classifier.classify.assert_called_once()


if __name__ == '__main__':
    main()
