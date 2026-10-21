#!/usr/bin/env python3
"""Motion -> saved picture -> on-device cat inference -> optional toy launch."""
import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from motion_policy import MotionPolicy, launch_sequence, cat_probability

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('MouseAirMotion')


class CatClassifier:
    def __init__(self, model, cat_index=0):
        import numpy as np
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        self.np = np
        self.interpreter = Interpreter(model_path=str(model), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.cat_index = cat_index
        metadata_path = Path(model).with_suffix('.json')
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            self.size = int(metadata['input_size'])
            self.normalization = metadata['input_normalization']
            self.cat_indices = metadata['cat_indices']
            output_classes = int(metadata['output_classes'])
            self.name = metadata['name']
            self.default_confidence = float(metadata.get('confidence', 0.60))
            self.crop_box = metadata.get('crop_box')
        else:
            self.size = 150
            self.normalization = 'zero_to_one'
            self.cat_indices = [cat_index]
            output_classes = 2
            self.name = 'Original CatNotCat'
            self.default_confidence = 0.90
            self.crop_box = None
        if list(self.input['shape']) != [1, self.size, self.size, 3]:
            raise ValueError('Model input does not match its metadata')
        if list(self.output['shape']) != [1, output_classes]:
            raise ValueError('Model output does not match its metadata')
        if self.normalization not in ('zero_to_one', 'minus_one_to_one', 'raw_zero_to_255'):
            raise ValueError('Unknown model normalization')
        cat_probability([0.0] * output_classes, self.cat_indices)
        if self.crop_box is not None:
            if len(self.crop_box) != 4 or not (0 <= self.crop_box[0] < self.crop_box[2] <= 1 and 0 <= self.crop_box[1] < self.crop_box[3] <= 1):
                raise ValueError('Invalid normalized classifier crop')

    def classify(self, bgr):
        views = [('full', bgr)]
        if self.crop_box is not None:
            height, width = bgr.shape[:2]
            left, top, right, bottom = self.crop_box
            crop = bgr[int(top*height):int(bottom*height), int(left*width):int(right*width)]
            if crop.size:
                views.append(('perch', crop))
        score, self.last_view = max((self._classify_one(frame), name) for name, frame in views)
        return score

    def _classify_one(self, bgr):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size)).astype(self.np.float32)
        if self.normalization == 'minus_one_to_one':
            rgb = rgb / 127.5 - 1.0
        elif self.normalization == 'zero_to_one':
            rgb = rgb / 255.0
        data = self.np.expand_dims(rgb, 0)
        if self.input['dtype'] != self.np.float32:
            scale, zero = self.input['quantization']
            if scale <= 0:
                raise ValueError('Invalid model input quantization')
            limits = self.np.iinfo(self.input['dtype'])
            data = self.np.clip(self.np.rint(data / scale + zero), limits.min, limits.max).astype(self.input['dtype'])
        self.interpreter.set_tensor(self.input['index'], data)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output['index'])[0]
        if self.output['dtype'] != self.np.float32:
            scale, zero = self.output['quantization']
            scores = (scores.astype(self.np.float32) - zero) * scale
        if not self.np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('Invalid classifier probabilities; refusing launch')
        return cat_probability(scores.tolist(), self.cat_indices)


class MotionDetector:
    def __init__(self, area=800):
        self.previous = None
        self.area = area

    def detect(self, frame):
        import cv2
        gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        previous, self.previous = self.previous, gray
        if previous is None:
            return False
        delta = cv2.absdiff(previous, gray)
        mask = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        return any(cv2.contourArea(c) >= self.area for c in contours)


class ToyLauncher:
    def __init__(self):
        self.motors = None
        self.detector = None
        self.lights = None

    def launch(self):
        if self.lights is None:
            from race_lights import RaceLights
            self.lights = RaceLights()
        self.lights.start()
        if self.detector is None:
            import MouseDetector
            self.detector = MouseDetector
            self.detector.SetupMouseDetector()
        if self.detector.CheckMousePresent() != 0:
            LOG.warning('Cat accepted but no toy is loaded; launch skipped')
            return False
        if self.motors is None:
            import Motors
            self.motors = Motors
        launch_sequence(self.motors)
        return True

    def stop(self):
        try:
            if self.motors is not None:
                self.motors.launchMotorsOff()
                self.motors.pwm.set_pwm(0, 0)
        finally:
            if self.lights is not None:
                self.lights.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'EfficientNetB0-cat.tflite')
    parser.add_argument('--confidence', type=float, default=None, help='Cat threshold; default 0.60 for ImageNet models, 0.90 for the original model')
    parser.add_argument('--cat-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--cooldown', type=float, default=30.0)
    parser.add_argument('--retry-seconds', type=float, default=5.0, help='Recheck continuing movement after a rejected image')
    parser.add_argument('--motion-area', type=int, default=800)
    parser.add_argument('--rotation', type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--exposure-compensation', type=int, default=6, help='Exposure adjustment in sixths of a stop; 6 is +1 stop')
    parser.add_argument('--meter-mode', choices=('average', 'spot', 'backlit', 'matrix'), default='backlit')
    parser.add_argument('--exposure-mode', choices=('auto', 'backlight', 'sports', 'night'), default='backlight')
    parser.add_argument('--drc', choices=('off', 'low', 'medium', 'high'), default='high')
    parser.add_argument('--brightness', type=int, default=55)
    parser.add_argument('--image', type=Path, help='Classify an image and exit; never operates motors')
    parser.add_argument('--seconds', type=float, default=0, help='Stop after this duration; 0 runs until Ctrl-C')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--armed', action='store_true', help='Enable real toy launches after cat detection')
    mode.add_argument('--dry-run', action='store_true', help='Log decisions without importing motor drivers (default)')
    args = parser.parse_args()
    if args.motion_area <= 0 or args.seconds < 0:
        parser.error('Motion area must be positive; seconds cannot be negative')
    if args.width < 320 or args.width > 1920 or args.height < 240 or args.height > 1440 or args.width % 32 or args.height % 16:
        parser.error('Use width 320..1920 divisible by 32 and height 240..1440 divisible by 16')
    if not -25 <= args.exposure_compensation <= 25 or not 0 <= args.brightness <= 100:
        parser.error('Exposure compensation must be -25..25; brightness must be 0..100')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.chdir(str(ROOT))
    if not args.model.is_file():
        parser.error('Model file missing; deploy the model and matching JSON metadata first.')
    import cv2
    classifier = CatClassifier(args.model, args.cat_index)
    if args.confidence is None:
        args.confidence = classifier.default_confidence
    policy = MotionPolicy(args.confidence, args.cooldown, retry_seconds=args.retry_seconds)
    if args.image:
        image = cv2.imread(str(args.image.resolve()))
        if image is None:
            parser.error('Cannot read image')
        before = time.monotonic()
        score = classifier.classify(image)
        print(json.dumps({'image': str(args.image), 'cat_confidence': score,
                          'accepted': policy.accepts_cat(score, time.monotonic()),
                          'inference_seconds': time.monotonic() - before,
                          'best_view': classifier.last_view, 'motors_enabled': False}))
        return
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    lock = open('/tmp/mouseair-motion-{}.lock'.format(os.getuid()), 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another motion controller is already running')
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    snapshots = ROOT / 'motion-captures'
    snapshots.mkdir(exist_ok=True)
    launcher = ToyLauncher()
    detector = MotionDetector(args.motion_area)
    start = time.monotonic()
    frame_count = 0
    capture_count = 0
    LOG.info('Mode=%s threshold=%.0f%% cooldown=%.0fs model=%s model_bytes=%s',
             'ARMED' if args.armed else 'DRY RUN', args.confidence * 100, args.cooldown, classifier.name, args.model.stat().st_size)
    try:
        with PiCamera() as camera:
            camera.resolution = (args.width, args.height)
            camera.framerate = 10
            camera.rotation = args.rotation
            camera.exposure_mode = args.exposure_mode
            camera.meter_mode = args.meter_mode
            camera.exposure_compensation = args.exposure_compensation
            camera.drc_strength = args.drc
            camera.brightness = args.brightness
            camera.awb_mode = 'auto'
            LOG.info('Camera=%sx%s metering=%s exposure=%s EV=%+.1f DRC=%s brightness=%s',
                     args.width, args.height, args.meter_mode, args.exposure_mode,
                     args.exposure_compensation / 6.0, args.drc, args.brightness)
            with PiRGBArray(camera, size=(args.width, args.height)) as raw:
                time.sleep(4)
                for index, captured in enumerate(camera.capture_continuous(raw, format='bgr', use_video_port=True)):
                    frame_count += 1
                    frame = captured.array.copy()
                    raw.truncate(0)
                    now = time.monotonic()
                    if stopping or (args.seconds and now - start >= args.seconds):
                        break
                    moving = detector.detect(frame)
                    if index < 20 or not policy.observe(moving, now):
                        continue
                    filename = snapshots / ('motion-{}.jpg'.format(time.time_ns()))
                    temporary = filename.with_name(filename.stem + '.tmp.jpg')
                    if not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError('Could not save motion snapshot')
                    temporary.replace(filename)
                    capture_count += 1
                    before = time.monotonic()
                    score = classifier.classify(frame)
                    accepted = policy.accepts_cat(score, now)
                    LOG.info('Motion captured=%s cat=%.1f%% accepted=%s view=%s inference=%.3fs', filename.name, score*100, accepted, classifier.last_view, time.monotonic()-before)
                    if accepted:
                        if args.armed:
                            if launcher.launch():
                                LOG.info('Toy launched')
                                policy.launched(time.monotonic())
                        else:
                            LOG.info('DRY RUN: would launch if a toy is loaded')
                            policy.launched(time.monotonic())
                    for old in sorted(snapshots.glob('motion-*.jpg'))[:-20]:
                        old.unlink()
    except KeyboardInterrupt:
        LOG.info('Stopping')
    finally:
        launcher.stop()
        lock.close()
        LOG.info('Stopped after %s camera frames and %s motion captures', frame_count, capture_count)


if __name__ == '__main__':
    main()
