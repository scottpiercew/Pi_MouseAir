"""On-device cat localization, including several cats in one picture."""
import hashlib
import json
import math
from pathlib import Path
import zipfile


def cat_detections(boxes, classes, scores, count, cat_class=16):
    """Validate detector output before considering any launch candidate."""
    if not math.isfinite(count) or count != int(count) or not 0 <= count <= min(len(boxes), len(classes), len(scores)):
        raise ValueError('Invalid detection count; refusing launch')
    cats = []
    for i in range(int(count)):
        score, category, box = scores[i], classes[i], boxes[i]
        if not math.isfinite(score) or not 0 <= score <= 1 or not math.isfinite(category) or category != int(category) or not 0 <= category < 90:
            raise ValueError('Invalid detector probabilities/classes; refusing launch')
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError('Invalid detector box; refusing launch')
        top, left, bottom, right = [min(1.0, max(0.0, v)) for v in box]
        if category == cat_class and score > 0 and bottom > top and right > left:
            cats.append({'score': float(score), 'box': [top, left, bottom, right]})
    return sorted(cats, key=lambda item: item['score'], reverse=True)


def map_boxes(cats, crop):
    left, top, right, bottom = crop
    return [{'score': cat['score'], 'box': [top + cat['box'][0] * (bottom - top),
                                           left + cat['box'][1] * (right - left),
                                           top + cat['box'][2] * (bottom - top),
                                           left + cat['box'][3] * (right - left)]}
            for cat in cats]


class CatObjectDetector:
    def __init__(self, model):
        import numpy as np
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
        model = Path(model)
        meta = json.loads(model.with_suffix('.json').read_text())
        if meta['task'] != 'object_detection' or hashlib.sha256(model.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Detector model does not match its verified metadata')
        with zipfile.ZipFile(str(model)) as archive:
            labels = archive.read('labelmap.txt').decode().splitlines()
        self.cat_class = meta['cat_class']
        if labels[self.cat_class] != 'cat':
            raise ValueError('Detector cat label is incorrect')
        self.np = np
        self.interpreter = Interpreter(model_path=str(model), num_threads=2)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        outputs = {d['name']: d for d in self.interpreter.get_output_details()}
        self.outputs = [outputs[name] for name in meta['output_names']]
        self.size = meta['input_size']
        if list(self.input['shape']) != [1, self.size, self.size, 3] or self.input['dtype'] != np.uint8:
            raise ValueError('Detector requires raw uint8 RGB input')
        if [list(d['shape']) for d in self.outputs] != [[1, 25, 4], [1, 25], [1, 25], [1]]:
            raise ValueError('Unexpected detector output layout')
        self.name = meta['name']
        self.default_confidence = float(meta['confidence'])
        self.crop = meta.get('fallback_crop')
        if self.crop is not None and (len(self.crop) != 4 or not all(math.isfinite(v) for v in self.crop) or
                                     not (0 <= self.crop[0] < self.crop[2] <= 1 and 0 <= self.crop[1] < self.crop[3] <= 1)):
            raise ValueError('Invalid detector fallback crop')
        self.last_view = 'full'
        self.last_detections = []

    def classify(self, bgr):
        self.last_view = 'full'
        cats = self._infer(bgr)
        score = cats[0]['score'] if cats else 0.0
        # Only difficult frames need a second, generic center view. This
        # recovers backlit/partial cats without slowing clear full-frame hits.
        if score < self.default_confidence and self.crop is not None:
            height, width = bgr.shape[:2]
            left, top, right, bottom = self.crop
            frame = bgr[int(top * height):int(bottom * height), int(left * width):int(right * width)]
            if frame.size:
                center = map_boxes(self._infer(frame), self.crop)
                if center and center[0]['score'] > score:
                    cats, score, self.last_view = center, center[0]['score'], 'center'
        self.last_detections = cats
        # Either cat can qualify. Never add two weak scores to force a launch.
        return score

    def _infer(self, bgr):
        import cv2
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size))
        self.interpreter.set_tensor(self.input['index'], self.np.expand_dims(rgb, 0))
        self.interpreter.invoke()
        boxes, classes, scores, count = [self.interpreter.get_tensor(d['index']) for d in self.outputs]
        return cat_detections(boxes[0].tolist(), classes[0].tolist(),
                              scores[0].tolist(), float(count[0]), self.cat_class)
