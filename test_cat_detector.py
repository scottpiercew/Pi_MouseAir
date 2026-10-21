import unittest
from cat_detector import CatObjectDetector, cat_detections, map_boxes
from motion_policy import MotionPolicy


class CatDetectorTests(unittest.TestCase):
    def detector(self, results):
        from unittest.mock import Mock
        detector = CatObjectDetector.__new__(CatObjectDetector)
        detector.default_confidence = .4
        detector.crop = [.15, .1, .85, .95]
        detector._infer = Mock(side_effect=results)
        return detector

    def test_clear_detection_does_not_need_second_inference(self):
        detector = self.detector([[{'score': .8, 'box': [0, 0, 1, 1]}]])
        self.assertEqual(detector.classify(None), .8)
        detector._infer.assert_called_once_with(None)
        self.assertEqual(detector.last_view, 'full')

    def test_uncertain_frame_can_be_recovered_in_center_view(self):
        from unittest.mock import MagicMock
        detector = self.detector([[], [{'score': .7, 'box': [0, 0, 1, 1]}]])
        image = MagicMock(shape=(960, 1280, 3))
        self.assertEqual(detector.classify(image), .7)
        self.assertEqual(detector.last_view, 'center')
        for actual, expected in zip(detector.last_detections[0]['box'], [.1, .15, .95, .85]):
            self.assertAlmostEqual(actual, expected)

    def test_center_boxes_are_mapped_back_to_original_picture(self):
        mapped = map_boxes([{'score': .7, 'box': [0, 0, 1, 1]}], [.15, .1, .85, .95])
        for actual, expected in zip(mapped[0]['box'], [.1, .15, .95, .85]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(mapped[0]['score'], .7)

    def test_one_or_two_cats_can_independently_qualify(self):
        cats = cat_detections([[.1, .1, .5, .4], [.2, .6, .7, .9]],
                              [16, 16], [.72, .65], 2)
        self.assertEqual(len(cats), 2)
        self.assertTrue(MotionPolicy(confidence=.4).accepts_cat(cats[0]['score'], 0))

    def test_dog_and_white_objects_are_not_cat_labels(self):
        self.assertEqual(cat_detections([[0, 0, 1, 1]] * 2, [17, 81], [.99, .98], 2), [])

    def test_two_weak_cats_do_not_combine_into_a_positive(self):
        cats = cat_detections([[0, 0, .4, .4]] * 2, [16, 16], [.3, .3], 2)
        self.assertFalse(MotionPolicy(confidence=.4).accepts_cat(cats[0]['score'], 0))

    def test_clips_border_boxes_and_rejects_zero_area(self):
        cats = cat_detections([[-.01, 0, 1.02, 1], [0, 0, 0, 0]], [16, 16], [.8, .9], 2)
        self.assertEqual(cats, [{'score': .8, 'box': [0, 0, 1, 1]}])

    def test_invalid_output_fails_closed(self):
        cases = [([[0, 0, 1, 1]], [16], [float('nan')], 1),
                 ([[0, 0, 1, 1]], [16.5], [.9], 1),
                 ([[0, 0, float('inf'), 1]], [16], [.9], 1),
                 ([[0, 0, 1, 1]], [16], [.9], 2),
                 ([[0, 0, 1, 1]], [16], [.9], float('nan'))]
        for args in cases:
            with self.assertRaises(ValueError):
                cat_detections(*args)

    def test_forty_percent_threshold_and_invalid_thresholds(self):
        policy = MotionPolicy(confidence=.4)
        self.assertTrue(policy.accepts_cat(.4, 0))
        self.assertFalse(policy.accepts_cat(.399, 0))
        for threshold in (0, -1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                MotionPolicy(confidence=threshold)


if __name__ == '__main__':
    unittest.main()
