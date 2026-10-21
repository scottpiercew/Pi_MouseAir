"""Build an ImageNet classifier on the Mac, then deploy its TFLite + JSON pair.

Build environment: TensorFlow 2.16.2 and tf-keras 2.16.0.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KERAS_HOME', str(Path(tempfile.gettempdir()) / 'mouseair-keras-cache'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=('efficientnet-b0', 'mobilenet-v2'), default='efficientnet-b0')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    import tensorflow as tf
    if args.architecture == 'efficientnet-b0':
        model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        name, prefix, normalization = 'EfficientNetB0 ImageNet', 'EfficientNetB0-cat', 'raw_zero_to_255'
    else:
        model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        name, prefix, normalization = 'MobileNetV2 ImageNet', 'MobileNetV2-cat', 'minus_one_to_one'
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = converter.convert()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / (prefix + '.tflite')
    target.write_bytes(blob)
    metadata = {
        'name': name, 'input_size': 224, 'input_normalization': normalization,
        'output_classes': 1000, 'cat_indices': [281, 282, 283, 284, 285],
        'cat_labels': {'281': 'tabby', '282': 'tiger_cat', '283': 'Persian_cat',
                       '284': 'Siamese_cat', '285': 'Egyptian_cat'},
        'confidence': 0.60}
    if args.architecture == 'efficientnet-b0':
        metadata['crop_box'] = [0.125, 0.25, 0.875, 1.0]
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'model': str(target), 'bytes': len(blob)}))


if __name__ == '__main__':
    main()
