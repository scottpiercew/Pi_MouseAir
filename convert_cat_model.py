"""Convert the existing local Keras model to a small, quantized TFLite model."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import json
from pathlib import Path
import tensorflow as tf

root = Path(__file__).resolve().parent
source = root / 'CatNotCat.trained'
target = root / 'CatNotCat.tflite'
if hasattr(tf.lite.TFLiteConverter, 'from_keras_model_file'):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(str(source))
else:
    model = tf.keras.models.load_model(str(source), compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
if hasattr(tf.lite, 'Optimize') and hasattr(tf.lite.Optimize, 'DEFAULT'):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
else:
    converter.post_training_quantize = True
model = converter.convert()
temporary = target.with_suffix('.tflite.tmp')
temporary.write_bytes(model)
temporary.replace(target)
print(json.dumps({'source_bytes': source.stat().st_size,
                  'tflite_bytes': target.stat().st_size,
                  'output': str(target)}))
