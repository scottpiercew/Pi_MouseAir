"""Fetch the official TensorFlow detector with a pinned SHA-256 checksum."""
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / 'EfficientDetLite0-cat.json').read_text())
    target = root / 'EfficientDetLite0-cat.tflite'
    with urllib.request.urlopen(metadata['source_url'], timeout=60) as response:
        blob = response.read()
    if hashlib.sha256(blob).hexdigest() != metadata['sha256']:
        raise ValueError('Model checksum mismatch; refusing installation')
    temporary = target.with_suffix('.tmp')
    temporary.write_bytes(blob)
    temporary.replace(target)
    print('Downloaded verified detector: {} ({} bytes)'.format(target, len(blob)))


if __name__ == '__main__':
    main()
