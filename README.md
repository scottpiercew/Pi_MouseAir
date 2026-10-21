# Pi_MouseAir

Motion-triggered, on-device cat detection and cat-toy control for Raspberry Pi.

`MouseAirMotion.py` detects camera movement, saves a picture, locates cats
locally with a quantized EfficientDet-Lite0 object detector, and launches a loaded toy when
the cat score meets the configured threshold. No cloud inference is used.

## Current controller

- Headless 1280×960 capture with configurable backlight exposure settings.
- Full-frame cat bounding boxes, including one or several cats in the image.
- Configurable confidence threshold (40% detection score by default).
- Five-second retries for continuing movement and a 30-second launch cooldown.
- Toy-present sensor interlock and motor shutdown after a launch error.
- Armed mode logs `Toy loaded: YES/NO` at startup and whenever it changes;
  a status line repeats the current state every 10 seconds.
- On cat acceptance, all eight LEDs show red for 1 second, yellow for 1 second,
  green for 3 seconds, then turn off, alongside the existing launch sequence.
  The lights also run when no toy is loaded; the sensor still prevents launch.
- At most 20 local motion pictures, excluded from Git.

## Hardware and software

The tested system is a Raspberry Pi 3 Model B with a CSI camera, PCA9685 motor
controller at I2C address `0x41`, the existing launch motors/servo assembly, and
a toy-present GPIO sensor. This checkout targets the legacy `picamera` stack
on 32-bit Raspberry Pi OS/Raspbian with Python 3.7. It has not been ported to
the modern Picamera2/libcamera stack.

Runtime dependencies include Python 3, NumPy, OpenCV, PiCamera, RPi.GPIO,
SMBus, and TFLite Runtime. TFLite Runtime 2.11.0 was tested on ARMv7/Python 3.7.
Original legacy programs have additional dependencies, including TensorFlow,
NeoPixel, Flask, and APScheduler; they are not the recommended entry point.

## Configure and build the model

Copy the sanitized example to a private local configuration:

```sh
cp config.example.py config.py
```

Review GPIO pins and the servo/motor calibration in `state.py` for your
hardware. An existing private `conflocal.py` takes precedence over `config.py`.

Model binaries and training/camera images are intentionally excluded. Fetch
our pinned official TensorFlow COCO detector (4.6 MB):

```sh
python3 download_cat_detector.py
```

The downloader checks SHA-256 before installation. Keep the TFLite file and
`EfficientDetLite0-cat.json` together. This detector uses raw RGB uint8 input
at 320×320 and identifies the COCO `cat` label, irrespective of coat color.
Either cat can meet the threshold; two weak scores are never added together.
Clear full-frame detections took about 0.55 seconds on the tested Pi 3;
uncertain frames get a second center view and take about 1.1 seconds. Downloading
requires internet; runtime inference does not.

The previous ImageNet classifiers can still be built on a Mac with
`build_imagenet_model.py` (TensorFlow 2.16.2 and tf-keras 2.16.0). Select an
older model explicitly with `--model EfficientNetB0-cat.tflite` if needed.

## Run and test

```sh
cd /home/pi/Pi_MouseAir
python3 -m unittest test_motion_policy test_race_lights test_cat_detector -v
python3 -u MouseAirMotion.py --dry-run
```

Dry-run logs decisions without importing the motor driver. Once the camera,
sensor, and calibration are checked, enable the launch sequence explicitly:

```sh
python3 -u MouseAirMotion.py --armed --confidence 0.40 --cooldown 30
```

Stop with Ctrl-C. Do not run another hardware controller or motor test script
concurrently. See [camera and confidence tuning](MOTION_README.md).

The RGBW stick uses the configured `PixelPin` (GPIO 21 on the tested Pi).
The installed legacy `neopixel`/`rpi-ws281x` driver needs root for DMA access;
the controller starts an LED-only helper with `sudo -n /usr/bin/python3`.
The Pi user therefore needs noninteractive sudo for that helper. Motors and
classification remain in the ordinary user process. An LED startup failure
cancels the launch. Dry-run and image classification leave the LEDs off.

## Local files and publishing

`.gitignore` excludes local configuration, credentials, captures, training
images, runtime state files, model binaries, environments, logs, databases, and
editor recovery files. Ignored files remain on the device. Do not add them
with `git add -f`; use `config.example.py` to document new settings.

## Attribution

The original hardware drivers and MouseAir controller are derived from
[SwitchDoc Labs SDL_Pi_MouseAir](https://github.com/switchdoclabs/SDL_Pi_MouseAir)
(V012, 2019). Original notices are retained in source files. The bundled
Adafruit driver retains its own [license](Adafruit_Python_PCA9685/LICENSE).
The motion-triggered controller, model build tooling, and accompanying tests
extend that original project. No new license is asserted for upstream code.

Default detector and pretrained weights:
[TensorFlow Raspberry Pi object detection example](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi).
The model URL and pinned checksum are recorded in `EfficientDetLite0-cat.json`.
Legacy classifier source: [Keras EfficientNet](https://keras.io/2/api/applications/efficientnet/).
