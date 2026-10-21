# Pi_MouseAir

Motion-triggered, on-device cat detection and cat-toy control for Raspberry Pi.

`MouseAirMotion.py` detects camera movement, saves a picture, classifies it
locally with a quantized EfficientNet-B0 model, and launches a loaded toy when
the cat score meets the configured threshold. No cloud inference is used.

## Current controller

- Headless 1280×960 capture with configurable backlight exposure settings.
- Full-frame and perch-view classification using domestic-cat ImageNet labels.
- Configurable confidence threshold (60% by default).
- Five-second retries for continuing movement and a 30-second launch cooldown.
- Toy-present sensor interlock and motor shutdown after a launch error.
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

Model binaries and training/camera images are intentionally excluded. On a
Mac or another suitable build computer with Python 3.12:

```sh
python3 -m venv .model-build
.model-build/bin/pip install tensorflow==2.16.2 tf-keras==2.16.0
.model-build/bin/python build_imagenet_model.py
```

This downloads official Keras ImageNet weights and generates
`EfficientNetB0-cat.tflite` plus matching JSON metadata. Copy the pair into the
Pi repository. The model is about 5.8 MB; two-view inference took about 1.2
seconds per picture on the tested Pi 3. Model building requires internet;
runtime image classification does not.

## Run and test

```sh
cd /home/pi/Pi_MouseAir
python3 -m unittest test_motion_policy test_race_lights -v
python3 -u MouseAirMotion.py --dry-run
```

Dry-run logs decisions without importing the motor driver. Once the camera,
sensor, and calibration are checked, enable the launch sequence explicitly:

```sh
python3 -u MouseAirMotion.py --armed --confidence 0.60 --cooldown 30
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

Model architecture and pretrained-weight source:
[Keras EfficientNet](https://keras.io/2/api/applications/efficientnet/).
