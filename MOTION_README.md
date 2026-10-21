# Motion-triggered on-device cat detection

Entry point: `/home/pi/Pi_MouseAir/MouseAirMotion.py`.
The original controller, configuration, and servo calibration are preserved.

## Current settings

- Camera: **1280×960**, 10 FPS, 180-degree rotation.
- Backlight exposure mode and backlit metering; **+1 stop** compensation,
  high dynamic-range compression, brightness 55, automatic white balance.
- Classifier: **EfficientNet-B0**, quantized TFLite, **5,819,240 bytes**.
- Launch threshold: **60%**, based on the summed five domestic-cat class scores.
- Check both the full picture and a tighter view of the lower central perch.
  Keep the higher cat score; the original full picture is saved unchanged.
- Retry ongoing movement every **5 seconds** if the previous picture is rejected.
- Retain the original toy-present sensor check and a **30-second launch cooldown**.
- Stop motors in `finally` if a launch step fails.

The camera settings are documented by [PiCamera](https://picamera.readthedocs.io/en/release-1.13/api_camera.html).
EfficientNet uses raw RGB pixels from 0..255, with normalization inside the
model, as described in [Keras documentation](https://keras.io/2/api/applications/efficientnet/).
The full frame and perch crop each use the model's 224×224 input.

## Run

```sh
cd /home/pi/Pi_MouseAir
# Camera and inference only; motors are never imported:
python3 -u MouseAirMotion.py --dry-run
# Enable launches:
python3 -u MouseAirMotion.py --armed --confidence 0.60 --cooldown 30
```

Stop with **Ctrl-C** in the Pi terminal. Do not run the old controller or motor
test scripts concurrently. This controller prevents another instance of itself
for the same user. No Mac or internet is required after the program starts;
the separate LLM administration agent is not involved in image classification.

Only the newest 20 motion pictures are retained in `motion-captures/`. Images
are written atomically so an interrupted write does not expose a partial JPEG.

## Tuning

```sh
# Be more selective if false triggers occur:
python3 -u MouseAirMotion.py --armed --confidence 0.75
# Brighten a very dark cat more (+2 stops, more background clipping/blur):
python3 -u MouseAirMotion.py --armed --exposure-compensation 12
# Restore ordinary metering/exposure for a less backlit location:
python3 -u MouseAirMotion.py --dry-run --meter-mode average --exposure-mode auto --exposure-compensation 0 --drc off --brightness 50
# Test an existing image; never operates motors, even with --armed:
python3 MouseAirMotion.py --image motion-captures/IMAGE.jpg
# Test the live camera for a bounded time:
python3 -u MouseAirMotion.py --dry-run --seconds 20
```

Exposure compensation is in sixths of a stop: 6 is +1 stop and 12 is +2 stops.
The +2-stop sample revealed the face but washed out more of the bright window,
so +1 stop is the default. Increasing exposure can blur moving subjects. A lamp
on the cat-facing side, or moving the camera so the window is not behind the
cat, can improve the lighting beyond software adjustments.

`--motion-area 1200` reduces motion sensitivity. `--retry-seconds 10` reduces
how often continuing movement is reclassified. The camera dimensions are
configurable with `--width` and `--height` within the validated CLI limits.
For another camera position, edit or remove `crop_box` in
`EfficientNetB0-cat.json`: its normalized bounds are left, top, right, bottom.
Restart the controller after changing code, model metadata, or settings.

The 60% score is not a calibrated probability. Lower thresholds accept more
borderline images and can increase false triggers. Cropped-view scoring also
makes recognition more permissive. Review live results and adjust as needed.

## Models and validation

The earlier small model gave one real backlit cat capture only **1.04%** cat
confidence. Simply reducing a 90% threshold slightly could not address that.
MobileNetV2 was also evaluated; EfficientNet-B0 handled the difficult side-on
sample better and is the current default.

Measured with the final model and full/perch views on the Raspberry Pi:

| Test image | Cat score | Decision at 60% |
|---|---:|---|
| Actual cat, +1-stop backlight sample | 67.92% | Accepted |
| Bundled dog sample | 0.13% | Rejected |

Two-view inference took about **1.2 seconds** per picture on the Pi. These are
selected samples, not an accuracy benchmark. Pose, light, focus, and background
can still cause missed cats or incorrect classifications.

Fourteen logic tests passed on the Pi. They cover confidence validation,
domestic-cat class aggregation, rejecting non-cat classes, debounce, retries,
cooldown, launch ordering, and motor shutdown on partial-start/servo failures.
Live-camera dry-runs verified the camera settings without moving the motors.

The previous models remain available:

```sh
# Faster but weaker on the real backlit examples:
python3 MouseAirMotion.py --model MobileNetV2-cat.tflite --dry-run
# Original two-class model; retain its higher threshold:
python3 MouseAirMotion.py --model CatNotCat.tflite --confidence 0.90 --dry-run
```

TFLite Runtime 2.11.0 runs inference on this 32-bit Pi without loading full
TensorFlow. Models and matching JSON metadata are kept on the Pi.
`build_imagenet_model.py` rebuilds EfficientNet-B0 or MobileNetV2 on a Mac with
TensorFlow 2.16.2 and tf-keras 2.16.0. It downloads official Keras ImageNet
weights and uses dynamic-range quantization. Model building is not needed
when starting the controller.

## Edit, deploy, and recover

Edit the new source files in this Mac folder, then run `./deploy`. It updates
only the listed files in `/home/pi/Pi_MouseAir`; original config and
calibration files remain untouched. Restart the controller afterward.

VS Code mounts `/home/pi` through SSH FS. Saving there updates the Pi directly.
Use the Pi SSH terminal to run commands.

