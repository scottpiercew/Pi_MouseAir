# Motion-triggered on-device cat detection

Entry point: `/home/pi/Pi_MouseAir/MouseAirMotion.py`.
Camera movement triggers a saved picture and local cat detection. The
ultrasonic experiment remains on its separate feature branch.

## Current settings

- Camera: 1280×960, 10 FPS, rotation 180 degrees.
- Backlight exposure, backlit metering, +1 stop compensation, high DRC,
  brightness 55, automatic white balance.
- EfficientDet-Lite0 COCO object detector: 4,563,519 bytes, 320×320 RGB input.
- A cat detection score of at least 40% qualifies. Either cat can qualify in a
  multi-cat frame; scores from different cats are never added together.
- Retry continuing movement every five seconds; 30-second launch cooldown.
- Original toy-present sensor interlock, motor shutdown on errors, and servo
  calibration are retained.
- Race lights run alongside launch preparation: red 1s, yellow 1s, green 3s,
  then off. Lights also run after acceptance when no toy is loaded.
- Armed mode logs toy-loaded status at startup, when it changes, and every 10s.

## Start and stop

```sh
cd /home/pi/Pi_MouseAir
python3 download_cat_detector.py  # only needed when the model is absent
python3 -u MouseAirMotion.py --dry-run
python3 -u MouseAirMotion.py --armed --confidence 0.40 --cooldown 30
```

Stop with Ctrl-C. Never run another hardware controller or motor test at the
same time. Classification runs on the Pi without a Mac, LLM, or cloud API.
Only the newest 20 pictures are retained in the ignored `motion-captures/`
directory. Neither pictures nor model binaries are published in Git.

## What changed and how it was checked

The earlier whole-scene ImageNet classifier often missed cats occupying a
small part of the frame, close partial cats, and frames with several cats.
The object detector locates individual cats and reports their bounding boxes.
It is not restricted to particular coat colors or breeds.

Replaying five recent cat-containing captures accepted four at the new 40%
threshold, compared with none at the previous classifier's 60% threshold.
The remaining frame showed only a blurred tail. A further two-cat frame
produced two qualifying cat detections. Dog and non-cat reference images
did not produce a qualifying cat detection. These selected examples are
regression checks, not an estimate of real-world accuracy or probability.

On the tested Pi 3, detector inference took about 0.55 seconds per picture,
compared with about 1.2 seconds for the previous full/perch classification.
Uncertain frames get a generic center view, taking about 1.1 seconds in total.
This also recovered earlier backlit cat samples. Boxes from that view are
mapped back to the original picture. No coat-color heuristics were added.

The detector is the official TensorFlow Raspberry Pi example model. Its URL,
SHA-256, label index, output layout, and default threshold are recorded in
`EfficientDetLite0-cat.json`. The downloader and runtime verify its checksum;
the runtime also validates that the label is `cat` and rejects invalid outputs.

```sh
python3 -m unittest test_motion_policy test_race_lights test_cat_detector -v
python3 MouseAirMotion.py --image motion-captures/IMAGE.jpg
python3 -u MouseAirMotion.py --dry-run --seconds 20
```

Image mode never operates motors, even with `--armed`. Its JSON includes
qualifying cats, scores, and normalized boxes `[top,left,bottom,right]`.
The live terminal log includes `cats=N` beside the highest cat score.

## Tuning and rollback

Increase `--confidence` if false detections occur. Detector scores are not
calibrated probabilities, and lowering the threshold can increase false
launches. The toy sensor and cooldown remain active at every threshold.

Exposure compensation is in sixths of a stop: 6 is +1 stop, 12 is +2 stops.
More exposure may brighten the face but blur moving cats and clip the
background. Front lighting or moving the camera away from a backlit view can
help. `--motion-area 1200` reduces movement sensitivity;
`--retry-seconds 10` reduces how often ongoing movement is rechecked.

The previous models remain on the existing Pi for comparison:

```sh
python3 MouseAirMotion.py --model EfficientNetB0-cat.tflite --confidence 0.60 --dry-run
python3 MouseAirMotion.py --model CatNotCat.tflite --confidence 0.90 --dry-run
```

Stop the current controller before running a different mode/model. Restart
after changing source, model metadata, or CLI settings. VS Code's SSH FS
workspace edits the Pi files directly; saving source does not restart it.
