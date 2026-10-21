"""Five-second RGBW race signal; only the LED helper needs root access."""
import logging
from pathlib import Path
import select
import subprocess
import sys

LOG = logging.getLogger('MouseAirMotion')
STAGES = (((255, 0, 0), 1.0), ((255, 255, 0), 1.0), ((0, 255, 0), 3.0))


def race_sequence(fill, wait):
    """Wait returns true to cancel. Always clear all pixels on exit."""
    try:
        for color, seconds in STAGES:
            fill(*color)
            if wait(seconds):
                break
    finally:
        fill(0, 0, 0)


class RaceLights:
    def __init__(self):
        self.process = None

    def start(self):
        # Repeated detections cannot restart or overlap an active race signal.
        if self.process is not None and self.process.poll() is None:
            return
        if self.process is not None:
            self.stop()
        self.process = subprocess.Popen(
            ['sudo', '-n', '/usr/bin/python3', '-u', str(Path(__file__).resolve()), '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        ready, _, _ = select.select([self.process.stdout], [], [], 4)
        if not ready or self.process.stdout.readline().strip() != 'RED':
            self.stop()
            raise RuntimeError('LED helper could not start; launch cancelled')
        LOG.info('Race lights: red 1s -> yellow 1s -> green 3s -> off')

    def stop(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        # EOF interrupts the helper's wait; it clears the LEDs in finally.
        process.stdin.close()
        try:
            process.wait(timeout=7)
        finally:
            process.stdout.close()
        if process.returncode:
            LOG.error('LED helper exited with status %s', process.returncode)


def worker():
    import signal
    from neopixel import Adafruit_NeoPixel, Color, ws
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        import conflocal as config
    except ImportError:
        import config
    if not config.RGBWLEDS:
        raise RuntimeError('RGBW LEDs are disabled in the local configuration')
    strip = Adafruit_NeoPixel(8, config.PixelPin, 800000, 10, False, 96, 0,
                             ws.SK6812_STRIP_RGBW)
    strip.begin()
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    first = True
    def fill(red, green, blue):
        nonlocal first
        # This SwitchDoc stick uses the same R/G swap as pixelDriver.RGBColor.
        color = Color(green, red, blue, 0)
        for pixel in range(strip.numPixels()):
            strip.setPixelColor(pixel, color)
        strip.show()
        LOG.info('LEDs %s', {(255, 0, 0): 'red', (255, 255, 0): 'yellow',
                             (0, 255, 0): 'green', (0, 0, 0): 'off'}[(red, green, blue)])
        if first:
            first = False
            print('RED', flush=True)
    try:
        race_sequence(fill, lambda seconds: bool(select.select([sys.stdin], [], [], seconds)[0]))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('This LED-only helper is started by MouseAirMotion.py')
    worker()
