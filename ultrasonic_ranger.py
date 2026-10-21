"""Bounded, single-SIG Grove ranger reads without a GPIO daemon or DMA."""
import time


class UltrasonicRanger:
    def __init__(self, pin=None, gpio=None, clock=time.monotonic, sleep=time.sleep):
        if gpio is None:
            import RPi.GPIO as gpio
        if pin is None:
            try:
                import conflocal as config
            except ImportError:
                import config
            pin = config.UltrasonicLevelPin
        self.gpio, self.pin, self.clock, self.sleep = gpio, pin, clock, sleep
        gpio.setwarnings(False)
        gpio.setmode(gpio.BCM)

    def read_cm(self):
        g, pin = self.gpio, self.pin
        g.setup(pin, g.OUT)
        g.output(pin, False)
        self.sleep(0.000005)
        g.output(pin, True)
        self.sleep(0.00001)
        g.output(pin, False)
        g.setup(pin, g.IN)
        deadline = self.clock() + 0.04
        # Short sleeps avoid a CPU-spinning loop on missing echoes. Distance
        # filtering tolerates Linux scheduling noise; this is not precision
        # ranging. Both edges share one deadline so every read is bounded.
        while not g.input(pin):
            if self.clock() >= deadline:
                return None
            self.sleep(0.00005)
        start = self.clock()
        while g.input(pin):
            if self.clock() >= deadline:
                return None
            self.sleep(0.00005)
        end = self.clock()
        return (end - start) * 17150 if end < deadline else None

    def close(self):
        # Leave the toy detector and LED pins alone.
        self.gpio.cleanup(self.pin)


if __name__ == '__main__':
    ranger = UltrasonicRanger()
    try:
        for _ in range(40):
            print(ranger.read_cm(), flush=True)
            time.sleep(0.25)
    finally:
        ranger.close()
