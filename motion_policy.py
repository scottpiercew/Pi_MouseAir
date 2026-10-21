"""Hardware-independent motion debounce and cat-confidence launch policy."""
import math


def cat_probability(scores, indices):
    if not scores or not indices:
        raise ValueError('Missing classifier output or cat labels')
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in scores):
        raise ValueError('Invalid classifier output; refusing launch')
    if any(i < 0 or i >= len(scores) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('Invalid cat label indices')
    return min(1.0, sum(float(scores[i]) for i in indices))


class MotionPolicy:
    def __init__(self, confidence=0.90, cooldown=30.0, motion_frames=3, quiet_frames=5, retry_seconds=None):
        if not 0.5 < confidence <= 1.0:
            raise ValueError('Cat confidence must be greater than 0.5 and at most 1')
        if cooldown < 0 or motion_frames < 1 or quiet_frames < 1:
            raise ValueError('Invalid debounce/cooldown settings')
        if retry_seconds is not None and (not math.isfinite(retry_seconds) or retry_seconds <= 0):
            raise ValueError('Retry interval must be finite and positive')
        self.confidence = confidence
        self.cooldown = cooldown
        self.motion_frames = motion_frames
        self.quiet_frames = quiet_frames
        self.moving = 0
        self.quiet = 0
        self.latched = False
        self.next_launch = 0.0
        self.retry_seconds = retry_seconds
        self.last_capture = float('-inf')

    def observe(self, moving, now):
        if moving:
            self.moving += 1
            self.quiet = 0
        else:
            self.moving = 0
            self.quiet += 1
            if self.quiet >= self.quiet_frames:
                self.latched = False
        retry = self.retry_seconds is not None and now - self.last_capture >= self.retry_seconds
        if moving and self.moving >= self.motion_frames and (not self.latched or retry) and now >= self.next_launch:
            self.latched = True
            self.last_capture = now
            return True
        return False

    def accepts_cat(self, score, now):
        return math.isfinite(score) and self.confidence <= score <= 1.0 and now >= self.next_launch

    def launched(self, now):
        self.next_launch = now + self.cooldown


def launch_sequence(motors):
    """Always stop the motors, including when a servo or I2C operation fails."""
    try:
        motors.launchMotorsOn()
        motors.launchServoStart()
    finally:
        motors.launchMotorsOff()
    motors.launchServoRetract()
