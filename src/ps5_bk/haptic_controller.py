from pydualsense import pydualsense

class HapticController:

    def __init__(self):
        self.ds = pydualsense()
        self.ds.init()
        self.ds.light.setColorI(0, 255, 0)

    def vibrate_left(self, intensity=255):
        self.ds.leftMotor = self._limit(intensity)
        self.ds.rightMotor = 0

    def vibrate_right(self, intensity=255):
        self.ds.leftMotor = 0
        self.ds.rightMotor = self._limit(intensity)

    def vibrate_both(self, left=255, right=255):
        self.ds.leftMotor = self._limit(left)
        self.ds.rightMotor = self._limit(right)

    def reset(self):
        self.ds.leftMotor = 0
        self.ds.rightMotor = 0

    def close(self):
        self.reset()
        self.ds.close()

    def _limit(self, value):
        return max(0, min(255, int(value)))
