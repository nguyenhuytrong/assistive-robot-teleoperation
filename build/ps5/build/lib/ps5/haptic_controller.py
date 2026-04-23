# haptic_controller
#!/usr/bin/env python3
"""
HapticController: Low-level DualSense rumble API + high-level patterns.
WARNING: fixed intensities (L25/R63). DANGER: fixed (L100/R255).
"""

import time
from typing import Optional, Tuple
from pydualsense import pydualsense

# Assuming you have DualSense library imported as:
# from your_dualsense_lib import DualSense  # Replace with actual import
class HapticController:
    """Encapsulates DualSense rumble control."""

    def __init__(self):
        self.ds = None  # DualSense handle
        self.available = False
        self._init_controller()

    def _init_controller(self):
        try:
            self.ds = pydualsense()
            self.ds.init()
            self.ds.light.setColorI(0, 255, 0)
            self.available = True
        except Exception as e:
            print(f"Could not init PS5 haptic: {e}")
            self.available = False

    def set(self, left: int = 0, right: int = 0):
        left  = max(0, min(255, int(left)))
        right = max(0, min(255, int(right)))
        if not self.available or self.ds is None:
            return
        self.ds.leftMotor  = left
        self.ds.rightMotor = right

    def close(self):
        if self.available and self.ds is not None:
            self.reset()
            self.ds.close()
        self.available = False

    def reset(self):
        """Stop all rumble."""
        self.set(0, 0)

    # Intensity for warning and danger zones
    def get_intensity(self, sector: str, zone: str) -> Tuple[int, int]:
        """
        Fixed intensities per zone/sector.
        zone: 'warning' (0.5-0.25m) or 'danger' (<0.25m)
        """
        if zone == 'warning':
            if sector == 'left':   return 25,  0
            if sector == 'right':  return  0, 63
            return 25, 63  # front, back, or both
        
        # danger zone
        if sector == 'left':   return 100,  0
        if sector == 'right':  return  0, 255
        return 100, 255  # front, back, or both

    # Pulse generator
    @staticmethod
    def pulse(period: float, on_beats: int, total_beats: int) -> bool:
        """True during ON beats."""
        if total_beats <= 0:
            return False
        t = time.monotonic() % (period * total_beats)
        return t < (period * on_beats)

    # High-level patterns
    def pattern_warning(self, sector: str, period: float = 0.4):
        """Warning zone patterns."""
        l, r = self.get_intensity(sector, 'warning')
        
        if sector == 'front':
            
            self.set(l, r)  # Continuous
            
        elif sector == 'left':
            on = self.pulse(period, 1, 2)
            self.set(left=l if on else 0, right=0)
            
        elif sector == 'right':
            on = self.pulse(period, 2, 3)
            self.set(left=0, right=r if on else 0)

        elif sector == 'back':
            # Alternating: Beat 1 -> Left ON, Beat 2 -> Right ON
            on = self.pulse(period, 1, 2)
            if on:
                self.set(left=l, right=0)
            else:
                self.set(left=0, right=r)

    def pattern_danger(self, sector: str, period: float = 0.15):
        """Danger zone patterns (faster alternating for back)."""
        l, r = self.get_intensity(sector, 'danger')
        
        if sector == 'front':
            self.set(l, r)  # Continuous
            
        elif sector == 'left':
            on = self.pulse(period, 1, 2)
            self.set(left=l if on else 0, right=0)
            
        elif sector == 'right':
            on = self.pulse(period, 2, 3)
            self.set(left=0, right=r if on else 0)

        elif sector == 'back':
            # Fast alternating for danger
            on = self.pulse(period, 2, 4)
            if on:
                self.set(left=l, right=0)
            else:
                self.set(left=0, right=r)
