#!/usr/bin/env python3
import math
import time
from threading import Lock
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from ps5.haptic_controller import HapticController


class PS5HapticNode(Node):
    """ROS2 node: LiDAR → haptic patterns via HapticController."""

    DANGER  = 0.3  # m
    WARNING = 0.7  # m

    def __init__(self):
        super().__init__('ps5_haptic_node')
        
        # Haptics
        self.haptic = HapticController()
        
        # LiDAR snapshot (thread-safe)
        self._front = 99.0
        self._left  = 99.0
        self._right = 99.0
        self._back  = 99.0
        self._scan_lock = Lock()
        
        # Sector stability
        self._current_sector: Optional[str] = None
        self._sector_since = time.monotonic()
        self._sector_hold = 0.20  # seconds

        # Change to /scan if testing in simulation, keep /scan_raw for real robot
        self.create_subscription(LaserScan, '/scan_raw', self.scan_callback, 10)
        self.create_timer(0.05, self.update_haptic)  # 20Hz
        
        self.get_logger().info("PS5HapticNode: 4-sector sensing (Front, Left, Right, Back) ready")

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR frame → update snapshot with 4 sectors."""
        front = left = right = back = 99.0

        for i, r in enumerate(msg.ranges):
            # 1. Check data validity
            if math.isinf(r) or math.isnan(r) or not (msg.range_min < r < msg.range_max):
                continue
                
            # 2. Calculate raw angle from LiDAR data
            raw_angle = msg.angle_min + i * msg.angle_increment
            
            # 3. NORMALIZE ANGLE: Wrap to [-pi, pi] range
            angle = math.atan2(math.sin(raw_angle), math.cos(raw_angle))

            # 4. Classify into sectors based on normalized angle
            if -0.349 < angle < 0.349:          # FRONT: ±20°
                front = min(front, r)
            elif 0.349 <= angle < 2.443:        # LEFT: 20° to 140°
                left = min(left, r)
            elif -2.443 < angle <= -0.349:      # RIGHT: -140° to -20°
                right = min(right, r)
            else:                               # BACK: Remaining 80°
                back = min(back, r)

        with self._scan_lock:
            self._front = front
            self._left = left
            self._right = right
            self._back = back

    def get_scan_snapshot(self) -> Tuple[float, float, float, float]:
        """Thread-safe LiDAR snapshot including the back sector."""
        with self._scan_lock:
            return self._front, self._left, self._right, self._back

    def pick_sector(self, front: float, left: float, right: float, back: float) \
                -> Tuple[Optional[str], float]:
        """Pick stable sector with hysteresis, supporting the back zone."""
        min_dist = min(front, left, right, back)

        # Safe zone → no haptic feedback
        if min_dist >= self.WARNING:
            self._current_sector = None
            self._sector_since = time.monotonic()
            return None, min_dist

        # Identify the closest candidate sector
        if front <= min_dist:
            candidate = 'front'
        elif left <= min_dist:
            candidate = 'left'
        elif right <= min_dist:
            candidate = 'right'
        else:
            candidate = 'back'

        now = time.monotonic()
        
        # Same sector → keep
        if candidate == self._current_sector:
            return self._current_sector, min_dist
        
        # New sector but not stable yet → keep old (prevents flickering)
        if now - self._sector_since < self._sector_hold:
            return self._current_sector, min_dist
        
        # Switch to the new stable sector
        self._current_sector = candidate
        self._sector_since = now
        return self._current_sector, min_dist

    def update_haptic(self):
        """20Hz: LiDAR → haptic pattern."""
        # Unpack all 4 values
        front, left, right, back = self.get_scan_snapshot()
        sector, min_dist = self.pick_sector(front, left, right, back)

        # SAFE: reset + green light
        if sector is None or min_dist >= self.WARNING:
            self.haptic.reset()
            if self.haptic.available and self.haptic.ds is not None:
                self.haptic.ds.light.setColorI(0, 255, 0)  # Green
            return

        # DANGER/WARNING: red light
        if self.haptic.available and self.haptic.ds is not None:
            self.haptic.ds.light.setColorI(255, 0, 0)  # Red                    

        if not self.haptic.available:
            return

        # FIXED INTENSITIES → pick pattern
        if min_dist >= self.DANGER:  # WARNING zone
            self.haptic.pattern_warning(sector)
        else:  # DANGER zone
            self.haptic.pattern_danger(sector)

    def destroy_node(self):
        self.haptic.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PS5HapticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()