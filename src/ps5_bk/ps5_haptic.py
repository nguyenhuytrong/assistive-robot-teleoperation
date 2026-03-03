import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
from haptic_controller import HapticController

 
class PS5HapticNode(Node):

    def __init__(self):
        super().__init__('ps5_haptic_node')

        self.haptic = HapticController()

        self.danger_zone = 0.25
        self.warning_zone = 0.5

        self.front = 99
        self.back = 99
        self.left = 99
        self.right = 99

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.create_timer(0.05, self.update_haptic)

    def scan_callback(self, msg):

        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        self.front = 99
        self.back = 99
        self.left = 99
        self.right = 99

        for i, r in enumerate(ranges):
            if not (msg.range_min < r < msg.range_max):
                continue
            if math.isinf(r):
                continue

            angle = angle_min + i * angle_increment

            # FRONT (-15° → +15°)
            if -0.26 < angle < 0.26:
                self.front = min(self.front, r)

            # LEFT (60° → 120°)
            elif 1.05 < angle < 2.09:
                self.left = min(self.left, r)

            # RIGHT (-120° → -60°)
            elif -2.09 < angle < -1.05:
                self.right = min(self.right, r)

            # BACK (|angle| > 165°)
            elif abs(angle) > 2.88:
                self.back = min(self.back, r)

    def compute_intensity(self, dist):
        if dist > self.warning_zone:
            return 0

        ratio = 1.0 - (dist / self.warning_zone)
        return 255 * ratio

    def update_haptic(self):

        min_dist = min(self.front, self.back, self.left, self.right)

        if min_dist > self.warning_zone:
            self.haptic.reset()
            self.haptic.ds.light.setColorI(0, 255, 0)
            return

        # PRIORITY: danger zone
        if min_dist < self.danger_zone:
            intensity = 255
        else:
            intensity = self.compute_intensity(min_dist)

        # Decide direction
        if self.front == min_dist:
            # Both motors strong & steady
            self.haptic.vibrate_both(intensity, intensity)

        elif self.back == min_dist:
            # Both motors but alternating pattern
            self.haptic.vibrate_both(intensity, 0)

        elif self.left == min_dist:
            self.haptic.vibrate_left(intensity)

        elif self.right == min_dist:
            self.haptic.vibrate_right(intensity)

        self.haptic.ds.light.setColorI(255, 0, 0)

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
