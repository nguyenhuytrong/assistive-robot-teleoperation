import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class PS5ControlNode(Node):
    def __init__(self):
        super().__init__('ps5_control_node')

        # ========= PS5 CONFIGURATION =========
        self.ENABLE_BUTTON = 10      # R1 button - hold to enable movement
        self.MODE_BUTTON = 9        # R2 button - toggle control mode
        self.SCALE_LINEAR = 0.3     # Maximum linear speed (m/s)
        self.SCALE_ANGULAR = 0.8    # Maximum angular speed (rad/s)
        # ======================================

        # Mode 1: Left stick Y + Right stick X
        self.AXIS_LINEAR_MODE1 = 1
        self.AXIS_ANGULAR_MODE1 = 2

        # Mode 2: Left stick only (original)
        self.AXIS_LINEAR_MODE2 = 1
        self.AXIS_ANGULAR_MODE2 = 0

        self.current_mode = 1       # Default mode
        self.prev_mode_button = 0   # Track R2 button state

        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.SMOOTH_FACTOR = 0.3

        self.subscription = self.create_subscription(
            Joy,
            '/ps5/joy',
            self.joy_callback,
            10
        )

        self.publisher = self.create_publisher(
            Twist,
            '/controller/cmd_vel',
            10
        )

        self.get_logger().info('PS5 Control Node started!')
        self.get_logger().info('Mode 1 active: Left stick + Right stick')
        self.get_logger().info('Press R2 to toggle mode')

    def joy_callback(self, msg):
        cmd = Twist()

        # Detect R2 press while holding R1 (toggle mode)
        if msg.buttons[self.ENABLE_BUTTON] == 1 and \
        msg.buttons[self.MODE_BUTTON] == 1 and \
        self.prev_mode_button == 0:
            if self.current_mode == 1:
                self.current_mode = 2
                self.get_logger().info('Switched to Mode 2: Strafe mode')
            else:
                self.current_mode = 1
                self.get_logger().info('Switched to Mode 1: Left stick + Right stick')
        self.prev_mode_button = msg.buttons[self.MODE_BUTTON]

        # Select axes based on current mode
        if self.current_mode == 1:
            axis_linear = self.AXIS_LINEAR_MODE1
            axis_angular = self.AXIS_ANGULAR_MODE1
        else:
            axis_linear = self.AXIS_LINEAR_MODE2
            axis_angular = self.AXIS_ANGULAR_MODE2

        # Only move when R1 is held
        if msg.buttons[self.ENABLE_BUTTON] == 1:
            target_linear = msg.axes[axis_linear] * self.SCALE_LINEAR

            # Mode 1: angular.z (rotation)
            if self.current_mode == 1:
                target_angular = msg.axes[axis_angular] * self.SCALE_ANGULAR
                if abs(target_linear) < 0.05:
                    target_linear = 0.0
                if abs(target_angular) < 0.05:
                    target_angular = 0.0

                cmd.linear.x = self.prev_linear + self.SMOOTH_FACTOR * (target_linear - self.prev_linear)
                cmd.angular.z = self.prev_angular + self.SMOOTH_FACTOR * (target_angular - self.prev_angular)

                self.prev_linear = cmd.linear.x
                self.prev_angular = cmd.angular.z

            # Mode 2: linear.y (strafe left/right)
            else:
                target_strafe = msg.axes[axis_angular] * self.SCALE_LINEAR
                if abs(target_linear) < 0.05:
                    target_linear = 0.0
                if abs(target_strafe) < 0.05:
                    target_strafe = 0.0

                cmd.linear.x = target_linear
                cmd.linear.y = target_strafe  # Strafe instead of rotate
                cmd.angular.z = 0.0

        self.publisher.publish(cmd)

def main():
    rclpy.init()
    node = PS5ControlNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()