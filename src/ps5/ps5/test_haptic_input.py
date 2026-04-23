#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class HapticTester(Node):
    def __init__(self):
        super().__init__('haptic_tester')
        # Publish lên /scan_raw để PS5HapticNode nhận được
        self.publisher_ = self.create_publisher(LaserScan, '/scan_raw', 10)
        
        print("--- PS5 Haptic Tester ---")
        print("Enter 'q' to quit")

    def send_fake_scan(self, angle_deg, distance):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # Giả lập Lidar 360 độ giống robot thật
        msg.angle_min = 0.0
        msg.angle_max = 2.0 * math.pi
        msg.angle_increment = math.pi / 180.0 # 1 độ mỗi bước
        msg.range_min = 0.1
        msg.range_max = 10.0
        
        # Tạo mảng ranges toàn giá trị lớn (99.0)
        msg.ranges = [99.0] * 360
        
        # Đưa vật cản vào góc bạn muốn (0-359)
        idx = int(angle_deg) % 360
        msg.ranges[idx] = float(distance)
        
        self.publisher_.publish(msg)
        print(f"Sent: Obstacle at {angle_deg}° with distance {distance}m")

def main():
    rclpy.init()
    tester = HapticTester()
    
    try:
        while True:
            inp = input("\nEnter 'angle distance' (e.g., '180 0.2'): ")
            if inp.lower() == 'q':
                break
            
            try:
                angle, dist = map(float, inp.split())
                tester.send_fake_scan(angle, dist)
            except ValueError:
                print("Invalid input. Format: [angle] [distance]")
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()