#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class CompressedNode(Node):
    def __init__(self):
        super().__init__("compressed_node")

        self.declare_parameter("jpeg_quality", 80)
        self.jpeg_quality = self.get_parameter("jpeg_quality").value

        self.bridge = CvBridge()

        # Subscribe to raw image from depth camera
        self.subscriber = self.create_subscription(
            Image,
            "/depth_cam/rgb0/image_raw",
            self.image_callback,
            10,
        )

        # Publish CompressedImage 
        self.publisher = self.create_publisher(
            CompressedImage,
            "/depth_cam/rgb0/image_raw/compressed",
            10,
        )

        self.get_logger().info(
            f"CompressedNode started | "
            f"in: /depth_cam/rgb0/image_raw → "
            f"out: /depth_cam/rgb0/image_raw/compressed | "
            f"JPEG quality: {self.jpeg_quality}"
        )

    def image_callback(self, msg: Image):
        try:
            # Convert ROS Image → cv2
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Compress to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, encoded = cv2.imencode(".jpg", cv_image, encode_params)

            if not success:
                self.get_logger().warn("Failed to encode image to JPEG")
                return

            # Build CompressedImage message
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header   # preserve timestamp & frame_id
            compressed_msg.format = "jpeg"
            compressed_msg.data = encoded.tobytes()

            self.publisher.publish(compressed_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CompressedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()