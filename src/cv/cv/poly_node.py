#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_msgs.msg import Poly
from cv.trt_py import SegmentVisual
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage


class PolyNode(Node):
    def __init__(self):
        super().__init__("poly_node")

        self.declare_parameter("color_option", 0)
        color_option = self.get_parameter("color_option").value

        self.subscriber = self.create_subscription(
            Poly,
            "/poly_pre",
            self.data_callback,
            10,
        )
        self.publisher = self.create_publisher(Image, "/poly_vis", 10)

        self.data_queue = []
        self.bridge     = CvBridge()
        self.visual     = SegmentVisual(color_option=color_option)
        self.timer      = self.create_timer(1.0 / 15.0, self.timer_callback)

        self.get_logger().info("PolyNode started")

    def data_callback(self, data):
        self.data_queue.append(data)
        if len(self.data_queue) > 2:
            self.data_queue.pop(0)

    def timer_callback(self):
        if not self.data_queue:
            return

        data = self.data_queue.pop(0)

        try:
            # seg_img: mono8 label map (ADE20K class IDs, 0-150)
            seg_img = self.bridge.imgmsg_to_cv2(data.segment_image, "mono8")

            # frame: bgr8 original image — convert to PIL for poly_visual
            frame_bgr = self.bridge.imgmsg_to_cv2(data.frame, "bgr8")
            frame_pil = PILImage.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            poly_image = self.visual.poly_visual(seg_img, frame_pil)

            if poly_image is None:
                self.get_logger().warning("poly_visual returned None")
                return

            if poly_image.ndim != 3 or poly_image.shape[2] != 3:
                self.get_logger().error(f"poly_visual invalid shape: {poly_image.shape}")
                return

            # Ensure same size as seg_img (should already be OUT_SIZE x OUT_SIZE)
            if poly_image.shape[:2] != seg_img.shape[:2]:
                poly_image = cv2.resize(
                    poly_image, (seg_img.shape[1], seg_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            poly_msg = self.bridge.cv2_to_imgmsg(poly_image, "bgr8")
            poly_msg.header = data.frame.header  # preserve timestamp
            self.publisher.publish(poly_msg)

        except Exception as e:
            self.get_logger().error(f"poly_node error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = PolyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()