#! /usr/bin/python3
import sys
print("PYTHON EXE =", sys.executable)
print("PYTHONPATH =", sys.path)

import multiprocessing as mp

# Must be called before any CUDA import or fork to avoid GPU context corruption
mp.set_start_method("spawn", force=True)

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

from cv_msgs.msg import Poly
from cv_msgs.msg import ObjectType

from cv_bridge import CvBridge, CvBridgeError

import cv2
from PIL import Image as PILImage
import threading
import time
import queue
from cv.trt_py import trt_infernce, SegmentVisual
import pycuda.driver as cuda
import numpy as np


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")
        self.flag = "trt"

        self.declare_parameter("color_option", 0)
        color_option = self.get_parameter("color_option").value

        self.subscriber = self.create_subscription(
            CompressedImage,
            "/depth_cam/rgb0/image_raw/compressed",
            self.data_callback,
            10,
        )

        self.publisher = self.create_publisher(Image, "/seg_img", 10)
        self.poly_data_publisher = self.create_publisher(Poly, "/poly_pre", 10)
        self.object_type_publisher = self.create_publisher(ObjectType, "/object_type", 10)

        self.bridge = CvBridge()

        # FIX: mp.Queue with maxsize to prevent unbounded growth if inference lags
        self.data_queue    = mp.Queue(maxsize=2)
        self.segment_queue = mp.Queue(maxsize=2)

        # FIX: threading.Lock is sufficient (and lighter) for the callback-side
        # frame counter, which only runs in the main process ROS spin thread.
        self._frame_lock  = threading.Lock()
        self.frame_count  = 0
        self.frame_skip   = 2

        # mp.Event so the subprocess can observe the stop signal
        self.running = mp.Event()
        self.running.set()

        self.visual_output = SegmentVisual(color_option=color_option)
        self.get_logger().info(f"Color option: {color_option}")
        self.get_logger().info("InferenceNode initialized")

        self.predict_prc = mp.Process(
            target=_inference_worker,          # module-level function — picklable
            args=(self.data_queue, self.segment_queue, self.running),
            daemon=True,
        )
        self.seg_thrt = threading.Thread(
            target=self.run_visualization, args=(self.segment_queue,), daemon=True
        )

    # ── ROS subscriber callback (runs in rclpy executor thread) ──────────────
    def data_callback(self, msg):
        with self._frame_lock:
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                return

        # Drop frame silently if the inference process hasn't consumed the last one
        try:
            self.data_queue.put_nowait(msg)
        except mp.queues.Full:
            self.get_logger().debug("data_queue full — frame dropped")

    # ── Visualization thread (main process) ──────────────────────────────────
    def run_visualization(self, segment_queue):
        self.get_logger().info("Visualization thread started")

        while self.running.is_set():
            try:
                # Blocking get with timeout so the loop can check running.is_set()
                seg, mask, img_array = segment_queue.get(timeout=0.05)
            except mp.queues.Empty:
                continue

            # Drain stale results — keep only the freshest
            while not segment_queue.empty():
                try:
                    seg, mask, img_array = segment_queue.get_nowait()
                except mp.queues.Empty:
                    break

            start = time.time()

            try:
                # img_array arrives as RGB numpy array from the subprocess
                img_pil = PILImage.fromarray(img_array)
                seg_img, poly_seg, object_type_ = self.visual_output.segment_visual(
                    seg, mask, img_pil
                )

                # Object type message
                object_type_msg = ObjectType()
                object_type_msg.human = bool(object_type_.get(12, 0))
                object_type_msg.box   = bool(object_type_.get(42, 0))
                self.object_type_publisher.publish(object_type_msg)
                self.get_logger().debug(f"Detected objects: {object_type_}")

                # Segmentation overlay
                seg_msg = self.bridge.cv2_to_imgmsg(seg_img, "bgr8")
                self.publisher.publish(seg_msg)

                # Poly message: mono8 label map + original frame
                poly_seg_msg = self.bridge.cv2_to_imgmsg(
                    poly_seg.astype(np.uint8), "mono8"
                )
                frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                frame_msg = self.bridge.cv2_to_imgmsg(frame_bgr, "bgr8")

                poly_msg = Poly()
                poly_msg.segment_image = poly_seg_msg
                poly_msg.frame = frame_msg
                self.poly_data_publisher.publish(poly_msg)

                self.get_logger().debug(
                    f"Published in {(time.time() - start) * 1000:.1f} ms"
                )

            except Exception as e:
                self.get_logger().error(f"Visualization error: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        self.predict_prc.start()
        self.seg_thrt.start()

    def stop(self):
        self.running.clear()

        if self.predict_prc.is_alive():
            self.predict_prc.terminate()
            self.predict_prc.join(timeout=3.0)

        if self.seg_thrt.is_alive():
            self.seg_thrt.join(timeout=2.0)


# ── Subprocess worker (module-level so it is picklable with 'spawn') ─────────
def _inference_worker(data_queue, segment_queue, running):
    """Runs entirely inside a separate process — owns its own CUDA context."""
    print("[inference] worker started")

    # These imports happen fresh in the spawned process — no fork-inherited state
    import pycuda.driver as cuda
    import cv2
    import numpy as np
    from PIL import Image as PILImage
    from cv_bridge import CvBridge
    from cv.trt_py import trt_infernce

    local_bridge = CvBridge()

    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    predictor = trt_infernce("/mask2former.engine")
    print("[inference] TRT engine loaded")

    try:
        while running.is_set():
            try:
                # Blocking get — no busy-wait, no lock needed (Queue is process-safe)
                msg = data_queue.get(timeout=0.05)
            except Exception:
                continue

            try:
                bgr = local_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img_pil = PILImage.fromarray(rgb)

                class_logits, mask_logits = predictor.predict(img_pil)

                # Send numpy array (serialisable) rather than PIL object
                try:
                    segment_queue.put_nowait([class_logits, mask_logits, rgb])
                except Exception:
                    pass  # visualization hasn't consumed yet — drop this result

            except Exception as e:
                print(f"[inference] error: {e}")
                import traceback
                traceback.print_exc()

    finally:
        ctx.pop()
        print("[inference] CUDA context released")


# ── Entry point ───────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()

    try:
        node.start()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()