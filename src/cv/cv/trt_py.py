import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import cv2
import torch

from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput
from cv.predict_utils import post_process_semantic_segmentation1, ade_palette
from transformers import AutoImageProcessor

import time
from PIL import Image as PILImage

# ── Category definitions ──────────────────────────────────────────────────────
WALL_IDS   = {0}        # wall
FLOOR_IDS  = {3, 28}    # floor, rug
PERSON_IDS = {12}       # person
# Everything else with a confident mask = "object"

CATEGORY_COLORS = {
    "wall":   (  0, 255, 255),  # yellow
    "floor":  (  0, 255,   0),  # green
    "person": (  0,   0, 255),  # red
    "object": (128,   0, 128),  # purple
}

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting",
    "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair",
    "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing",
    "cushion", "base", "box", "column", "signboard", "chest", "counter", "sand",
    "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path",
    "stairs", "runway", "case", "pool", "pillow", "screen", "stairway", "river",
    "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book",
    "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "tv", "airplane", "dirt track", "apparel", "pole", "land", "bannister",
    "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van",
    "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
    "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
    "trash can", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag"
]

def get_category(class_id):
    if class_id in WALL_IDS:   return "wall"
    if class_id in FLOOR_IDS:  return "floor"
    if class_id in PERSON_IDS: return "person"
    return "object"


# ── TensorRT inference ────────────────────────────────────────────────────────
class TensorRTInference:
    def __init__(self, engine_path):
        self.logger  = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine  = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host   = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size        = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype       = trt.nptype(engine.get_tensor_dtype(tensor_name))
            host_mem    = cuda.pagelocked_empty(size, dtype)
            device_mem  = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def infer(self, pixel_values, pixel_mask):
        np.copyto(self.inputs[0].host, pixel_values.ravel())
        np.copyto(self.inputs[1].host, pixel_mask.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        cuda.memcpy_htod_async(self.inputs[1].device, self.inputs[1].host, self.stream)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()

        # outputs[0] = class_queries_logits  (1, 100, 151)
        # outputs[1] = masks_queries_logits  (1, 100, 96, 96)
        return self.outputs[0].host, self.outputs[1].host


# ── High-level predictor ──────────────────────────────────────────────────────
class trt_infernce:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.image_processor, self.trt_inference = self.load(engine_path)

    def preprocess_image(self, image):
        img          = np.array(image.resize((384, 384))).astype(np.float32) / 255.0
        img          = np.transpose(img, (2, 0, 1))
        pixel_values = np.ascontiguousarray(np.expand_dims(img, 0))   # (1,3,384,384)
        pixel_mask   = np.ones((1, 384, 384), dtype=np.int64)         # all valid pixels
        return pixel_values, pixel_mask

    @staticmethod
    def load(engine_path):
        inference        = TensorRTInference(engine_path)
        image_processor  = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-tiny-ade-semantic"
        )
        return image_processor, inference

    def predict(self, image):
        pixel_values, pixel_mask = self.preprocess_image(image)
        class_queries_logits, masks_queries_logits = self.trt_inference.infer(pixel_values, pixel_mask)
        print("done prediction")
        return class_queries_logits, masks_queries_logits


# ── Visualization ─────────────────────────────────────────────────────────────
class SegmentVisual:

    SCORE_THRESH = 0.5
    MASK_SIZE    = 96      # engine output mask resolution
    OUT_SIZE     = 512     # display resolution

    def __init__(self, color_option=None):
        # color_option kept for backwards-compat but no longer used for coloring
        self.last_poly_image = np.zeros((self.OUT_SIZE, self.OUT_SIZE, 3), dtype=np.uint8)
        self.selected_labels = [3, 12, 42, 138]   # floor, person, box, ?
        self.epsilon_const   = {0: 0.01, 3: 0.01, 12: 0.007, 42: 0.01, 138: 0.01}

    # ── internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _softmax(logits):
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _draw_legend(img):
        for i, (cat, color) in enumerate(CATEGORY_COLORS.items()):
            y = 25 + i * 30
            cv2.rectangle(img, (10, y - 16), (30, y + 4), color, -1)
            cv2.rectangle(img, (10, y - 16), (30, y + 4), (255, 255, 255), 1)
            cv2.putText(img, cat, (38, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def _build_seg(self, class_logits_raw, mask_logits_raw):
        class_logits = class_logits_raw.reshape(1, 100, 151)
        mask_logits  = mask_logits_raw.reshape(1, 100, self.MASK_SIZE, self.MASK_SIZE)

        class_probs  = self._softmax(class_logits[0])
        fg_probs     = 1.0 - class_probs[:, -1]
        pred_classes = np.argmax(class_probs[:, :-1], axis=1)

        confident = np.where(fg_probs > self.SCORE_THRESH)[0]
        confident = confident[np.argsort(fg_probs[confident])[::-1]]

        color_seg   = np.zeros((self.MASK_SIZE, self.MASK_SIZE, 3), dtype=np.uint8)
        seg_map     = np.full((self.MASK_SIZE, self.MASK_SIZE), -1, dtype=np.int32)
        seg_map96   = np.zeros((self.MASK_SIZE, self.MASK_SIZE), dtype=np.uint8)  # ← ADE20K labels
        object_type = {12: 0, 42: 0}

        for rank, q in enumerate(confident):
            mask_prob  = 1.0 / (1.0 + np.exp(-mask_logits[0, q]))
            binary     = mask_prob > 0.5
            unassigned = (seg_map == -1) & binary
            if not unassigned.any():
                continue

            class_id = pred_classes[q]
            category = get_category(class_id)

            seg_map[unassigned]    = rank
            color_seg[unassigned]  = CATEGORY_COLORS[category]
            seg_map96[unassigned]  = class_id          # ← store ADE20K id, not rank

            if class_id in object_type:
                object_type[class_id] = 1

        return color_seg, object_type, seg_map96      # ← return 3 values

    # ── public API ────────────────────────────────────────────────────────────
    def segment_visual(self, class_queries_logits, masks_queries_logits, image):
        color_seg, object_type, seg_map96 = self._build_seg(
            class_queries_logits, masks_queries_logits
        )

        color_seg_up = cv2.resize(
            color_seg, (self.OUT_SIZE, self.OUT_SIZE), interpolation=cv2.INTER_NEAREST
        )
        orig_bgr = cv2.cvtColor(
            np.array(image.resize((self.OUT_SIZE, self.OUT_SIZE))), cv2.COLOR_RGB2BGR
        )
        overlay = cv2.addWeighted(orig_bgr, 0.5, color_seg_up, 0.5, 0)
        #self._draw_legend(overlay)

        poly_seg = cv2.resize(
            seg_map96.astype(np.uint8), (self.OUT_SIZE, self.OUT_SIZE),
            interpolation=cv2.INTER_NEAREST
        )
        return overlay, poly_seg, object_type

    def poly_visual(self, seg_img, image):
        """
        seg_img : uint8 ADE20K label map (OUT_SIZE x OUT_SIZE)
        """
        if isinstance(image, PILImage.Image):
            canvas = cv2.cvtColor(
                np.array(image.resize((self.OUT_SIZE, self.OUT_SIZE))), cv2.COLOR_RGB2BGR
            )
        else:
            canvas = image

        # Build category mask from label map
        category_masks = {
            "wall":   np.isin(seg_img, list(WALL_IDS)),
            "floor":  np.isin(seg_img, list(FLOOR_IDS)),
            "person": np.isin(seg_img, list(PERSON_IDS)),
            "object": ~np.isin(seg_img, list(WALL_IDS | FLOOR_IDS | PERSON_IDS)) & (seg_img > 0),
        }

        result = np.zeros_like(canvas)

        for category, mask in category_masks.items():
            if not mask.any():
                continue
            color    = CATEGORY_COLORS[category]
            binary   = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            epsilon  = 0.01 * cv2.arcLength(contours[0], True)
            polygons = [cv2.approxPolyDP(c, epsilon, True) for c in contours]
            cv2.fillPoly(result, polygons, color)

        # Fill black pixels with wall color as background
        result[np.all(result == [0, 0, 0], axis=-1)] = list(CATEGORY_COLORS["wall"])

        if result.any():
            self.last_poly_image = result

        #self._draw_legend(result)
        return result