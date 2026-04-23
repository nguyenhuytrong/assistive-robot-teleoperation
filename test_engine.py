#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from PIL import Image as PILImage

class TensorRTPredictor:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        # 1. Load engine
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.names = []

        # 2. Phân bổ bộ nhớ GPU (Pinned memory)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.names.append(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            
            # Allocate memory
            allocation = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.allocations.append(allocation)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                self.input_dtype = dtype
            else:
                self.outputs.append(np.zeros(shape, dtype=dtype))

    def predict(self, pixel_values, pixel_mask):
        # Copy both inputs to GPU
        cuda.memcpy_htod(self.allocations[0], pixel_values.ravel())
        cuda.memcpy_htod(self.allocations[1], pixel_mask.ravel())

        # Set tensor addresses and run
        for i, name in enumerate(self.names):
            self.context.set_tensor_address(name, int(self.allocations[i]))
        self.context.execute_v2(self.allocations)

        # Copy outputs back — inputs are at [0,1], outputs start at [2]
        for i, out in enumerate(self.outputs):
            cuda.memcpy_dtoh(out, self.allocations[2 + i])

        return self.outputs
        


def main():
    print("=== Mask2Former TensorRT Engine Test ===")
    engine_path = "mask2former.engine"
    image_path  = "image.png"

    if not os.path.exists(engine_path):
        print(f"❌ Không tìm thấy file {engine_path}!")
        return

    predictor = TensorRTPredictor(engine_path)

    # Preprocessing — 384x384 to match model
    pil_img    = PILImage.open(image_path).convert("RGB")
    img_np     = np.array(pil_img.resize((384, 384))).astype(np.float32) / 255.0
    img_np     = np.transpose(img_np, (2, 0, 1))
    img_np     = np.ascontiguousarray(np.expand_dims(img_np, 0))  # (1,3,384,384)
    pixel_mask = np.ones((1, 384, 384), dtype=np.int64)           # all valid pixels

    # Warmup
    print("🔥 Warmup engine...")
    for _ in range(5):
        predictor.predict(img_np, pixel_mask)

    # Benchmark
    start   = time.time()
    outputs = predictor.predict(img_np, pixel_mask)
    end     = time.time()

    print(f"✅ TRT Inference: {(end-start)*1000:.2f}ms")
    print(f"✅ Tốc độ ước tính: {1/(end-start):.1f} FPS")

    print("💾 Lưu kết quả onnx_outputs.npz...")
    np.savez("onnx_outputs.npz", out0=outputs[0], out1=outputs[1])
    print("🎉 ENGINE READY!")

if __name__ == "__main__":
    main()