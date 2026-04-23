#!/usr/bin/env python3
import os, cv2, numpy as np
from PIL import Image as PILImage
import onnxruntime as ort
import time

class onnx_infernce:
    def __init__(self):
        self.model = ort.InferenceSession("mask2former.onnx", providers=["CUDAExecutionProvider"])
    
    def onnx_preprocessing(self, image):
        img_array = np.array(image.resize((384, 384))).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)  # (1, 3, 384, 384)

        pixel_mask = np.ones((1, 384, 384), dtype=np.int64)

        return {
            "pixel_values": img_array,
            "pixel_mask":   pixel_mask
        }
    
    def predict(self, inputs):
        return self.model.run(None, inputs)
    
    def warmup(self, inputs, warmup_itr=3):
        for i in range(warmup_itr):
            self.predict(inputs)
        print("✅ warmup done")
    
    def run(self, image):
        inputs = self.onnx_preprocessing(image)
        start = time.time()
        outputs = self.predict(inputs)
        end = time.time()
        print(f"✅ Inference: {end-start:.3f}s, outputs: {[o.shape for o in outputs[:2]]}")
        return outputs

def main():
    print("=== Mask2Former ONNX Backend Test ===")
    
    # Create exact 512x512 test image
    if not os.path.exists("test.jpeg"):
        img = PILImage.new('RGB', (512, 512), color=(128,128,128))
        img.save("test.jpeg")
    
    pil_img = PILImage.open("test.jpeg").convert("RGB")
    print(f"Image size: {pil_img.size}")
    
    infer = onnx_infernce()
    inputs = infer.onnx_preprocessing(pil_img)
    infer.warmup(inputs)
    outputs = infer.run(pil_img)
    
    print("🎉 BACKEND 100% READY FOR ROS2!")
    print(f"Save outputs: onnx_outputs.npz")
    np.savez("onnx_outputs.npz", **{f'out{i}': outputs[i] for i in range(2)})

    data = np.load("onnx_outputs.npz")
    print(data.files)  # ['out0.npy', 'out1.npy']
    print(data['out0'].shape)  # (1, 100, 151) = class probabilities
    print(data['out1'].shape) 

if __name__ == "__main__":
    main()
