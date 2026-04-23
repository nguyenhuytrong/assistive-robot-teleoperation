from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image

image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
model = model.eval().cuda()

image = Image.open("test.jpeg")
inputs = image_processor(images=image, return_tensors="pt")
inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    torch.onnx.export(
        model,
        (inputs["pixel_values"], inputs["pixel_mask"]),  # pass as tuple, not dict
        "mask2former.onnx",
        export_params=True,
        opset_version=16,
        input_names=["pixel_values", "pixel_mask"],
        output_names=["cls_logits", "mask_logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "pixel_mask":   {0: "batch_size"},
            "cls_logits":   {0: "batch_size"},
            "mask_logits":  {0: "batch_size"},
        }
    )

print("✅ ONNX export done!")