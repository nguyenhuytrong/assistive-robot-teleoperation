#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image as PILImage

# ADE20K class indices for categories we care about
WALL_IDS   = {0}                          # wall
FLOOR_IDS  = {3, 28}                      # floor, rug
PERSON_IDS = {12}                         # person
# Everything else with a confident mask = "object"

CATEGORY_COLORS = {
    "wall":   (  0, 215, 255),  # yellow
    "floor":  (  0, 200,   0),  # green
    "person": (  0,   0, 220),  # red
    "object": (180,   0, 180),  # purple
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

def draw_label(img, text, cx, cy, color):
    """Draw a filled label box centered at (cx, cy)."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 0.5
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1 = cx - tw // 2 - 4
    y1 = cy - th // 2 - 4
    x2 = cx + tw // 2 + 4
    y2 = cy + th // 2 + 4
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 1)
    cv2.putText(img, text, (cx - tw//2, cy + th//2),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def visualize_onnx_cv2(npz_path="onnx_outputs.npz", image_path="image.png"):
    data         = np.load(npz_path)
    class_logits = data['out0']
    mask_logits  = data['out1']

    MASK_H, MASK_W = mask_logits.shape[2], mask_logits.shape[3]
    OUT_SIZE = 512

    # Softmax → foreground score
    exp_logits   = np.exp(class_logits[0] - class_logits[0].max(axis=-1, keepdims=True))
    class_probs  = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    fg_probs     = 1.0 - class_probs[:, -1]
    pred_classes = np.argmax(class_probs[:, :-1], axis=1)

    SCORE_THRESH = 0.5
    confident = np.where(fg_probs > SCORE_THRESH)[0]
    confident = confident[np.argsort(fg_probs[confident])[::-1]]

    print(f"🔥 {len(confident)} confident queries (threshold={SCORE_THRESH})")

    # Build segmentation at mask resolution
    color_seg = np.zeros((MASK_H, MASK_W, 3), dtype=np.uint8)
    seg_map   = np.full((MASK_H, MASK_W), -1, dtype=np.int32)

    for rank, q in enumerate(confident):
        mask_prob  = 1 / (1 + np.exp(-mask_logits[0, q]))
        binary     = mask_prob > 0.5
        unassigned = (seg_map == -1) & binary

        if not unassigned.any():
            continue

        class_id   = pred_classes[q]
        class_name = ADE20K_CLASSES[class_id] if class_id < len(ADE20K_CLASSES) else f"cls{class_id}"
        category   = get_category(class_id)
        color      = CATEGORY_COLORS[category]

        seg_map[unassigned]   = rank
        color_seg[unassigned] = color

        print(f"  {category:<8s} | {class_name:<20s} | score={fg_probs[q]:.3f}")

    # Upsample to 512x512
    orig_img      = PILImage.open(image_path).convert("RGB").resize((OUT_SIZE, OUT_SIZE))
    orig_bgr      = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    color_seg_512 = cv2.resize(color_seg, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_NEAREST)
    overlay       = cv2.addWeighted(orig_bgr, 0.5, color_seg_512, 0.5, 0)

    # Legend (top-left, 4 fixed categories only)
    for i, (cat, color) in enumerate(CATEGORY_COLORS.items()):
        y = 25 + i * 30
        cv2.rectangle(overlay, (10, y - 16), (30, y + 4), color, -1)
        cv2.rectangle(overlay, (10, y - 16), (30, y + 4), (255, 255, 255), 1)
        cv2.putText(overlay, cat, (38, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("1. Original",       orig_bgr)
    cv2.imshow("2. Segmentation",   color_seg)
    cv2.imshow("3. OVERLAY RESULT", overlay)

    print("\n✅ Press any key to save...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("segmentation_result.jpg", overlay)
    print("💾 Saved: segmentation_result.jpg")

if __name__ == "__main__":
    visualize_onnx_cv2()