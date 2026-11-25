import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from huggingface_hub import hf_hub_download
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Hide the main tkinter window so only the file dialog shows
root = tk.Tk()
root.withdraw()

# Ask the user to pick an image
image_path = filedialog.askopenfilename(
    title="Select an Image for Head Counting",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)
root.destroy()

# Exit if the dialog was cancelled
if not image_path:
    sys.exit("No file selected.")

# Load the image (imdecode handles non-ASCII paths on Windows)
image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
if image is None:
    sys.exit(f"Could not load {image_path}")

# Upscale so small heads get more pixels per tile
UPSCALE = 3.0 if image.shape[1] < 1200 else 2.0
big = cv2.resize(image, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

# Download the head detection weights (cached after the first run)
weights = hf_hub_download("wesjos/YOLO-head-detection", "v8m-head.pt")

# Load the head model through SAHI (use device="cpu" if you have no GPU)
detector = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=weights,
    confidence_threshold=0.35,
    device="cuda:0",
)

# Detect on overlapping tiles and merge duplicate boxes
result = get_sliced_prediction(
    cv2.cvtColor(big, cv2.COLOR_BGR2RGB),
    detector,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.25,
    overlap_width_ratio=0.25,
    postprocess_type="NMS",
    postprocess_match_metric="IOU",
    postprocess_match_threshold=0.4,
)

# Scale text size with image width
scale = max(image.shape[1] / 1000, 0.5)

# Count heads and draw their boxes
count = 0
for p in result.object_prediction_list:
    b = p.bbox
    w, h = b.maxx - b.minx, b.maxy - b.miny
    if not 0.6 < w / h < 1.6: # Skip weird boxes
        continue
    count += 1
    x1, y1, x2, y2 = [int(v / UPSCALE) for v in (b.minx, b.miny, b.maxx, b.maxy)]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Write the total on the image
cv2.putText(image, f"Total Heads: {count}", (30, int(50 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), max(int(3 * scale), 2))
print(f"Total heads found: {count}")

# Save the result next to the source image
out = os.path.join(os.path.dirname(image_path), "output_counted.jpg")
cv2.imwrite(out, image)

# Show the result in a resizable window
cv2.namedWindow("Head Counter", cv2.WINDOW_NORMAL)
cv2.imshow("Head Counter", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
