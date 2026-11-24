import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# Load a pre-trained YOLO model (YOLOv8 small is a better accuracy/speed balance than nano)
model = YOLO('yolov8s.pt')

# Hide the main tkinter root window so only the file dialog appears
root = tk.Tk()
root.withdraw()

# Open a file dialog to choose an image interactively
print("Please select an image file in the popup window...")
image_path = filedialog.askopenfilename(
    title="Select an Image for Person Counting",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)
root.destroy()

# Check if the user cancelled the dialog
if not image_path:
    print("No file selected. Exiting program.")
    sys.exit()

print(f"Loaded image from: {image_path}")

# Load your target image (imdecode handles non-ASCII paths on Windows)
image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    sys.exit()

# Run object detection on the image (class 0 corresponds to 'person' in COCO dataset)
results = model(image, classes=[0], conf=0.3, imgsz=1280, verbose=False)

# Scale drawing sizes to the image width so text is readable on large photos
scale = max(image.shape[1] / 1000, 0.5)

# Extract the bounding boxes for detected people
person_count = 0
for result in results:
    boxes = result.boxes
    for box in boxes:
        person_count += 1

        # Get coordinates for the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw a green rectangle around each detected person
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), max(int(2 * scale), 2))

        # Label each box with a number (clamped so it stays on-screen)
        cv2.putText(image, str(person_count), (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 255, 0), max(int(2 * scale), 1))

# Display the total count on the image canvas
text = f"Total Persons: {person_count}"
cv2.putText(image, text, (30, int(50 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), max(int(3 * scale), 2))

print(f"Total people found: {person_count}")

# Save the result next to the source image
output_path = os.path.join(os.path.dirname(image_path), "output_counted.jpg")
cv2.imwrite(output_path, image)

# Show the result in a resizable popup window
cv2.namedWindow("Person Counter", cv2.WINDOW_NORMAL)
cv2.imshow("Person Counter", image)

cv2.waitKey(0)
cv2.destroyAllWindows()