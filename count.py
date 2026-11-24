import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# Load a pre-trained YOLO model (YOLOv8 nano is fast and lightweight)
model = YOLO('yolov8n.pt')

# Hide the main tkinter root window so only the file dialog appears
root = tk.Tk()
root.withdraw()

# Open a file dialog to choose an image interactively
print("Please select an image file in the popup window...")
image_path = filedialog.askopenfilename(
    title="Select an Image for Person Counting",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

# Check if the user cancelled the dialog
if not image_path:
    print("No file selected. Exiting program.")
    exit()

print(f"Loaded image from: {image_path}")

# Load your target image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Run object detection on the image (class 0 corresponds to 'person' in COCO dataset)
results = model(image, classes=[0]) 

# Extract the bounding boxes for detected people
person_count = 0
for result in results:
    boxes = result.boxes
    for box in boxes:
        person_count += 1
        
        # Get coordinates for the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Draw a green rectangle around each detected person
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label each box with a number
        cv2.putText(image, str(person_count), (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the total count on the image canvas
text = f"Total Persons: {person_count}"
cv2.putText(image, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

print(f"Total people found: {person_count}")

# Show the result in a popup window and save it
cv2.imshow("Person Counter", image)
cv2.imwrite("output_counted.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()