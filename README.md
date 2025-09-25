## Object Detection Using Webcam
## AIM:
To write a Python program to perform real-time object detection using a webcam.

## PROCEDURE:
STEP-1: Load the pre-trained YOLOv4 network (.weights and .cfg) using cv2.dnn.readNet().

STEP-2: Read class labels (COCO dataset) from the coco.names file.

STEP-3: Get the output layer names from the YOLO network using getLayerNames() and getUnconnectedOutLayers().

STEP-4: Start webcam video capture using cv2.VideoCapture(0).

STEP-5: Process each frame:

Convert the frame to a YOLO-compatible input using cv2.dnn.blobFromImage().
Pass the blob into the network (net.setInput()), run a forward pass to get detections (net.forward()).
Parse the output to extract bounding boxes, confidence scores, and class IDs for detected objects.
STEP-6: Use Non-Maximum Suppression (NMS) to remove overlapping bounding boxes and retain the best ones.

STEP-7: Draw bounding boxes and labels on detected objects using cv2.rectangle() and cv2.putText().

STEP-8: Show the processed video frames with object detections using cv2.imshow().

STEP-9: Exit the loop if the 'q' key is pressed.

STEP-10: Release the video capture and close any OpenCV windows (cap.release() and cv2.destroyAllWindows()).

## PROGRAM:
**NAME**: Jegatheeswari R 

**REG.NO**: 212223230092
```
import cv2
import numpy as np

# Function to automatically find a working webcam
def get_working_camera(max_cams=5):
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use CAP_V4L2 on Linux
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
        cap.release()
    return None

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Initialize webcam
cap = get_working_camera()
if cap is None:
    print("Error: Could not find a working webcam.")
    exit()

print("Webcam detected and opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: Failed to grab frame.")
        continue

    # Optional: resize for faster processing
    frame = cv2.resize(frame, (640, 480))
    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```
## OUTPUT:
<img width="1920" height="1080" alt="Screenshot 2025-09-25 160730" src="https://github.com/user-attachments/assets/353d8351-4158-487f-a146-d6033543afd9" />

## RESULT:
Thus, the Python program for real-time object detection using a webcam has been successfully executed.

