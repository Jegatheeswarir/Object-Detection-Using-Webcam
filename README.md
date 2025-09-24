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

import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO output
    outputs = net.forward(output_layers)
    
    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
## OUTPUT:
<img width="746" height="605" alt="image" src="https://github.com/user-attachments/assets/6099a3e0-0e4c-4e50-826d-c8b6e35fdde4" />

RESULT:
Thus, the Python program for real-time object detection using a webcam has been successfully executed.

