import cv2
import torch
from torchvision import models, transforms
import numpy as np
from collections import defaultdict
import tkinter as tk
from tkinter import simpledialog
import os




#pirma
zone_top_left = (370, 180)
zone_bottom_right = (1280, 960)

# trešā
# zone_top_left = (300, 205)
# zone_bottom_right = (1000, 1000)

#Otrā
# zone_top_left = (0, 205)
# zone_bottom_right = (700, 1000)

# Load a pre-trained Mask R-CNN model from torchvision
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the COCO classes indices for vehicles
vehicle_classes = [2, 3, 5, 7]  # Classes: car, motorcycle, bus, truck

# Video capture
cap = cv2.VideoCapture('video_samples/cam1_test.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save the output video
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Directory to save cropped images
output_dir = "vehicle_images"
os.makedirs(output_dir, exist_ok=True)

# Random color generator for masks
def get_random_color():
    return np.random.randint(0, 255, size=3).tolist()

# Vehicle tracking dictionary
tracked_vehicles = {}
clicked_vehicle_id = None

# Initialize Tkinter for input prompts
root = tk.Tk()
root.withdraw()

# Function to generate a unique file name to avoid overwriting
def get_unique_filename(directory, base_filename, extension):
    counter = 1
    while True:
        filename = f"{base_filename}_{counter}{extension}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1


# Mouse callback function to assign vehicle ID
def assign_vehicle_id(event, x, y, flags, param):
    global clicked_vehicle_id, tracked_vehicles
    #print(len(tracked_vehicles))
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is within any vehicle's mask
        for vehicle_id, vehicle in tracked_vehicles.items():
            #print(vehicle_id)
            if vehicle['mask'][y, x]:
                #print("yes")

                clicked_vehicle_id = vehicle_id
                # Prompt for ID input, showing the existing ID if available
                initial_id = tracked_vehicles[vehicle_id]['id'] or ""
                new_id = simpledialog.askstring("Input", f"Enter vehicle ID for vehicle {vehicle_id}:", initialvalue=initial_id)
                if new_id is not None:
                    tracked_vehicles[vehicle_id]['id'] = new_id

                    x1, y1, x2, y2 = vehicle['bbox']
                    cropped_img = original_frame_forCrop[y1:y2, x1:x2]


                    # Generate a unique filename to avoid overwriting
                    save_path = get_unique_filename(output_dir, new_id, '.png')

                    cv2.imwrite(save_path, cropped_img)
                    print(f"Saved cropped image as {save_path}")                    
                break

# Create window and set mouse callback
cv2.namedWindow('Masked Video')
cv2.setMouseCallback('Masked Video', assign_vehicle_id)

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Clone the original frame for reference
    original_frame = frame.copy()

    original_frame_forCrop = frame.copy()

    # Crop the frame to the designated zone
    #cropped_frame = frame[zone_top_left[1]:zone_bottom_right[1], zone_top_left[0]:zone_bottom_right[0]]

    # Convert frame to tensor and normalize
    img = transform(frame).unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        predictions = model(img)

    # Process predictions
    masks = predictions[0]['masks'].numpy()
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()

    # Set a confidence threshold (e.g., 0.5)
    threshold = 0.5

    # Update vehicle tracking information
    new_tracked_vehicles = defaultdict(dict)

    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0] > 0.5  # Binarize the mask
            
            # Calculate the bounding box from the mask
            y_indices, x_indices = np.where(mask)
            x1, y1, x2, y2 = min(x_indices), min(y_indices), max(x_indices), max(y_indices)

            # # Adjust bounding box coordinates to match the original frame
            # x1 += zone_top_left[0]
            # y1 += zone_top_left[1]
            # x2 += zone_top_left[0]
            # y2 += zone_top_left[1]

            # # Adjust mask to the original frame size
            # full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            # full_mask[y1:y2, x1:x2] = mask[y1 - zone_top_left[1]:y2 - zone_top_left[1], x1 - zone_top_left[0]:x2 - zone_top_left[0]]


            # Compute centroid of the mask for tracking
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            if centroid[0] > zone_top_left[0] and centroid[0] < zone_bottom_right[0] and centroid[1] < zone_bottom_right[1] and centroid[1] > zone_top_left[1] :
            
                # Find the closest existing vehicle to update or create a new one
                vehicle_id = None
                min_distance = float('inf')
                for vid, vehicle in tracked_vehicles.items():
                    dist = np.linalg.norm(np.array(vehicle['centroid']) - np.array(centroid))
                    if dist < min_distance:
                        min_distance = dist
                        vehicle_id = vid

                if min_distance > 50:  # If no close vehicle is found, assign a new ID
                    vehicle_id = len(tracked_vehicles) + 1

                # Update tracking information
                tracked_vehicles[vehicle_id] = {
                    'mask': mask,
                    'bbox': (x1, y1, x2, y2),
                    'centroid': centroid,
                    'color': tracked_vehicles.get(vehicle_id, {}).get('color', get_random_color()),
                    'id': tracked_vehicles.get(vehicle_id, {}).get('id', None)
                }
                # print(vehicle_id)
                # print(len(tracked_vehicles))

                #print(new_tracked_vehicles)

                # Draw bounding box and mask on the original frame
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), tracked_vehicles[vehicle_id]['color'], 2)
                if tracked_vehicles[vehicle_id]['id'] is not None:
                    cv2.putText(original_frame, f'ID: {tracked_vehicles[vehicle_id]["id"]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                frame[mask] = tracked_vehicles[vehicle_id]['color']
                #frame[full_mask > 0] = tracked_vehicles[vehicle_id]['color']

    print(len(tracked_vehicles))
    # Update tracked vehicles
    #tracked_vehicles = tracked_vehicles

    # Display the frame with masks in one window
    cv2.imshow('Masked Video', frame)
    
    # Display the original frame with bounding boxes and IDs in another window
    cv2.imshow('Original Video', original_frame)

    # Write the masked frame to the output video
    out.write(frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
