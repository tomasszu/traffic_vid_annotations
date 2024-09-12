import cv2
import torch
import numpy as np
from torchvision import models, transforms

# Load a pre-trained DeepLabV3 model for semantic segmentation
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Video capture
cap = cv2.VideoCapture('video_samples/cam1_test.mp4')

# Define the codec and create VideoWriter object to save the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Tracking dictionary
tracked_objects = {}
object_id_counter = 0

# Mouse callback function for flood fill and bounding box assignment
def on_mouse_click(event, x, y, flags, param):
    global tracked_objects, object_id_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        # Find the class of the clicked pixel
        class_id = output.argmax(0)[y, x].item()

        # Create a mask to identify the region
        mask = np.zeros(output.shape[1:], dtype=np.uint8)
        mask[output.argmax(0) == class_id] = 255

        # Perform flood fill from the clicked point
        mask_floodfill = mask.copy()
        h, w = mask_floodfill.shape
        floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_floodfill, floodfill_mask, (x, y), 255)

        # Get the bounding box of the filled area
        contours, _ = cv2.findContours(mask_floodfill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            tracked_objects[object_id_counter] = {'bbox': (x, y, x+w, y+h), 'mask': mask_floodfill}
            object_id_counter += 1

# Create window and set mouse callback
cv2.namedWindow('Segmentation')
cv2.setMouseCallback('Segmentation', on_mouse_click)

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Clone the original frame for reference
    original_frame = frame.copy()

    # Preprocess the frame for segmentation
    input_tensor = transform(frame).unsqueeze(0)

    # Perform segmentation
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output = output.detach().cpu().numpy()

    # Generate segmentation map
    seg_map = output.argmax(0)

    # Visualize segmentation map
    seg_frame = np.zeros_like(frame)
    seg_frame[seg_map == seg_map] = [0, 255, 0]  # Visualize the segmented class in green

    # Apply tracked objects and draw bounding boxes
    for object_id, obj in tracked_objects.items():
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(original_frame, f'ID: {object_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display the segmentation map
    cv2.imshow('Segmentation', seg_frame)

    # Display the original frame with bounding boxes
    cv2.imshow('Original', original_frame)

    # Write the frame with bounding boxes to the output video
    out.write(original_frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
