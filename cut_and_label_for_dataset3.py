import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob

from collections import defaultdict
import tkinter as tk
from tkinter import simpledialog

import sys

sys.path.append('segment-anything-2')
from sam2.build_sam import build_sam2_video_predictor


# select the device for computation
if torch.cuda.is_available():    # TE ORIGINALI IR BEZ "NOT", VNK CUDA NEIET
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

checkpoint = "/home/tomass/tomass/traffic_vid_annotations/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

# Directory for saving cropped images
vehicle_images_dir = "vehicle_images"
os.makedirs(vehicle_images_dir, exist_ok=True)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def onclick(event):

    #global ann_obj_id
    global tracked_ids

    ax = plt.gca()


    if event.inaxes:
        # Get the x, y coordinates of the click
        x, y = int(event.xdata), int(event.ydata)

        # Define label based on mouse button: 1 (left-click) for positive, 3 (right-click) for negative
        label = 1 if event.button == 1 else 0

        # Add the point and label to the lists
        points.append([x, y])
        labels.append(label)

        # Convert to NumPy arrays
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, np.int32)

        #Ask for ID
        some_id = simpledialog.askstring("Input", f"Enter vehicle ID for clicked vehicle", initialvalue=1)

        # Clear the axes and re-display the image
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"frame {ann_frame_idx}")


        # Show the points and masks
        show_points(points_np, labels_np, ax)
        
        try:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=int(some_id),
                points=points,
                labels=labels,
            )
        except Exception as e:
            print(e)
            print("\nResetting all tracked objects, please click all trackable objects again!\n")
            predictor.reset_state(inference_state)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=int(some_id),
                points=points,
                labels=labels,
            )

        for i, out_obj_id in enumerate(out_obj_ids):
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        #show_mask((out_mask_logits[0] > 0.0).astype(np.uint8), ax, obj_id=out_obj_ids[0])

        # Refresh the plot
        plt.draw()

def get_bounding_box(mask):
    """Calculate bounding box from a binary mask."""

    mask = np.squeeze(mask)
    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8)
    
    # Find non-zero values in the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Check if there are any non-zero rows and columns
    if np.any(rows) and np.any(cols):
        # Get the min and max row indices where mask is non-zero
        y_min, y_max = np.where(rows)[0][[0, -1]]
        # Get the min and max column indices where mask is non-zero
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [x_min, y_min, x_max, y_max]
    
    return None  # If no non-zero pixels are found in the mask

def show_bboxes_in_second_window(video_segments, frame_idx):
    """Display bounding boxes in a second window and save cropped images."""

    
    # Open a new figure for the bounding boxes
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.set_title(f"Bounding Boxes for frame {frame_idx}")

    # Load the image for this frame
    img = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
    ax2.imshow(img)

    bbox_data = {}

    # Loop through all the objects in this frame
    if frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            bbox = get_bounding_box(out_mask)
            if bbox is not None:
                # Draw the bounding box on frame and save cropped image
                x_min, y_min, x_max, y_max = bbox

                rect = plt.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min, 
                    edgecolor='red', facecolor='none', lw=2
                )
                ax2.add_patch(rect)

                # Add the object ID label above the bounding box
                ax2.text(x_min, y_min - 10, f"ID: {out_obj_id}", 
                         color='yellow', fontsize=12, fontweight='bold')
                bbox_data[out_obj_id] = bbox
                
    fig2.canvas.mpl_connect('button_press_event', lambda event: on_click_bboxes(event, bbox_data))

    # Show the bounding boxes window
    # plt.show(block=False)

def on_click_bboxes(event, bbox_data):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        for obj_id, bbox in bbox_data.items():
            x_min, y_min, x_max, y_max = bbox
            if x_min <= x <= x_max and y_min <= y <= y_max:
                img = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                cropped_img_name = f"id-{obj_id}_{ann_frame_idx}.png"
                cropped_img_path = os.path.join(vehicle_images_dir, cropped_img_name)
                cropped_img.save(cropped_img_path)
                print(f"Cropped image saved: {cropped_img_path}")
                break


def on_keypress(event):
    """Callback function to handle keypress events."""
    global keyboardClick
    if event.key == 'q':
        plt.close('all')  # Close all matplotlib windows
        sys.exit("Program exited.")  # Exit the program if 'q' is pressed
    keyboardClick = True  # Set the flag to continue if any other key is pressed

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/home/tomass/tomass/traffic_vid_annotations/video_samples/cam1_test_jpg"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)

#Initialize frame idx and first object idx
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

tracked_ids = []

prompts = {}  # hold all the clicks we add for visualization

# Initialize lists to store points and labels
points = []
labels = []
video_segments = {}  # video_segments contains the per-frame segmentation results

# Iterate through each file
for frame_name in frame_names:
    img = Image.open(os.path.join(video_dir, frame_name))

    # Window 1. for segmentation masks
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"frame {ann_frame_idx}")
    ax.imshow(img)

    if ann_frame_idx > 0 and video_segments:
        for out_obj_id, out_mask in video_segments[ann_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)



    # Connect the click event to the function
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', on_keypress)  # Connect key press event



    # print(inference_state.keys())
    # print(inference_state['obj_ids'])

    tracked_ids = inference_state['obj_ids']
    
    if (points and labels) or video_segments:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state = inference_state, start_frame_idx= ann_frame_idx, max_frame_num_to_track = 1):
            # if out_frame_idx >= ann_frame_idx + 2: break  # Netaisam predictions taalaak par 2 frames uz prieksu

            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    ann_frame_idx = ann_frame_idx + 1

    points.clear()
    labels.clear()

    if ann_frame_idx > 0 and video_segments:
        # Display the bounding boxes in the second window
        show_bboxes_in_second_window(video_segments, ann_frame_idx - 1)


    plt.show(block=False)
    # input("Press Enter to continue...")
    keyboardClick=False

    # while keyboardClick != True:
    #     keyboardClick=plt.waitforbuttonpress()

    while not keyboardClick:  # Wait for any key press to continue
        plt.pause(0.1)
    
    plt.close("all")
