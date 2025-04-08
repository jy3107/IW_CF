import sys
import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import supervision as sv
# Add repository utils directory to system path
sys.path.append('/scratch/network/jy3107/IW/text-guided-mask-for-inpainting')

from utils.florence import load_florence_model, run_florence_inference
from utils.sam import load_sam_image_model, run_sam_inference

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
florence_model, florence_processor = load_florence_model(device)
sam_predictor = load_sam_image_model(device)

# ======= Core Inference Function =======

def open_vocabulary_detection(image, prompt):
    print(f"Running Florence with prompt: '{prompt}'")
    task = "<OPEN_VOCABULARY_DETECTION>"
    _, response = run_florence_inference(florence_model, florence_processor, device, image, task, prompt)

    if task in response:
        labels = response[task].get('labels', [])
        bboxes = response[task].get('bboxes', [])
        return [{"label": labels[i], "bbox": bboxes[i]} for i in range(len(labels))]
    return []

# ======= Mask Generation =======

def generate_masks(image, response, image_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Extract bounding boxes from the response
    boxes = response["<OPEN_VOCABULARY_DETECTION>"]["bboxes"][0]  # List of bounding boxes

    # Convert the list to a numpy array for easier handling
    boxes = np.array(boxes)
    boxes = boxes.reshape(1, 4)

    # Check if the bounding boxes array is non-empty
    if boxes.size > 0:  # This ensures there are bounding boxes
        # Create a supervision.Detections object using the boxes
        detections = sv.Detections(xyxy=boxes)
    else:
        raise ValueError("No bounding boxes found in Florence response. Check the response.")
    print("about ot run sam inference")
    detections_with_mask = run_sam_inference(sam_predictor, image, detections)
    mask = detections_with_mask.mask.astype(np.uint8) * 255
    mask_image = np.squeeze(mask)

    output_path = os.path.join(save_dir, f"{image_name}_mask.jpg")
    cv2.imwrite(output_path, mask_image)
    print(f"Saved mask: {output_path}")


# ======= Image Processing Pipeline =======

def get_prompt_mapping(metadata_path):
    print(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    # Use the ID column as the key, and Object Label as the value
    return dict(zip(df['ID'], df['Object Label']))


def find_all_images(root_dir, exts={'.jpg', '.jpeg', '.png'}):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in exts:
                full_path = os.path.join(dirpath, fname)
                image_paths.append(full_path)
    return image_paths

def process_images(image_paths, prompt_map, input_root):
    for img_path in image_paths:
        filename = os.path.splitext(os.path.basename(img_path))[0]  # e.g., DW0001_masked → DW0001_masked
        # Match filename by prefixing against known IDs
        matched_id = next((id for id in prompt_map if filename.startswith(id)), None)

        if not matched_id:
            print(f"No matching ID found for {filename}, skipping.")
            continue
        prompt = prompt_map[matched_id]
        print(prompt)

        try:
            image = Image.open(img_path).convert("RGB")

            task = "<OPEN_VOCABULARY_DETECTION>"
            text_prompt = f"{prompt}"
            generated_text, response = run_florence_inference(
                florence_model,
                florence_processor,
                device,
                image,
                task,
                text_prompt
            )

            print("Florence output:", generated_text)
            print("Florence response:", response)

            save_dir = "/scratch/network/jy3107/IW/mask_images"
            generate_masks(image, response, matched_id, save_dir)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


# ======= Main =======

if __name__ == "__main__":
    input_root = "/scratch/network/jy3107/IW/inputs"
    metadata_csv = "/scratch/network/jy3107/IW/metadata.csv"

    prompt_map = get_prompt_mapping(metadata_csv)
    image_paths = find_all_images(input_root)
    process_images(image_paths, prompt_map, input_root)
