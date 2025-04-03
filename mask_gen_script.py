import sys
import os

# Add repository utils directory to system path
sys.path.append('/scratch/network/jy3107/IW/text-guided-mask-for-inpainting')

import torch
import numpy as np
import cv2
# import nltk
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')
# # Ensure NLTK resources are available
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils.florence import load_florence_model, run_florence_inference
from utils.sam import load_sam_image_model, run_sam_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
florence_model, florence_processor = load_florence_model(device)
sam_predictor = load_sam_image_model(device)

lemmatizer = WordNetLemmatizer()

def clean_labels_and_preserve_bboxes(labels, bboxes):
    print("clean labels and bboxes start")
    label_to_bbox = {}
    for label, bbox in zip(labels, bboxes):
        words = word_tokenize(label.lower())
        words = [word for word in words if word not in ["the", "a", "an"]]
        lemmatized_label = " ".join(words)
        # [lemmatizer.lemmatize(word, pos="n") for word in words]
        if lemmatized_label not in label_to_bbox:
            label_to_bbox[lemmatized_label] = bbox

    return list(label_to_bbox.keys()), list(label_to_bbox.values())

def caption_img(image):
    print("caption_img start")
    task = "<MORE_DETAILED_CAPTION>"
    _, response = run_florence_inference(florence_model, florence_processor, device, image, task, "")
    
    if task in response:
        return response[task]
    return None

def ground_caption(image, output):
    print("ground_caption start")
    task2 = "<CAPTION_TO_PHRASE_GROUNDING>"
    _, response = run_florence_inference(florence_model, florence_processor, device, image, task2, output)
    
    if task2 in response:
        res = response[task2].get('labels', [])
        bb = response[task2].get('bboxes', [])
        return clean_labels_and_preserve_bboxes(res, bb)
    return [], []

def compute_bb(cleaned_bboxes, cleaned_labels):
    print("compute_bb start")
    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in cleaned_bboxes]
    sort_idxs = np.argsort(areas)
    sample_amt = 3
    smallest_idxs = sort_idxs[:sample_amt]

    return [{'label': cleaned_labels[i], 'bbox': cleaned_bboxes[i]} for i in smallest_idxs]

def mask_gen(image, sample_obj, name):
    print("mask_gen start")
    obj_length = len(sample_obj)
    boxes = [box['bbox'] for box in sample_obj]
    objs = [obj['label'] for obj in sample_obj]

    output_dir = f"/scratch/network/jy3107/IW/mask_images/{name}"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(obj_length):
        bb = np.array(boxes[i]).reshape(1, 4)
        if bb.size > 0:
            detections = sv.Detections(xyxy=bb)
            detections_with_mask = run_sam_inference(sam_predictor, image, detections)
            mask = detections_with_mask.mask.astype(np.uint8) * 255
            mask_image = np.squeeze(mask)
            
            cv2.imwrite(f"{output_dir}/{name}_{objs[i]}.jpg", mask_image)
            print("image done")

def process_images(input_dir):
    for filename in os.listdir(input_dir):
        name, extension = os.path.splitext(filename)
        input_path = os.path.join(input_dir, filename)
        
        try:
            image = Image.open(input_path).convert("RGB")
            output = caption_img(image)
            if output:
                cleaned_labels, cleaned_bboxes = ground_caption(image, output)
                sample_obj = compute_bb(cleaned_bboxes, cleaned_labels)
                mask_gen(image, sample_obj, name)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "/scratch/network/jy3107/IW/test"
    process_images(input_dir)