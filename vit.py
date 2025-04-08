from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
import csv

# ----------------------------
# Config & Paths
# ----------------------------

RESIZED_DIR = "/scratch/network/jy3107/IW/resized_inputs"
OUTPUT_DIR = "/scratch/network/jy3107/IW/outputs"
CSV_OUTPUT = "/scratch/network/jy3107/IW/vit_captions.csv"
HF_CACHE_DIR = "/scratch/network/jy3107/hf_cache"

# ----------------------------
# Model Setup
# ----------------------------

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning", cache_dir=HF_CACHE_DIR
)
feature_extractor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning", cache_dir=HF_CACHE_DIR
)
tokenizer = AutoTokenizer.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning", cache_dir=HF_CACHE_DIR
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generation settings
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# ----------------------------
# Helper Functions
# ----------------------------

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def get_all_image_paths(directory):
    return [os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

def run_captioning_and_save_csv(resized_dir, output_dir, csv_path):
    all_rows = []

    # Caption resized inputs (and adjust name)
    resized_images = get_all_image_paths(resized_dir)
    print(f"Found {len(resized_images)} resized input images.")

    for img_path in resized_images:
        try:
            caption = predict_step([img_path])[0]
            filename = os.path.basename(img_path)
            id_part = filename.rsplit("_resized", 1)[0]
            cleaned_name = f"{id_part}.jpg"
            all_rows.append([cleaned_name, caption])
            print(f"Captioned {cleaned_name}: {caption}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Caption output images (no name changes)
    output_images = get_all_image_paths(output_dir)
    print(f"Found {len(output_images)} output images.")

    for img_path in output_images:
        try:
            caption = predict_step([img_path])[0]
            filename = os.path.basename(img_path)
            all_rows.append([filename, caption])
            print(f"Captioned {filename}: {caption}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Save to CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Caption"])
        writer.writerows(all_rows)

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    run_captioning_and_save_csv(RESIZED_DIR, OUTPUT_DIR, CSV_OUTPUT)
